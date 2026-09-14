"""Round-based exact ReN distillation on the repository's HF/PEFT stack.

Inference workers are independent. DDP is used only for student updates.
Dense targets are reconstructed from frozen hidden states in position chunks;
there is no top-k approximation of the Student background distribution.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from lulu.data import build_prompt_views, load_prepared_jsonl, reasoning_token_mask
from lulu.objective import build_cached_target, forward_kl
from lulu.paths import PROJECT_ROOT

METHODS = ('ren_opd', 'vanilla_opd', 'opsd', 'causal_topk', 'union_topk')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('run', 'init', 'collect', 'teacher', 'update'), default='run')
    p.add_argument('--backend', choices=('persistent', 'staged'), default='persistent',
                   help='Resident GPU roles and in-memory pipeline, or legacy subprocess phases')
    p.add_argument('--student-gpus', default='auto', help='Persistent backend: disjoint GPU IDs for Student DDP')
    p.add_argument('--hindsight-gpus', default='auto', help='Persistent backend: one GPU for synchronized privileged Student')
    p.add_argument('--teacher-gpus', default='auto', help='Persistent backend: GPU IDs in one true tensor-parallel group')
    p.add_argument('--save-every', type=int, default=20, help='Retain every N updates; persistent latest is saved each update')
    p.add_argument('--worker-timeout', type=float, default=1800, help='Seconds without any worker progress before failing')
    p.add_argument('--model', default='Qwen/Qwen3-1.7B')
    p.add_argument('--teacher-model', default='Qwen/Qwen3-32B')
    p.add_argument('--train-data', required=True, help='Prepared train.jsonl; use prepare_lulu_data.py')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--method', choices=METHODS, default='ren_opd')
    p.add_argument('--rounds', type=int, default=100)
    p.add_argument('--round', type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument('--global-batch-prompts', type=int, default=64)
    p.add_argument('--rollouts-per-prompt', type=int, default=1)
    p.add_argument('--rollout-batch-size', type=int, default=4)
    p.add_argument('--score-batch-size', type=int, default=1)
    p.add_argument('--train-micro-batch-size', type=int, default=1)
    p.add_argument('--update-passes', type=int, default=1, help='Optimizer steps per refreshed rollout batch')
    p.add_argument('--top-k', type=int, default=32)
    p.add_argument('--max-new-tokens', type=int, default=8192)
    p.add_argument('--max-prompt-tokens', type=int, default=4096)
    p.add_argument('--max-sequence-tokens', type=int, default=16384)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--logit-chunk-size', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--lora-rank', type=int, default=16, help='0 for full parameter training')
    p.add_argument('--lora-alpha', type=int, default=32)
    p.add_argument('--lora-target-modules', default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    p.add_argument('--dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpus', default='auto', help='Visible GPU IDs; all visible GPUs by default')
    p.add_argument('--teacher-gpus-per-worker', type=int, default=2)
    p.add_argument('--teacher-memory-gib', type=int, default=55, help='Per GPU HF model dispatch budget')
    p.add_argument('--teacher-workers', type=int, default=0, help='0 uses all disjoint GPU groups')
    p.add_argument('--worker-rank', type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument('--worker-world', type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--keep-round-cache', action='store_true')
    p.add_argument('--resume', action='store_true', help='Resume only from a completed round checkpoint')
    p.add_argument('--dry-run', action='store_true', help='Print phase plan without loading data/models')
    p.add_argument('--cpu', action='store_true', help='Tiny-model integration tests only')
    return p


def validate_args(a):
    for name in ('rounds', 'global_batch_prompts', 'rollouts_per_prompt', 'rollout_batch_size',
                 'score_batch_size', 'train_micro_batch_size', 'update_passes', 'top_k',
                 'max_new_tokens', 'max_prompt_tokens', 'max_sequence_tokens', 'logit_chunk_size',
                 'teacher_gpus_per_worker', 'teacher_memory_gib', 'save_every', 'worker_timeout'):
        if getattr(a, name) < 1:
            raise ValueError(f'{name} must be positive')
    if a.temperature <= 0 or not 0 < a.top_p <= 1:
        raise ValueError('Require temperature > 0 and 0 < top_p <= 1')
    if a.lora_rank < 0 or a.teacher_workers < 0 or a.learning_rate <= 0:
        raise ValueError('Invalid LoRA rank, teacher worker count or learning rate')
    unsupported_heads = {'lm_head', 'embed_tokens', 'embed_in', 'embed_out', 'wte',
                         'word_embeddings', 'tok_embeddings', 'embeddings'}
    targets = {name.strip().rsplit('.', 1)[-1] for name in a.lora_target_modules.split(',')}
    if a.lora_rank and targets & unsupported_heads:
        raise ValueError('LoRA targets must exclude output heads and embeddings; frozen head reconstruction requires a plain Linear')
    if a.max_sequence_tokens < a.max_prompt_tokens + a.max_new_tokens:
        raise ValueError('max_sequence_tokens must cover max_prompt_tokens + max_new_tokens')
    if a.cpu and a.dtype != 'float32':
        raise ValueError('--cpu requires --dtype float32')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def load_tensor_file(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def paths(a):
    root = Path(a.output_dir).resolve()
    return root, root / 'round_cache' / f'round_{a.round:04d}', root / 'checkpoints' / f'round_{a.round:04d}'


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(a):
    if a.cpu:
        return torch.device('cpu')
    local = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local)
    return torch.device('cuda', local)


def dtype_for(a):
    return getattr(torch, a.dtype)


def base_model(model):
    return model.get_base_model() if hasattr(model, 'peft_config') else model


def load_tokenizer(model_id):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise ValueError('Tokenizer needs an EOS or padding token')
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = 'left'
    return tok


def disable_dropout(model):
    """Keep update forwards deterministic like the frozen inference snapshot.

    Qwen attention uses functional dropout with a numeric module attribute,
    while other architectures commonly use nn.Dropout or config attributes.
    Disabling only nn.Dropout would leave those policies stochastic in train().
    """
    names = ('dropout', 'attention_dropout', 'hidden_dropout', 'activation_dropout',
             'hidden_dropout_prob', 'attention_probs_dropout_prob', 'attn_pdrop',
             'resid_pdrop', 'embd_pdrop', 'classifier_dropout')
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        for name in names:
            if isinstance(getattr(module, name, None), (int, float)):
                setattr(module, name, 0.0)
    for name in names:
        if isinstance(getattr(model.config, name, None), (int, float)):
            setattr(model.config, name, 0.0)


def load_student(a, checkpoint_dir=None, trainable=False):
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    source = str(checkpoint_dir) if checkpoint_dir and not a.lora_rank else a.model
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype_for(a),
        attn_implementation='sdpa', trust_remote_code=False, low_cpu_mem_usage=True)
    if checkpoint_dir and a.lora_rank:
        model = PeftModel.from_pretrained(model, str(checkpoint_dir), is_trainable=trainable)
    model.to(device_for(a))
    if not trainable:
        model.requires_grad_(False).eval()
    else:
        model.train()
        # Frozen snapshot and update forward must use the same stochastic policy.
        disable_dropout(model)
        if a.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            if hasattr(model, 'enable_input_require_grads'):
                model.enable_input_require_grads()
        model.config.use_cache = False
    return model


def check_vocab(student_tok, teacher_tok, student_size, teacher_size):
    if student_size != teacher_size or student_tok.get_vocab() != teacher_tok.get_vocab():
        raise ValueError('Token-distribution distillation requires identical token IDs/vocabulary and output size')


def head_state(model):
    head = base_model(model).get_output_embeddings()
    return {'weight': head.weight.detach().cpu(),
            'bias': None if getattr(head, 'bias', None) is None else head.bias.detach().cpu()}


def load_head(path, device):
    state = load_tensor_file(path)
    weight = state['weight'].to(device)
    head = nn.Linear(weight.shape[1], weight.shape[0], bias=state['bias'] is not None,
                     device=device, dtype=weight.dtype)
    head.weight = nn.Parameter(weight, requires_grad=False)
    if state['bias'] is not None:
        head.bias = nn.Parameter(state['bias'].to(device), requires_grad=False)
    return head.eval()


def padded_batch(sequences, pad_id, device, *, left=False):
    lengths = [len(s) for s in sequences]
    width = max(lengths)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences):
        start = width - len(seq) if left else 0
        ids[i, start:start+len(seq)] = torch.tensor(seq, device=device)
        mask[i, start:start+len(seq)] = 1
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    return dict(input_ids=ids, attention_mask=mask, position_ids=positions)


def selected_hidden(model, tok, records, view, a):
    """Return only hidden states that predict masked response tokens."""
    sequences = [r[view] + r['response_ids'] for r in records]
    if max(map(len, sequences)) > a.max_sequence_tokens:
        raise ValueError('Scoring context exceeds max_sequence_tokens; refusing truncation')
    device = base_model(model).get_input_embeddings().weight.device
    batch = padded_batch(sequences, tok.pad_token_id, device)
    hidden = base_model(model).get_decoder()(**batch, use_cache=False, return_dict=True).last_hidden_state
    result = []
    for i, r in enumerate(records):
        index = torch.tensor(r['positions'], dtype=torch.long, device=hidden.device) + len(r[view]) - 1
        result.append(hidden[i].index_select(0, index))
    return result


def schedule(n, batch, round_index, seed):
    """Balanced deterministic stream without allocating all previous rounds."""
    if n <= 0:
        raise ValueError('Training data is empty')
    out, offset = [], round_index * batch
    while len(out) < batch:
        epoch, within = divmod(offset, n)
        ids = list(range(n))
        random.Random(seed + epoch * 1000003).shuffle(ids)
        take = min(batch - len(out), n - within)
        out.extend(ids[within:within+take])
        offset += take
    return out


def save_checkpoint(model, tok, path, optimizer=None, metadata=None):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tok.save_pretrained(tmp)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), tmp / 'optimizer.pt')
    atomic_json(tmp / 'lulu_state.json', metadata or {})
    if path.exists():
        raise FileExistsError(path)
    tmp.rename(path)


def init_phase(a):
    from peft import LoraConfig, get_peft_model
    _, _, current = paths(a)
    seed_all(a.seed)
    model = load_student(a)
    if a.lora_rank:
        model = get_peft_model(model, LoraConfig(r=a.lora_rank, lora_alpha=a.lora_alpha,
            lora_dropout=0.0, target_modules=a.lora_target_modules.split(','),
            bias='none', task_type='CAUSAL_LM'))
    tok = load_tokenizer(a.model)
    save_checkpoint(model, tok, current, metadata={'completed_rounds': 0, 'model': a.model})


def collect_phase(a):
    rank, world = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
    _, cache, current = paths(a)
    cache.mkdir(parents=True, exist_ok=True)
    seed_all(a.seed + 100003 * a.round + rank)
    tok = load_tokenizer(a.model)
    model = load_student(a, current)
    rows = load_prepared_jsonl(a.train_data)
    chosen = schedule(len(rows), a.global_batch_prompts, a.round, a.seed)
    jobs = [(i * a.rollouts_per_prompt + j, rows[k]) for i, k in enumerate(chosen)
            for j in range(a.rollouts_per_prompt)]
    jobs = jobs[rank::world]
    if rank == 0:
        atomic_torch(cache / 'student_head.pt', head_state(model))
    eos = model.generation_config.eos_token_id or tok.eos_token_id
    eos_set = set(eos if isinstance(eos, list) else [eos])
    total_reason, total_tokens = 0, 0
    for start in range(0, len(jobs), a.rollout_batch_size):
        chunk = jobs[start:start+a.rollout_batch_size]
        records = []
        for index, row in chunk:
            views = build_prompt_views(tok, row['messages'], row['gold_answer'], enable_thinking=True)
            cp, hp = views['causal_prompt_ids'], views['hindsight_prompt_ids']
            if max(len(cp), len(hp)) > a.max_prompt_tokens:
                raise ValueError(f'Prompt {row["id"]} exceeds max_prompt_tokens; increase budget')
            if max(len(cp), len(hp)) + a.max_new_tokens > a.max_sequence_tokens:
                raise ValueError('Prompt + requested response exceeds sequence budget')
            records.append({'index': index, 'source_id': row['id'],
                'causal_prompt_ids': cp, 'hindsight_prompt_ids': hp,
                'snapshot_round': a.round})
        batch = padded_batch([r['causal_prompt_ids'] for r in records], tok.pad_token_id,
                             device_for(a), left=True)
        # generate computes its own position IDs as the sequence grows.
        batch.pop('position_ids')
        with torch.inference_mode():
            output = model.generate(**batch, max_new_tokens=a.max_new_tokens, do_sample=True,
                temperature=a.temperature, top_p=a.top_p, top_k=0, repetition_penalty=1.0,
                pad_token_id=tok.pad_token_id, eos_token_id=eos, use_cache=True)
        width = batch['input_ids'].shape[1]
        for record, response in zip(records, output[:, width:].tolist()):
            stop = next((i + 1 for i, token in enumerate(response) if token in eos_set), len(response))
            record['response_ids'] = response[:stop]
            mask = reasoning_token_mask(tok, record['response_ids'], prompt_ids=record['causal_prompt_ids'])
            record['positions'] = [i for i, keep in enumerate(mask) if keep]
            record['truncated'] = not bool(response[:stop] and response[stop - 1] in eos_set)
            total_reason += len(record['positions'])
            total_tokens += len(record['response_ids'])
        for score_start in range(0, len(records), a.score_batch_size):
            scored = records[score_start:score_start+a.score_batch_size]
            with torch.inference_mode():
                causal = selected_hidden(model, tok, scored, 'causal_prompt_ids', a)
                need_h = a.method in ('ren_opd', 'opsd', 'union_topk')
                hindsight = selected_hidden(model, tok, scored, 'hindsight_prompt_ids', a) if need_h else [None]*len(scored)
                head = base_model(model).get_output_embeddings()
                for record, c, h in zip(scored, causal, hindsight):
                    record['student_hidden'] = (h if a.method == 'opsd' else c).cpu()
                    if a.method not in ('vanilla_opd', 'opsd'):
                        all_ids, all_causal_probs = [], []
                        k = min(a.top_k, head.weight.shape[0])
                        for pos in range(0, c.shape[0], a.logit_chunk_size):
                            c_logits = head(c[pos:pos+a.logit_chunk_size]).float()
                            cs = c_logits.topk(k, dim=-1).indices
                            if a.method == 'causal_topk':
                                ids = cs
                            else:
                                hs = head(h[pos:pos+a.logit_chunk_size]).float().topk(k, dim=-1).indices
                                # K^2 is tiny compared with the dense vocabulary.
                                novel = ~hs.unsqueeze(-1).eq(cs.unsqueeze(-2)).any(-1)
                                ids = hs.masked_fill(~novel, -1)
                                if a.method == 'union_topk':
                                    ids = torch.cat((cs, ids), dim=-1)
                            all_ids.append(ids.cpu())
                            cp = (c_logits.gather(-1, ids.clamp_min(0)) - c_logits.logsumexp(-1, keepdim=True)).exp()
                            all_causal_probs.append(cp.masked_fill(ids < 0, 0).cpu())
                        columns = 2*k if a.method == 'union_topk' else k
                        record['correction_ids'] = torch.cat(all_ids) if all_ids else torch.empty((0, columns), dtype=torch.long)
                        record['causal_candidate_probs'] = torch.cat(all_causal_probs) if all_causal_probs else torch.empty((0, columns), dtype=torch.float32)
                    # Gold view never enters the teacher artifact or student update.
                    record.pop('hindsight_prompt_ids')
                    atomic_torch(cache / f'rollout_{record["index"]:06d}.pt', record)
        print(json.dumps({'phase': 'collect', 'rank': rank, 'done': min(start+a.rollout_batch_size, len(jobs)),
                          'total': len(jobs)}), flush=True)
    atomic_json(cache / f'collect_rank_{rank}.json', {'reasoning_tokens': total_reason, 'response_tokens': total_tokens})


def teacher_phase(a):
    if a.method == 'opsd':
        return
    from transformers import AutoModelForCausalLM
    _, cache, _ = paths(a)
    student_tok, teacher_tok = load_tokenizer(a.model), load_tokenizer(a.teacher_model)
    kwargs = dict(torch_dtype=dtype_for(a), attn_implementation='sdpa', trust_remote_code=False,
                  low_cpu_mem_usage=True)
    if not a.cpu:
        kwargs.update(device_map='auto', max_memory={i: f'{a.teacher_memory_gib}GiB' for i in range(torch.cuda.device_count())})
    teacher = AutoModelForCausalLM.from_pretrained(a.teacher_model, **kwargs).requires_grad_(False).eval()
    if a.cpu:
        teacher.to('cpu')
    if not a.cpu and any(str(v) in ('cpu', 'disk') for v in getattr(teacher, 'hf_device_map', {}).values()):
        raise RuntimeError('Teacher spilled to CPU/disk: increase --teacher-gpus-per-worker or GPU memory budget')
    state = load_tensor_file(cache / 'student_head.pt')
    check_vocab(student_tok, teacher_tok, state['weight'].shape[0], teacher.get_output_embeddings().weight.shape[0])
    del state
    if a.method == 'vanilla_opd' and a.worker_rank == 0:
        atomic_torch(cache / 'teacher_head.pt', head_state(teacher))
    files = sorted(cache.glob('rollout_*.pt'))[a.worker_rank::a.worker_world]
    head = teacher.get_output_embeddings()
    for start in range(0, len(files), a.score_batch_size):
        batch_files = files[start:start+a.score_batch_size]
        records = [load_tensor_file(f) for f in batch_files]
        with torch.inference_mode():
            hidden = selected_hidden(teacher, student_tok, records, 'causal_prompt_ids', a)
            for record, h, path in zip(records, hidden, batch_files):
                if a.method == 'vanilla_opd':
                    record['teacher_hidden'] = h.cpu()
                else:
                    probabilities = []
                    for pos in range(0, len(h), a.logit_chunk_size):
                        logits = head(h[pos:pos+a.logit_chunk_size].to(head.weight.device)).float()
                        ids = record['correction_ids'][pos:pos+a.logit_chunk_size].to(logits.device)
                        logp = logits.gather(-1, ids.clamp_min(0)) - logits.logsumexp(-1, keepdim=True)
                        probabilities.append(logp.exp().masked_fill(ids < 0, 0).cpu())
                    record['teacher_probs'] = torch.cat(probabilities) if probabilities else torch.empty_like(record['correction_ids'], dtype=torch.float32)
                if a.method not in ('vanilla_opd', 'opsd'):
                    positive = (record['teacher_probs'] > record['causal_candidate_probs']) & (record['correction_ids'] >= 0)
                    record['diagnostics'] = {
                        'frontier_actions': int((record['correction_ids'] >= 0).sum()),
                        'positive_corrections': int(positive.sum()),
                        'corrected_positions': int(positive.any(-1).sum()),
                        'added_mass_sum': float((record['teacher_probs'] - record['causal_candidate_probs']).clamp_min(0).sum()),
                    }
                record['teacher_scored'] = True
                atomic_torch(path, record)
        print(json.dumps({'phase': 'teacher', 'worker': a.worker_rank, 'done': min(start+a.score_batch_size, len(files)),
                          'total': len(files)}), flush=True)


class DistillationStep(nn.Module):
    def __init__(self, student, frozen_head, teacher_head, tok, args):
        super().__init__()
        self.student, self.frozen_head, self.teacher_head = student, frozen_head, teacher_head
        self.tok, self.args = tok, args

    def forward(self, records):
        a = self.args
        hidden = selected_hidden(self.student, self.tok, records, 'causal_prompt_ids', a)
        head = base_model(self.student).get_output_embeddings()
        total = hidden[0].sum() * 0.0
        for r, hs in zip(records, hidden):
            n = len(r['positions'])
            if not n or r.get('dummy', False):
                total = total + hs.sum() * 0.0
                # Keep a trainable full-finetuning LM head in DDP's graph.
                total = total + head.weight.reshape(-1)[0] * 0.0
                continue
            for start in range(0, n, a.logit_chunk_size):
                end = min(n, start+a.logit_chunk_size)
                frozen = r['student_hidden'][start:end].to(hs.device)
                ids = r.get('correction_ids')
                probs = r.get('teacher_probs')
                ids = ids[start:end].to(hs.device) if ids is not None else None
                probs = probs[start:end].to(hs.device) if probs is not None else None
                th = r.get('teacher_hidden')
                th = th[start:end].to(hs.device) if th is not None else None

                def chunk_loss(live, frozen, ids, probs, th):
                    with torch.no_grad():
                        if a.method == 'vanilla_opd':
                            target = self.teacher_head(th).float().softmax(-1)
                        else:
                            logits = self.frozen_head(frozen)
                            if a.method == 'opsd':
                                target = logits.float().softmax(-1)
                            else:
                                target = build_cached_target(logits, ids, probs, method=a.method)
                    return forward_kl(head(live), target, reduction='none').sum()

                value = checkpoint(chunk_loss, hs[start:end], frozen, ids, probs, th,
                                   use_reentrant=False) if a.gradient_checkpointing else chunk_loss(hs[start:end], frozen, ids, probs, th)
                total = total + value / n
        return total


def update_phase(a):
    rank, world = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
    device = device_for(a)
    if world > 1:
        dist.init_process_group('gloo' if a.cpu else 'nccl')
    seed_all(a.seed + a.round)
    root, cache, current = paths(a)
    files = sorted(cache.glob('rollout_*.pt'))
    expected = a.global_batch_prompts * a.rollouts_per_prompt
    if len(files) != expected:
        raise RuntimeError(f'Incomplete round: {len(files)} rollouts, expected {expected}')
    student, tok = load_student(a, current, trainable=True), load_tokenizer(a.model)
    frozen_head = load_head(cache / 'student_head.pt', device)
    teacher_head = load_head(cache / 'teacher_head.pt', device) if a.method == 'vanilla_opd' else None
    step_model = DistillationStep(student, frozen_head, teacher_head, tok, a)
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                                 lr=a.learning_rate, weight_decay=a.weight_decay)
    if (current / 'optimizer.pt').exists():
        optimizer.load_state_dict(load_tensor_file(current / 'optimizer.pt'))
    distributed = DDP(step_model, device_ids=[device.index] if device.type == 'cuda' else None,
                      broadcast_buffers=False, find_unused_parameters=False) if world > 1 else step_model
    metrics = []
    for update in range(a.update_passes):
        order = list(files)
        random.Random(a.seed + 1009*a.round + update).shuffle(order)
        local = [(order[i], False) if i < len(order) else (order[0], True)
                 for i in range(rank, math.ceil(len(order)/world)*world, world)]
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        positions = 0
        active_trajectories = 0
        diagnostic_keys = ('frontier_actions', 'positive_corrections', 'corrected_positions', 'added_mass_sum')
        diagnostic_sums = [0.0] * len(diagnostic_keys)
        started = time.monotonic()
        for start in range(0, len(local), a.train_micro_batch_size):
            records = []
            for path, dummy in local[start:start+a.train_micro_batch_size]:
                r = load_tensor_file(path)
                if r['snapshot_round'] != a.round:
                    raise RuntimeError('Stale rollout snapshot')
                if a.method != 'opsd' and not r.get('teacher_scored'):
                    raise RuntimeError('Missing teacher scoring')
                r['dummy'] = dummy
                positions += 0 if dummy else len(r['positions'])
                active_trajectories += int(not dummy and bool(r['positions']))
                if not dummy:
                    cached_metrics = r.get('diagnostics', {})
                    for index, key in enumerate(diagnostic_keys):
                        diagnostic_sums[index] += float(cached_metrics.get(key, 0.0))
                records.append(r)
            last = start+a.train_micro_batch_size >= len(local)
            ctx = distributed.no_sync() if world > 1 and not last else contextlib.nullcontext()
            with ctx:
                value = distributed(records)
                loss = value * (world / expected)
                loss.backward()
            loss_sum += value.detach()
        counts = torch.tensor([positions, active_trajectories], device=device, dtype=torch.long)
        diagnostic_totals = torch.tensor(diagnostic_sums, device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(loss_sum)
            dist.all_reduce(counts)
            dist.all_reduce(diagnostic_totals)
        total_positions, total_active = counts.tolist()
        if total_active == 0:
            raise RuntimeError('No reasoning tokens in round: inspect thinking tags/rollout budget')
        # DDP averages ranks; backward above sums trajectories / expected.
        # Exclude empty reasoning spans from the per-sequence expectation.
        # Rescaling after the final all-reduce avoids a preliminary cache pass.
        for parameter in student.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(expected / total_active)
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), a.max_grad_norm, error_if_nonfinite=True)
        if not torch.isfinite(loss_sum):
            raise FloatingPointError('Non-finite distillation loss')
        optimizer.step()
        entry = {'round': a.round, 'update': update, 'forward_kl': loss_sum.item()/total_active,
                 'grad_norm': grad_norm.item(), 'reasoning_tokens': total_positions,
                 'trajectories': expected, 'supervised_trajectories': total_active, 'seconds': time.monotonic()-started,
                 **dict(zip(diagnostic_keys, diagnostic_totals.tolist()))}
        metrics.append(entry)
        if rank == 0:
            print(json.dumps(entry), flush=True)
    if rank == 0:
        next_path = root / 'checkpoints' / f'round_{a.round+1:04d}'
        save_checkpoint(student, tok, next_path, optimizer,
                        {'completed_rounds': a.round+1, 'model': a.model, 'method': a.method, 'metrics': metrics})
        atomic_json(root / 'metrics' / f'round_{a.round:04d}.json', metrics)
        atomic_json(root / 'latest.json', {'checkpoint': str(next_path), 'completed_rounds': a.round+1})
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def gpu_ids(a):
    if a.cpu:
        return []
    if a.gpus != 'auto':
        ids = [x.strip() for x in a.gpus.split(',') if x.strip()]
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        ids = [x.strip() for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x.strip()]
    else:
        ids = [str(i) for i in range(torch.cuda.device_count())]
    if not ids or '-1' in ids or len(set(ids)) != len(ids):
        raise ValueError('No GPUs or duplicate GPU IDs; use --cpu only for tiny smoke tests')
    return ids


def child_arguments(a, phase, round_index):
    args = []
    excluded = {'phase', 'round', 'worker_rank', 'worker_world', 'resume', 'dry_run'}
    for name, value in vars(a).items():
        if name in excluded:
            continue
        flag = '--' + name.replace('_', '-')
        if isinstance(value, bool):
            if value:
                args.append(flag)
            elif name == 'gradient_checkpointing':
                args.append('--no-gradient-checkpointing')
        else:
            args.extend((flag, str(value)))
    return args + ['--phase', phase, '--round', str(round_index)]


def run_phase(a, phase, round_index, ids, *, parallel=False):
    command = [sys.executable]
    if parallel and len(ids) > 1:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        command += ['-m', 'torch.distributed.run', '--nnodes', '1', '--node_rank', '0',
                    '--master_addr', '127.0.0.1', '--master_port', str(port),
                    '--nproc_per_node', str(len(ids))]
    command += ['-m', 'lulu.training'] + child_arguments(a, phase, round_index)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(ids), TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '1'))
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(PROJECT_ROOT), env.get('PYTHONPATH', '')]))
    env.setdefault('NCCL_SOCKET_IFNAME', 'lo')
    env.setdefault('GLOO_SOCKET_IFNAME', 'lo')
    subprocess.run(command, env=env, check=True)


def run_teachers(a, round_index, ids):
    if a.method == 'opsd':
        return
    group_size = min(a.teacher_gpus_per_worker, len(ids)) if ids else 1
    groups = [ids[i:i+group_size] for i in range(0, len(ids)-group_size+1, group_size)] if ids else [[]]
    if a.teacher_workers:
        groups = groups[:a.teacher_workers]
    workers = []
    try:
        for rank, group in enumerate(groups):
            cmd = [sys.executable, '-m', 'lulu.training'] + child_arguments(a, 'teacher', round_index)
            cmd += ['--worker-rank', str(rank), '--worker-world', str(len(groups))]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(group), TOKENIZERS_PARALLELISM='false',
                       OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '1'))
            env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(PROJECT_ROOT), env.get('PYTHONPATH', '')]))
            workers.append(subprocess.Popen(cmd, env=env))
        failures = [p.wait() for p in workers]
        if any(failures):
            raise RuntimeError(f'Teacher workers failed: {failures}')
    finally:
        for process in workers:
            if process.poll() is None:
                process.terminate()
                process.wait()


def resume_settings(config):
    """Compare scientific settings independent of data/output path spelling.

    The manifest is read from the requested output directory; train_sha256 pins
    the actual data bytes. These two locations may change when code or artifacts
    move, without changing trajectories or optimization. All other settings,
    including model identity and the data hash, must still match exactly.
    """
    config = dict(config)
    config.setdefault('backend', 'staged')  # Manifests before resident training.
    if config['backend'] == 'staged':
        for key in ('student_gpus', 'hindsight_gpus', 'teacher_gpus', 'save_every', 'worker_timeout'):
            config.pop(key, None)
    return {key: value for key, value in config.items() if key not in {'train_data', 'output_dir'}}


def run(a):
    ids = gpu_ids(a)
    plan = {'model': a.model, 'teacher': a.teacher_model, 'method': a.method, 'rounds': a.rounds,
            'gpus': ids, 'student_workers': max(1, len(ids)),
            'teacher_gpus_per_worker': min(len(ids), a.teacher_gpus_per_worker),
            'rollouts_per_round': a.global_batch_prompts*a.rollouts_per_prompt,
            'optimizer_updates_per_round': a.update_passes,
            'phases': ['student_rollout_and_frozen_views', 'answer_blind_teacher_score', 'ddp_student_update'],
            'loss': 'exact full-vocabulary forward KL, mean of per-trajectory reasoning-token means',
            'rollout_policy': {'temperature': a.temperature, 'top_p': a.top_p, 'top_k': 0}}
    print(json.dumps(plan, indent=2), flush=True)
    if a.dry_run:
        return
    root = Path(a.output_dir).resolve()
    manifest_path = root / 'run_config.json'
    config = {k: v for k, v in vars(a).items() if k not in ('resume', 'phase', 'round', 'dry_run', 'worker_rank', 'worker_world')}
    config['train_sha256'] = hashlib.sha256(Path(a.train_data).read_bytes()).hexdigest()
    if manifest_path.exists():
        if not a.resume:
            raise FileExistsError('Run exists; use --resume with the identical configuration')
        if resume_settings(json.loads(manifest_path.read_text())) != resume_settings(config):
            raise ValueError('Resume configuration/data differs from original run')
    else:
        if a.resume:
            raise FileNotFoundError('Cannot resume without run_config.json')
        if root.exists() and any(root.iterdir()):
            raise FileExistsError('Output directory must be empty for a new run')
        atomic_json(manifest_path, config)
    initial = root / 'checkpoints' / 'round_0000'
    if not (initial / 'lulu_state.json').exists():
        run_phase(a, 'init', 0, ids[:1])
    for round_index in range(a.rounds):
        finished = root / 'checkpoints' / f'round_{round_index+1:04d}' / 'lulu_state.json'
        if finished.exists():
            continue
        cache = root / 'round_cache' / f'round_{round_index:04d}'
        if cache.exists():
            shutil.rmtree(cache)  # Only this run's incomplete, regenerable cache.
        run_phase(a, 'collect', round_index, ids, parallel=True)
        run_teachers(a, round_index, ids)
        run_phase(a, 'update', round_index, ids, parallel=True)
        if not a.keep_round_cache:
            shutil.rmtree(cache)
    print(f'Completed: {root / "checkpoints" / f"round_{a.rounds:04d}"}', flush=True)


def main(argv=None):
    a = parser().parse_args(argv)
    validate_args(a)
    if a.phase == 'run':
        if a.backend == 'persistent':
            from lulu.persistent import run_persistent
            run_persistent(a)
        else:
            run(a)
    else:
        {'init': init_phase, 'collect': collect_phase, 'teacher': teacher_phase, 'update': update_phase}[a.phase](a)


if __name__ == "__main__":
    main()
