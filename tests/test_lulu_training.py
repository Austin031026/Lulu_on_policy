"""Exact dense/chunked gradients and two-rank CPU updates on a tiny Qwen3."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lulu import training
from lulu.objective import build_target, forward_kl


def _student(lora=False):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.set_num_threads(1)
    torch.manual_seed(91)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=31, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
        attention_dropout=0., tie_word_embeddings=False, pad_token_id=0,
        bos_token_id=1, eos_token_id=2))
    if lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4,
            target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                torch.nn.init.normal_(parameter, std=.03)
    return model


def _args(method='ren_opd', chunk_size=2, checkpointing=True):
    return SimpleNamespace(method=method, max_sequence_tokens=128,
        logit_chunk_size=chunk_size, gradient_checkpointing=checkpointing)


def _records(snapshot, method, *, include_empty=False):
    tok = SimpleNamespace(pad_token_id=0)
    records = [
        {'index': 0, 'causal_prompt_ids': [3, 4], 'hindsight_prompt_ids': [3, 4, 20, 21],
         'response_ids': [6, 7, 8, 9, 10], 'positions': [0, 1, 2, 4], 'snapshot_round': 0},
        {'index': 1, 'causal_prompt_ids': [3, 5, 11], 'hindsight_prompt_ids': [3, 5, 11, 24],
         'response_ids': [12, 13, 14], 'positions': [0, 2], 'snapshot_round': 0},
    ]
    if include_empty:
        records.insert(0, {'index': 2, 'causal_prompt_ids': [3, 4], 'hindsight_prompt_ids': [3, 4, 20],
                          'response_ids': [2], 'positions': [], 'snapshot_round': 0})
    frozen_head = copy.deepcopy(training.base_model(snapshot).get_output_embeddings()).requires_grad_(False)
    teacher_head = torch.nn.Linear(16, 31, bias=False).requires_grad_(False)
    torch.nn.init.normal_(teacher_head.weight, std=.15)
    with torch.no_grad():
        c = training.selected_hidden(snapshot, tok, records, 'causal_prompt_ids', _args())
        h = training.selected_hidden(snapshot, tok, records, 'hindsight_prompt_ids', _args())
        for record, causal, hindsight in zip(records, c, h):
            record['student_hidden'] = (hindsight if method == 'opsd' else causal).detach().clone()
            record['hindsight_hidden'] = hindsight.detach().clone()
            record['causal_hidden'] = causal.detach().clone()
            record['teacher_hidden'] = torch.randn_like(causal)
            record['teacher_scored'] = True
            record['diagnostics'] = {'frontier_actions': len(causal) * 3,
                'positive_corrections': len(causal), 'corrected_positions': len(causal),
                'added_mass_sum': len(causal) * .1}
            if method in ('ren_opd', 'causal_topk', 'union_topk'):
                cs = frozen_head(causal).topk(3, dim=-1).indices
                hs = frozen_head(hindsight).topk(3, dim=-1).indices
                novel = ~hs.unsqueeze(-1).eq(cs.unsqueeze(-2)).any(-1)
                novel_ids = hs.masked_fill(~novel, -1)
                ids = cs if method == 'causal_topk' else torch.cat((cs, novel_ids), -1) if method == 'union_topk' else novel_ids
                record['correction_ids'] = ids
                teacher_p = teacher_head(record['teacher_hidden']).softmax(-1)
                record['teacher_probs'] = teacher_p.gather(-1, ids.clamp_min(0)).masked_fill(ids < 0, 0)
    return records, frozen_head, teacher_head, tok


def _dense_loss(student, records, frozen_head, teacher_head, tok, method):
    hidden = training.selected_hidden(student, tok, records, 'causal_prompt_ids', _args())
    head = training.base_model(student).get_output_embeddings()
    total = hidden[0].sum() * 0.
    for r, live in zip(records, hidden):
        if not r['positions'] or r.get('dummy'):
            total = total + live.sum() * 0. + head.weight.reshape(-1)[0] * 0.
            continue
        target = build_target(frozen_head(r['causal_hidden']), frozen_head(r['hindsight_hidden']),
                              teacher_head(r['teacher_hidden']), 3, method)
        total = total + forward_kl(head(live), target)
    return total


@pytest.mark.parametrize('method', training.METHODS)
@pytest.mark.parametrize('checkpointing,chunk_size', [(False, 50), (True, 1), (True, 3)])
def test_real_qwen3_chunked_loss_and_gradients_match_dense(method, checkpointing, chunk_size):
    snapshot = _student().eval().requires_grad_(False)
    records, frozen_head, teacher_head, tok = _records(snapshot, method)
    live = copy.deepcopy(snapshot).train().requires_grad_(True)
    with torch.no_grad():
        live.model.layers[0].self_attn.q_proj.weight.add_(.015)
    expected = _dense_loss(live, records, frozen_head, teacher_head, tok, method)
    expected.backward()
    gradients = {name: p.grad.clone() for name, p in live.named_parameters() if p.grad is not None}
    live.zero_grad(set_to_none=True)
    if checkpointing:
        live.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    actual = training.DistillationStep(live, frozen_head, teacher_head, tok,
                                      _args(method, chunk_size, checkpointing))(records)
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
    for name, parameter in live.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[name], atol=3e-6, rtol=3e-4)
    assert all(p.grad is None for p in frozen_head.parameters())
    assert all(p.grad is None for p in teacher_head.parameters())


def test_real_peft_qwen3_checkpointed_chunk_gradients_and_bounded_vocab_saves():
    live = _student(lora=True)
    snapshot = copy.deepcopy(live).eval().requires_grad_(False)
    records, frozen_head, teacher_head, tok = _records(snapshot, 'ren_opd')
    expected = _dense_loss(live, records, frozen_head, teacher_head, tok, 'ren_opd')
    expected.backward()
    grads = {n: p.grad.clone() for n, p in live.named_parameters() if p.grad is not None}
    live.zero_grad(set_to_none=True)
    live.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    live.enable_input_require_grads()
    shapes = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: (shapes.append(tuple(t.shape)) or t), lambda t: t):
        actual = training.DistillationStep(live, frozen_head, teacher_head, tok, _args())(records)
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
    for n, p in live.named_parameters():
        if n in grads:
            torch.testing.assert_close(p.grad, grads[n], atol=3e-6, rtol=3e-4)
    assert not any(len(shape) == 2 and shape[-1] == 31 for shape in shapes)


def _save_tokenizer(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    vocabulary = {f'token{i}': i for i in range(31)}
    vocabulary.update({'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 30})
    for i in [0, 1, 2, 30]:
        vocabulary.pop(f'token{i}')
    backend = Tokenizer(WordLevel(vocabulary, unk_token='<unk>'))
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>',
                                 bos_token='<bos>', eos_token='<eos>', unk_token='<unk>')
    tok.save_pretrained(path)
    return tok


def _ddp_update_worker(rank, args, rendezvous):
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE='2', LOCAL_RANK=str(rank))
    original_init = dist.init_process_group
    with patch.object(training.dist, 'init_process_group', side_effect=lambda backend:
                      original_init(backend, init_method=f'file://{rendezvous}', rank=rank, world_size=2)):
        training.update_phase(args)


@pytest.mark.skipif(not dist.is_available(), reason='PyTorch distributed unavailable')
def test_two_rank_update_matches_global_nonempty_sequence_mean_with_dummy_rank(tmp_path):
    """One rank gets only an empty trajectory + dummy; both must still all-reduce."""
    from transformers import AutoModelForCausalLM
    base = tmp_path / 'base'
    root = tmp_path / 'run'
    model = _student()
    model.save_pretrained(base)
    tok = _save_tokenizer(base)
    checkpoint_dir = root / 'checkpoints/round_0000'
    training.save_checkpoint(model, tok, checkpoint_dir)
    records, frozen_head, teacher_head, _ = _records(copy.deepcopy(model).eval().requires_grad_(False),
                                                    'ren_opd', include_empty=True)
    # For seed 42, shuffle [0,1,2] -> [1,0,2]. Rank 1 sees empty record 0 + dummy.
    order = list(range(3))
    random.Random(42).shuffle(order)
    assert order[1] == 0 and records[0]['positions'] == []
    cache = root / 'round_cache/round_0000'
    training.atomic_torch(cache / 'student_head.pt', {'weight': frozen_head.weight, 'bias': None})
    for index, record in enumerate(records):
        training.atomic_torch(cache / f'rollout_{index:06d}.pt', record)
    args = training.parser().parse_args(['--phase', 'update', '--cpu', '--dtype', 'float32',
        '--model', str(base), '--train-data', str(tmp_path/'unused.jsonl'), '--output-dir', str(root),
        '--global-batch-prompts', '3', '--lora-rank', '0', '--max-sequence-tokens', '128',
        '--max-prompt-tokens', '64', '--learning-rate', '0.0001', '--logit-chunk-size', '2'])
    expected = copy.deepcopy(model).train()
    optimizer = torch.optim.AdamW(expected.parameters(), lr=args.learning_rate, weight_decay=0.)
    # Use the same FP32 cached-target reconstruction and microbatch shapes as
    # production; Adam amplifies harmless dense-vs-sparse rounding at gradients
    # near its epsilon. Dense mathematical agreement is checked separately above.
    serial_step = training.DistillationStep(expected, frozen_head, teacher_head, tok, _args())
    reference = sum(serial_step([records[index]]) for index in order) / 2
    reference.backward()
    torch.nn.utils.clip_grad_norm_(expected.parameters(), args.max_grad_norm)
    optimizer.step()
    mp.spawn(_ddp_update_worker, args=(args, str(tmp_path/'gloo_init')), nprocs=2, join=True)
    actual = AutoModelForCausalLM.from_pretrained(root / 'checkpoints/round_0001')
    saved_optimizer = training.load_tensor_file(root / 'checkpoints/round_0001/optimizer.pt')
    for index, (name, parameter) in enumerate(expected.named_parameters()):
        state = saved_optimizer['state'][index]
        expected_state = optimizer.state[parameter]
        # Optimizer moments preserve the actual all-reduced gradient, including
        # near-zero coordinates where Adam makes weight equality ill-conditioned.
        torch.testing.assert_close(state['exp_avg'], expected_state['exp_avg'], atol=4e-9, rtol=3e-4)
        torch.testing.assert_close(state['exp_avg_sq'], expected_state['exp_avg_sq'], atol=1e-11, rtol=3e-4)
        meaningful = parameter.grad.abs() > 1e-6
        torch.testing.assert_close(actual.state_dict()[name][meaningful], parameter.detach()[meaningful], atol=2e-6, rtol=3e-5)
        assert torch.isfinite(actual.state_dict()[name]).all()
    assert not torch.equal(actual.lm_head.weight, model.lm_head.weight)
    metrics = json.loads((root / 'metrics/round_0000.json').read_text())[0]
    assert metrics['trajectories'] == 3
    assert metrics['supervised_trajectories'] == 2
    assert metrics['reasoning_tokens'] == 6
    assert metrics['frontier_actions'] == 18
    assert metrics['forward_kl'] == pytest.approx(reference.item(), abs=2e-7, rel=2e-5)


def test_schedule_resume_round_indices_equal_uninterrupted_stream():
    for size in (1, 3, 17):
        uninterrupted = training.schedule(size, 35, 0, 42)
        resumed_rounds = sum((training.schedule(size, 7, r, 42) for r in range(5)), [])
        assert resumed_rounds == uninterrupted


def test_functional_qwen3_dropout_is_disabled_for_snapshot_update_consistency():
    model = _student().train()
    model.config.attention_dropout = .4
    model.model.layers[0].self_attn.attention_dropout = .4
    training.disable_dropout(model)
    assert model.config.attention_dropout == 0
    assert model.model.layers[0].self_attn.attention_dropout == 0
    batch = training.padded_batch([[3, 4, 5, 6]], 0, 'cpu')
    with torch.no_grad():
        training_outputs = model(**batch).logits
        model.eval()
        snapshot_outputs = model(**batch).logits
    torch.testing.assert_close(training_outputs, snapshot_outputs)


def test_init_worker_runs_after_relocation_from_another_cwd(tmp_path, monkeypatch):
    """The real child Python must find LuLu without the old Soraka scripts path."""
    base = tmp_path / 'local_model'
    model = _student()
    model.save_pretrained(base)
    _save_tokenizer(base)
    work = tmp_path / 'caller'
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.delenv('PYTHONPATH', raising=False)
    args = training.parser().parse_args([
        '--cpu', '--dtype', 'float32', '--model', str(base),
        '--train-data', 'unused.jsonl', '--output-dir', 'run', '--lora-rank', '0',
    ])
    training.run_phase(args, 'init', 0, [])
    checkpoint_dir = work / 'run/checkpoints/round_0000'
    assert (checkpoint_dir / 'model.safetensors').is_file()
    assert json.loads((checkpoint_dir / 'lulu_state.json').read_text())['completed_rounds'] == 0
    assert not (work / 'Soraka').exists()


def test_resume_after_code_move_accepts_paths_but_rejects_data_or_settings_changes(tmp_path, monkeypatch):
    """Moving the caller must not invalidate historical path strings in manifests."""
    import hashlib
    import shutil
    work = tmp_path / 'new_project'
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(training, 'gpu_ids', lambda args: [])
    root = tmp_path / 'existing_run'
    (root / 'checkpoints/round_0000').mkdir(parents=True)
    (root / 'checkpoints/round_0001').mkdir(parents=True)
    for i in range(2):
        (root / f'checkpoints/round_{i:04d}/lulu_state.json').write_text('{}')
    data = tmp_path / 'data.jsonl'
    data.write_text('{"same_data": true}\n')
    copied = work / 'relocated_data.jsonl'
    shutil.copyfile(data, copied)
    args = training.parser().parse_args([
        '--cpu', '--dtype', 'float32', '--train-data', str(data),
        '--output-dir', str(root), '--rounds', '1', '--resume',
    ])
    original = {k: v for k, v in vars(args).items()
                if k not in ('resume', 'phase', 'round', 'dry_run', 'worker_rank', 'worker_world')}
    original['train_sha256'] = hashlib.sha256(data.read_bytes()).hexdigest()
    (root / 'run_config.json').write_text(json.dumps(original))
    args.train_data = 'relocated_data.jsonl'
    args.output_dir = '../existing_run'
    with patch.object(training, 'run_phase', side_effect=AssertionError('completed checkpoint must not rerun')):
        training.run(args)
    assert json.loads((root / 'run_config.json').read_text()) == original
    args.learning_rate *= 2
    with pytest.raises(ValueError, match='configuration/data differs'):
        training.run(args)
    args.learning_rate /= 2
    copied.write_text('{"same_data": false}\n')
    with pytest.raises(ValueError, match='configuration/data differs'):
        training.run(args)
