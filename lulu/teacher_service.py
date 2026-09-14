"""Persistent answer-blind Teacher scoring with native Transformers tensor parallel.

The controller sends tokenized causal prefixes and a sparse list of token IDs.
Probabilities for those IDs use an exact full-vocabulary normalizer. Native TP
shards Qwen3 attention/MLP matrices, unlike ``device_map='auto'`` layer dispatch.
The model and its CUDA allocations live until the service receives ``stop``.
"""
from __future__ import annotations

from datetime import timedelta
import os
import time
import traceback

import torch
import torch.distributed as dist

from lulu import training


_RECORD_KEYS = frozenset(('causal_prompt_ids', 'response_ids', 'positions', 'correction_ids'))
_REQUEST_KEYS = frozenset(('op', 'round', 'request_id', 'records'))
_TEACHER_METHODS = frozenset(('ren_opd', 'causal_topk', 'union_topk', 'vanilla_opd'))


def _integers(value, name, *, nonempty=False):
    if not isinstance(value, (list, tuple)) or (nonempty and not value):
        raise ValueError(f'{name} must be a {"nonempty " if nonempty else ""}sequence of integer IDs')
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in value):
        raise ValueError(f'{name} must contain nonnegative integer IDs')
    return list(value)


def sanitize_score_request(request, method, *, vocab_size=None, max_sequence_tokens=None):
    """Validate and copy the strict answer-blind IPC schema.

    Rejecting unknown keys makes an accidental gold/hindsight transfer visible;
    callers must construct the allowed payload before sending it to the Teacher.
    Only local trusted multiprocessing pipes are supported (pickle transport).
    """
    if method not in _TEACHER_METHODS:
        raise ValueError(f'Teacher service does not support method {method!r}')
    if not isinstance(request, dict) or set(request) != _REQUEST_KEYS or request.get('op') != 'score':
        raise ValueError('Teacher score request must contain only op, round, request_id, records')
    round_index, request_id = request['round'], request['request_id']
    if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index < 0:
        raise ValueError('Teacher request round must be a nonnegative integer')
    if not isinstance(request_id, (int, str)) or isinstance(request_id, bool):
        raise ValueError('Teacher request_id must be an integer or string')
    if not isinstance(request['records'], (list, tuple)):
        raise ValueError('Teacher records must be a sequence')
    result = []
    for record in request['records']:
        required = _RECORD_KEYS - {'correction_ids'} if method == 'vanilla_opd' else _RECORD_KEYS
        if not isinstance(record, dict) or not required <= set(record) <= _RECORD_KEYS:
            raise ValueError('Teacher record contains missing or forbidden fields; only causal tokens and candidate IDs are allowed')
        prompt = _integers(record['causal_prompt_ids'], 'causal_prompt_ids', nonempty=True)
        response = _integers(record['response_ids'], 'response_ids')
        positions = _integers(record['positions'], 'positions')
        if positions != sorted(set(positions)) or any(p >= len(response) for p in positions):
            raise ValueError('Teacher positions must be distinct, increasing response-token indices')
        if max_sequence_tokens is not None and len(prompt) + len(response) > max_sequence_tokens:
            raise ValueError('Scoring context exceeds max_sequence_tokens; refusing truncation')
        if vocab_size is not None and any(t >= vocab_size for t in prompt + response):
            raise ValueError('Causal token IDs exceed the Teacher vocabulary')
        clean = {'causal_prompt_ids': prompt, 'response_ids': response, 'positions': positions}
        if 'correction_ids' in record:
            ids = torch.as_tensor(record['correction_ids'])
            if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError('Teacher correction_ids must be integer tensors')
            if ids.ndim != 2 or ids.shape[0] != len(positions):
                raise ValueError('Teacher correction_ids must have shape [reasoning_positions, candidates]')
            if bool((ids < -1).any()) or (vocab_size is not None and bool((ids >= vocab_size).any())):
                raise ValueError('Teacher correction_ids must be -1 padding or valid vocabulary IDs')
            clean['correction_ids'] = ids.detach().to(device='cpu', dtype=torch.long).contiguous()
        result.append(clean)
    return {'op': 'score', 'round': round_index, 'request_id': request_id, 'records': result}


def _full_tensor(value):
    # Native Qwen3 uses replicated decoder outputs and an unsharded LM head.
    # Future supported plans may return DTensors; every TP rank must join any
    # redistribution, including ranks that do not send the final IPC response.
    from torch.distributed.tensor import DTensor
    return value.full_tensor() if isinstance(value, DTensor) else value


@torch.inference_mode()
def score_records(model, tokenizer, records, args):
    """Return exact selected probabilities (or vanilla hidden states), in order.

    Call identically on all TP ranks: decoder forwards and any DTensor gathers
    are collectives. Only rank zero returns these CPU tensors to the controller.
    The output head is evaluated in position chunks, avoiding an allocation of
    [batch, sequence_length, vocabulary_size] logits.
    """
    if args.method not in _TEACHER_METHODS:
        raise ValueError(f'Teacher service does not support method {args.method!r}')
    if args.score_batch_size < 1 or args.logit_chunk_size < 1:
        raise ValueError('Teacher scoring batch and logit chunk sizes must be positive')
    result = []
    head = training.base_model(model).get_output_embeddings()
    for start in range(0, len(records), args.score_batch_size):
        batch = records[start:start + args.score_batch_size]
        states = training.selected_hidden(model, tokenizer, batch, 'causal_prompt_ids', args)
        for record, hidden in zip(batch, states):
            hidden = _full_tensor(hidden)
            if args.method == 'vanilla_opd':
                result.append({'teacher_hidden': hidden.detach().cpu().contiguous(), 'teacher_scored': True})
                continue
            chunks = []
            for offset in range(0, len(hidden), args.logit_chunk_size):
                stop = offset + args.logit_chunk_size
                logits = _full_tensor(head(hidden[offset:stop].to(head.weight.device))).float()
                ids = record['correction_ids'][offset:stop].to(logits.device)
                # Do not normalize within the candidate set: q_T uses the full
                # vocabulary even for IDs outside the Teacher's own top-k.
                logp = logits.gather(-1, ids.clamp_min(0)) - logits.logsumexp(-1, keepdim=True)
                chunks.append(logp.exp().masked_fill(ids < 0, 0).cpu())
            probabilities = torch.cat(chunks) if chunks else torch.empty_like(record['correction_ids'], device='cpu', dtype=torch.float32)
            result.append({'teacher_probs': probabilities.contiguous(), 'teacher_scored': True})
    return result


def install_synchronous_rowwise_tp():
    """Keep native TP sharding while materializing each rowwise reduction.

    Transformers 4.52 returns lazy asynchronous rowwise DTensors. With PyTorch
    2.7 we observed a large-batch Qwen3 run stall in these reductions, leaving
    ranks at different collective sequence numbers. Explicit completion avoids
    carrying pending collective tensors into subsequent DTensor shape dispatch.
    This changes scheduling only: matrices remain column/row sharded and sums
    are exactly the native TP reductions. Registration is local to this Teacher
    subprocess; it does not modify Transformers files or Student DDP workers.
    """
    from transformers.integrations.tensor_parallel import ALL_PARALLEL_STYLES, RowwiseParallel

    class SynchronousRowwiseParallel(RowwiseParallel):
        @staticmethod
        def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
            if outputs.placements != output_layouts:
                outputs = outputs.redistribute(placements=output_layouts, async_op=False)
            if hasattr(mod, '_bias'):
                outputs = outputs + mod._bias
            return outputs.to_local() if use_local_output else outputs

    # Qwen3's official plan uses "rowwise" for attention o_proj and MLP
    # down_proj. Reuse the official partition implementation and only change
    # when the all-reduce output becomes available to subsequent operations.
    ALL_PARALLEL_STYLES['rowwise'] = SynchronousRowwiseParallel()


def _load_teacher(args, world, device):
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(args.teacher_model, trust_remote_code=False)
    kwargs = dict(torch_dtype=training.dtype_for(args), attn_implementation='sdpa',
                  trust_remote_code=False, low_cpu_mem_usage=True)
    if world > 1:
        if not getattr(config, 'base_model_tp_plan', None):
            raise ValueError('Teacher model has no native Transformers tensor-parallel plan; use a supported model or one Teacher GPU')
        install_synchronous_rowwise_tp()
        kwargs['tp_plan'] = 'auto'
    # Transformers 4.52 may redirect nonzero local-rank stdout/stderr while
    # initializing TP. Restore both so a failed worker retains useful logs.
    import sys
    streams = sys.stdout, sys.stderr
    try:
        model = AutoModelForCausalLM.from_pretrained(args.teacher_model, **kwargs)
    finally:
        sys.stdout, sys.stderr = streams
    if world == 1:
        model.to(device)
    elif not getattr(model, '_tp_plan', None):
        raise RuntimeError('Teacher tensor parallel was requested but no TP plan was installed')
    model.requires_grad_(False).eval()
    student_tokenizer = training.load_tokenizer(args.model)
    teacher_tokenizer = training.load_tokenizer(args.teacher_model)
    student_config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
    training.check_vocab(student_tokenizer, teacher_tokenizer,
                         student_config.vocab_size, model.get_output_embeddings().weight.shape[0])
    return model, teacher_tokenizer


def teacher_worker(args, rank, world, device_ids, address, port, conn):
    """Spawn entry point for one rank of the persistent Teacher TP group.

    ``device_ids`` are CUDA IDs visible to the parent, kept in the same order
    for every rank. Only rank 0 receives a controller Connection; peers use the
    TP process group's broadcast. The parent must poll process exit codes as
    well as the rank-zero pipe, and terminate the group on a failure/timeout.
    """
    request = None
    try:
        if not isinstance(world, int) or world < 1 or not 0 <= rank < world:
            raise ValueError('Invalid Teacher rank/world')
        if (rank == 0) != (conn is not None):
            raise ValueError('Only Teacher rank zero must own the controller connection')
        if args.cpu and world != 1:
            raise ValueError('CPU Teacher service supports one rank only')
        if not args.cpu and len(device_ids) != world:
            raise ValueError('Teacher device_ids must contain one GPU per TP rank')
        if not args.cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank),
                          MASTER_ADDR=str(address), MASTER_PORT=str(port))
        device = torch.device('cpu') if args.cpu else torch.device('cuda', rank)
        if not args.cpu:
            torch.cuda.set_device(device)
        if world > 1:
            dist.init_process_group('nccl', rank=rank, world_size=world, device_id=device,
                init_method=f'tcp://{address}:{port}',
                timeout=timedelta(seconds=getattr(args, 'worker_timeout', 1800)))
        started = time.monotonic()
        model, tokenizer = _load_teacher(args, world, device)
        ready = {'op': 'ready', 'role': 'teacher', 'world': world,
                 'backend': 'transformers_tp' if world > 1 else 'transformers',
                 'tp_reductions': 'synchronous_rowwise' if world > 1 else None,
                 'seconds': time.monotonic() - started}
        if args.method == 'vanilla_opd':
            # All TP ranks participate should a future head itself be sharded.
            head = model.get_output_embeddings()
            weight = _full_tensor(head.weight.detach()).cpu().contiguous()
            bias = None if getattr(head, 'bias', None) is None else _full_tensor(head.bias.detach()).cpu().contiguous()
            if rank == 0:
                ready['teacher_head'] = {'weight': weight, 'bias': bias}
        if world > 1:
            dist.barrier()
        if rank == 0:
            conn.send(ready)
        last_round, seen_requests = -1, set()
        while True:
            payload = [conn.recv() if rank == 0 else None]
            if world > 1:
                dist.broadcast_object_list(payload, src=0, device=device)
            request = payload[0]
            if isinstance(request, dict) and request.get('op') in ('stop', 'shutdown'):
                if world > 1:
                    dist.barrier()
                if rank == 0:
                    conn.send({'op': 'stopped', 'role': 'teacher'})
                return
            request = sanitize_score_request(request, args.method,
                vocab_size=model.get_output_embeddings().weight.shape[0],
                max_sequence_tokens=args.max_sequence_tokens)
            if request['round'] < last_round:
                raise RuntimeError('Teacher rejected a stale scoring round')
            if request['round'] != last_round:
                last_round, seen_requests = request['round'], set()
            if request['request_id'] in seen_requests:
                raise RuntimeError('Teacher rejected a duplicate request_id in this round')
            seen_requests.add(request['request_id'])
            started = time.monotonic()
            records = score_records(model, tokenizer, request['records'], args)
            if rank == 0:
                conn.send({'op': 'scored', 'role': 'teacher', 'round': request['round'],
                           'request_id': request['request_id'], 'records': records,
                           'seconds': time.monotonic() - started})
    except BaseException as error:
        if rank == 0 and conn is not None:
            try:
                conn.send({'op': 'error', 'role': 'teacher', 'rank': rank,
                    'round': request.get('round') if isinstance(request, dict) else None,
                    'request_id': request.get('request_id') if isinstance(request, dict) else None,
                    'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()})
            except (BrokenPipeError, EOFError, OSError):
                pass
        raise
    finally:
        if conn is not None:
            conn.close()
        if dist.is_initialized():
            dist.destroy_process_group()
