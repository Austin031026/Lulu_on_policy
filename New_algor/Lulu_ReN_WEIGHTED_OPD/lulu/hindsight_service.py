"""Resident answer-conditioned Student scoring with explicit snapshot barriers.

The service does not train or generate. Its weights are synchronized from the
causal Student before every round, and its responses contain only hidden states
or vocabulary IDs; answer-bearing prompts stay inside the scoring request.
"""
from __future__ import annotations

import os
from pathlib import Path
import time
import traceback
from collections.abc import Mapping

import torch

from lulu.training import base_model, disable_dropout, load_student, load_tokenizer, selected_hidden


def sync_snapshot(model, state, expected_names):
    """Validate the complete update before copying any resident parameters."""
    if not isinstance(state, Mapping):
        raise ValueError('Snapshot state must be a parameter-name mapping')
    expected = set(expected_names)
    if set(state) != expected:
        missing, extra = sorted(expected - set(state)), sorted(set(state) - expected)
        raise ValueError(f'Snapshot parameter mismatch: missing={missing}, extra={extra}')
    parameters = dict(model.named_parameters())
    if not expected <= parameters.keys():
        raise ValueError('Snapshot names do not match the resident Student architecture')
    for name, source in state.items():
        parameter = parameters[name]
        if not isinstance(source, torch.Tensor):
            raise ValueError(f'Snapshot parameter {name} must be a tensor')
        if source.device.type != 'cpu':
            raise ValueError(f'Snapshot parameter {name} must be sent on CPU')
        if source.shape != parameter.shape or source.dtype != parameter.dtype:
            raise ValueError(f'Snapshot shape/dtype mismatch for {name}: '
                             f'{source.shape}/{source.dtype} versus {parameter.shape}/{parameter.dtype}')
    with torch.no_grad():
        for name, source in state.items():
            parameters[name].copy_(source)
    model.requires_grad_(False).eval()


def _causal_ids(record, k, vocabulary_size):
    ids = record.get('causal_topk_ids')
    if not isinstance(ids, torch.Tensor) or ids.dtype != torch.long:
        raise ValueError('causal_topk_ids must be an int64 tensor')
    if tuple(ids.shape) != (len(record['positions']), k):
        raise ValueError('causal_topk_ids shape does not match selected positions and top_k')
    if ids.numel() and ((ids < 0).any() or (ids >= vocabulary_size).any()):
        raise ValueError('causal_topk_ids contains an invalid vocabulary ID')
    return ids


@torch.inference_mode()
def score_hindsight(model, tok, records, a):
    """Return sparse H\\C IDs (or H hidden states for the OPSD control)."""
    if not records:
        return []
    if a.method not in ('ren_opd', 'ren_graft', 'union_topk', 'causal_topk', 'opsd', 'vanilla_opd'):
        raise ValueError(f'Unsupported hindsight method: {a.method}')
    head = base_model(model).get_output_embeddings()
    k, vocabulary_size = min(a.top_k, head.weight.shape[0]), head.weight.shape[0]
    causal = []
    for record in records:
        if not record.get('hindsight_prompt_ids'):
            raise ValueError('Hindsight prompt must be nonempty')
        positions, response = record['positions'], record['response_ids']
        if any(not isinstance(p, int) or p < 0 or p >= len(response) for p in positions):
            raise ValueError('Selected reasoning position is outside the response')
        if positions != sorted(set(positions)):
            raise ValueError('Selected reasoning positions must be unique and increasing')
        if a.method in ('ren_graft', 'union_topk', 'causal_topk'):
            causal.append(_causal_ids(record, k, vocabulary_size))
    if a.method == 'vanilla_opd':
        return [{} for _ in records]
    if a.method == 'causal_topk':
        return [{'correction_ids': ids.detach().cpu().clone()} for ids in causal]
    results = []
    for start in range(0, len(records), a.score_batch_size):
        batch = records[start:start + a.score_batch_size]
        hidden = selected_hidden(model, tok, batch, 'hindsight_prompt_ids', a)
        for offset, (record, h) in enumerate(zip(batch, hidden)):
            if a.method == 'opsd':
                results.append({'student_hidden': h.detach().cpu().clone()})
                continue
            chunks = []
            if a.method == 'ren_opd':
                for position in range(0, h.shape[0], a.logit_chunk_size):
                    hs = head(h[position:position + a.logit_chunk_size]).float().topk(k, dim=-1).indices
                    chunks.append(hs.cpu())
                results.append({'recognition_ids': torch.cat(chunks) if chunks else
                                torch.empty((0, k), dtype=torch.long)})
                continue
            cs = causal[start + offset]
            for position in range(0, h.shape[0], a.logit_chunk_size):
                hs = head(h[position:position + a.logit_chunk_size]).float().topk(k, dim=-1).indices
                c = cs[position:position + len(hs)].to(hs.device)
                novel = ~hs.unsqueeze(-1).eq(c.unsqueeze(-2)).any(-1)
                ids = hs.masked_fill(~novel, -1)
                if a.method == 'union_topk':
                    ids = torch.cat((c, ids), dim=-1)
                chunks.append(ids.cpu())
            columns = k * (2 if a.method == 'union_topk' else 1)
            results.append({'correction_ids': torch.cat(chunks) if chunks else
                            torch.empty((0, columns), dtype=torch.long)})
    return results


def hindsight_worker(a, device_ids, conn):
    """Serve ``sync``, ``score`` and ``stop`` requests through a duplex pipe.

    ``a.initial_checkpoint`` must point to the same initial/resumed checkpoint
    used by Student workers. ``sync`` accepts the complete set of parameters
    trainable in that architecture, including all parameters for full tuning.
    A score is accepted only for the most recently synchronized round.
    """
    request = {}
    try:
        if not a.cpu:
            if len(device_ids) != 1:
                raise ValueError('Each privileged Student worker requires exactly one GPU')
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))
        os.environ.update(LOCAL_RANK='0', RANK='0', WORLD_SIZE='1')
        torch.set_num_threads(getattr(a, 'worker_cpu_threads', 1))
        initial_checkpoint = getattr(a, 'initial_checkpoint', None)
        if a.lora_rank and not initial_checkpoint:
            raise ValueError('A LoRA privileged Student needs the initial Student checkpoint')
        tok = load_tokenizer(a.model)
        model = load_student(a, checkpoint_dir=Path(initial_checkpoint) if initial_checkpoint else None,
                             trainable=True)
        expected_names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
        disable_dropout(model)
        model.requires_grad_(False).eval()
        synchronized_round = None
        conn.send({'op': 'ready', 'role': 'hindsight', 'sync_names': expected_names})
        while True:
            request = conn.recv()
            operation = request.get('op')
            if operation == 'stop':
                conn.send({'op': 'stopped', 'role': 'hindsight'})
                return
            if operation == 'sync':
                round_index = request['round']
                if not isinstance(round_index, int) or round_index < 0:
                    raise ValueError('Snapshot round must be a nonnegative integer')
                if synchronized_round is not None and round_index < synchronized_round:
                    raise ValueError('Cannot synchronize an older Student snapshot')
                started = time.monotonic()
                sync_snapshot(model, request['state'], expected_names)
                synchronized_round = round_index
                conn.send({'op': 'synced', 'role': 'hindsight', 'round': round_index,
                           'seconds': time.monotonic() - started})
            elif operation == 'score':
                if synchronized_round is None or request['round'] != synchronized_round:
                    raise ValueError('Hindsight score must use the synchronized Student round')
                started = time.monotonic()
                result = score_hindsight(model, tok, request['records'], a)
                conn.send({'op': 'scored', 'role': 'hindsight', 'round': synchronized_round,
                           'request_id': request['request_id'], 'records': result,
                           'seconds': time.monotonic() - started})
            else:
                raise ValueError(f'Unknown privileged Student operation: {operation}')
    except EOFError:
        return
    except BaseException as error:
        try:
            conn.send({'op': 'error', 'role': 'hindsight', 'round': request.get('round'),
                       'request_id': request.get('request_id'), 'error': str(error),
                       'traceback': traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        conn.close()
