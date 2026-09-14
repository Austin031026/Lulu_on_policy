"""Persistent pipeline scheduling, IPC isolation and distributed-update regressions."""
from __future__ import annotations

import copy
import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from lulu import persistent, training
from lulu.teacher_service import sanitize_score_request
from test_lulu_training import _student, _records


def _args(**overrides):
    args = training.parser().parse_args(['--train-data', 'unused.jsonl', '--output-dir', 'unused',
        '--gpus', '0,1,2,3,4,5,6,7'])
    vars(args).update(overrides)
    return args


def test_default_eight_gpu_allocation_is_five_student_one_hindsight_two_teacher():
    assert persistent.allocate_roles(_args()) == {
        'student': ['0', '1', '2', '3', '4'], 'hindsight': ['5'], 'teacher': ['6', '7']}


def test_explicit_role_allocation_preserves_visible_gpu_ids_and_disjointness():
    roles = persistent.allocate_roles(_args(student_gpus='2,3,4,5,6', hindsight_gpus='1', teacher_gpus='0,7'))
    assert roles == {'student': ['2', '3', '4', '5', '6'], 'hindsight': ['1'], 'teacher': ['0', '7']}
    assert len({gpu for group in roles.values() for gpu in group}) == 8


@pytest.mark.parametrize('options, message', [
    ({'student_gpus': '0,1', 'hindsight_gpus': '1', 'teacher_gpus': '6,7'}, 'disjoint'),
    ({'student_gpus': '0,0'}, 'Invalid student GPU'),
    ({'teacher_gpus': '8'}, 'Invalid teacher GPU'),
    ({'hindsight_gpus': '4,5'}, 'exactly one privileged'),
    ({'gpus': '0,1,2'}, 'at least one Student'),
    ({'gpus': '0,1'}, 'Insufficient GPUs'),
])
def test_invalid_or_overlapping_gpu_roles_fail_before_launch(options, message):
    with pytest.raises(ValueError, match=message):
        persistent.allocate_roles(_args(**options))


@pytest.mark.parametrize('method, expected', [
    ('opsd', {'student': list('0123456'), 'hindsight': ['7'], 'teacher': None}),
    ('causal_topk', {'student': list('012345'), 'hindsight': None, 'teacher': ['6', '7']}),
    ('vanilla_opd', {'student': list('012345'), 'hindsight': None, 'teacher': ['6', '7']}),
])
def test_methods_only_reserve_the_roles_they_use(method, expected):
    assert persistent.allocate_roles(_args(method=method)) == expected


def test_teacher_payload_cannot_include_gold_or_hindsight_hidden_states():
    records = [{'index': 9, 'causal_prompt_ids': [3, 4], 'response_ids': [5, 6],
        'positions': [0, 1], 'correction_ids': torch.tensor([[1, -1], [2, 3]]),
        'gold_answer': 'SECRET GOLD', 'hindsight_prompt_ids': [29, 30],
        'student_hidden': torch.randn(2, 16), 'causal_topk_ids': torch.tensor([[7, 8], [9, 10]])}]
    payload = persistent.teacher_payload(records)
    clean = sanitize_score_request({'op': 'score', 'round': 0, 'request_id': 9, 'records': payload},
                                   'ren_opd', vocab_size=31)
    assert set(clean['records'][0]) == {'causal_prompt_ids', 'response_ids', 'positions', 'correction_ids'}
    assert 'gold_answer' in records[0]  # Boundary extraction also leaves the caller's record intact.


class _FakeConnection:
    def __init__(self, harness, role, rank=None):
        self.harness, self.role, self.rank = harness, role, rank

    def send(self, message):
        self.harness.sent(self, copy.deepcopy(message))


class _PipelineHarness:
    """Deterministic asynchronous response queues with recorded causal ordering."""
    def __init__(self, corrupt=None, method='ren_opd'):
        self.events, self.pending, self.corrupt = [], [], corrupt
        self.students = [_FakeConnection(self, 'student', rank) for rank in range(2)]
        self.hindsight = _FakeConnection(self, 'hindsight') if method != 'causal_topk' else None
        self.teacher = _FakeConnection(self, 'teacher') if method != 'opsd' else None
        self.method = method

    def _queue(self, connection, message):
        if self.corrupt:
            self.corrupt(connection, message)
        self.pending.append((connection, message))

    def sent(self, conn, message):
        op, version = message['op'], message.get('round')
        self.events.append(('send', conn.role, op, conn.rank))
        if op == 'snapshot':
            self._queue(conn, {'op': 'snapshot', 'round': version, 'state': {'adapter': torch.tensor([version])}})
        elif op == 'sync':
            assert message['state']['adapter'].item() == version
            self._queue(conn, {'op': 'synced', 'round': version})
        elif op == 'collect':
            records = [{'index': index, 'causal_prompt_ids': [3, 4],
                'hindsight_prompt_ids': [3, 4, 29, 30], 'response_ids': [5, 6],
                'positions': [0, 1], 'causal_topk_ids': torch.tensor([[1, 2], [3, 4]])}
                for index in (conn.rank, conn.rank+2)]
            self._queue(conn, {'op': 'batch', 'rank': conn.rank, 'round': version, 'records': records})
            self._queue(conn, {'op': 'collected', 'rank': conn.rank, 'round': version, 'response_tokens': 4})
        elif op == 'score':
            if conn.role == 'teacher':
                sanitize_score_request(message, self.method, vocab_size=31)
                result = [{'teacher_probs': torch.full_like(record['correction_ids'], .02, dtype=torch.float32)}
                          for record in message['records']]
            elif self.method == 'opsd':
                result = [{'student_hidden': torch.ones(len(record['positions']), 16)} for record in message['records']]
            else:
                result = [{'correction_ids': torch.tensor([[7, -1], [8, 9]])} for _ in message['records']]
            self._queue(conn, {'op': 'scored', 'round': version, 'request_id': message['request_id'],
                               'records': result, 'seconds': .125})
        elif op == 'update':
            final_role = 'hindsight' if self.method == 'opsd' else 'teacher'
            assert sum(e[:3] == ('receive', final_role, 'scored') for e in self.events) == 2
            assert set(message['records']) == {conn.rank, conn.rank+2}
            for target in message['records'].values():
                assert 'hindsight_prompt_ids' not in target
                assert 'causal_prompt_ids' not in target
                if self.method != 'opsd':
                    assert target['teacher_scored']
            self._queue(conn, {'op': 'updated', 'round': version, 'rank': conn.rank,
                'metrics': {'forward_kl': .2, 'round': version, 'checkpoint': 'latest'}})
        else:
            raise AssertionError(f'Unexpected request {op}')

    def receive(self, connections):
        for index, (conn, message) in enumerate(self.pending):
            if conn in connections:
                self.pending.pop(index)
                self.events.append(('receive', conn.role, message['op'], conn.rank))
                return conn, message
        raise AssertionError('Pipeline waited for a response that no worker can produce')

    def expect(self, conn, op):
        _, message = self.receive([conn])
        assert message['op'] == op
        return message


@pytest.mark.parametrize('method', ['ren_opd', 'opsd', 'causal_topk'])
def test_round_synchronizes_h_before_rollout_overlaps_scoring_and_waits_for_all_targets(method):
    workers = _PipelineHarness(method=method)
    args = _args(method=method, round=3, global_batch_prompts=4)
    result = persistent.pipeline_round(args, workers, workers.students, workers.hindsight, workers.teacher)
    assert result['response_tokens'] == 8
    assert result['round'] == 3
    assert not workers.pending
    if workers.hindsight:
        synced = workers.events.index(('receive', 'hindsight', 'synced', None))
        collect = workers.events.index(('send', 'student', 'collect', 0))
        assert synced < collect
        # H starts on rank 0's batch while Student rank 1 still has collect work.
        first_h = workers.events.index(('send', 'hindsight', 'score', None))
        last_collect = workers.events.index(('receive', 'student', 'collected', 1))
        assert first_h < last_collect
    if workers.teacher:
        assert result['teacher_seconds'] == pytest.approx(.25)
    if workers.hindsight and workers.teacher:
        teacher_start = workers.events.index(('send', 'teacher', 'score', None))
        h_finished = max(i for i, event in enumerate(workers.events) if event[:3] == ('receive', 'hindsight', 'scored'))
        assert teacher_start < h_finished


@pytest.mark.parametrize('role,op,field,value,match', [
    ('student', 'batch', 'round', 2, 'stale'),
    ('teacher', 'scored', 'round', 2, 'stale'),
    ('hindsight', 'scored', 'round', 2, 'stale'),
    ('teacher', 'scored', 'request_id', 999, 'Mismatched'),
    ('hindsight', 'scored', 'request_id', 999, 'Mismatched'),
    ('teacher', 'scored', 'records', [], 'Mismatched'),
    ('student', 'snapshot', 'round', 2, 'stale|snapshot'),
    ('hindsight', 'synced', 'round', 2, 'stale|sync'),
])
def test_stale_or_mismatched_service_messages_never_trigger_update(role, op, field, value, match):
    def corrupt(conn, message):
        if conn.role == role and message['op'] == op:
            message[field] = value
    workers = _PipelineHarness(corrupt=corrupt)
    with pytest.raises(RuntimeError, match=match):
        persistent.pipeline_round(_args(round=3, global_batch_prompts=4), workers,
                                  workers.students, workers.hindsight, workers.teacher)
    assert not any(event[0] == 'send' and event[2] == 'update' for event in workers.events)


def test_incomplete_pipeline_cannot_update_any_student():
    workers = _PipelineHarness()
    with pytest.raises(RuntimeError, match='Incomplete'):
        persistent.pipeline_round(_args(global_batch_prompts=5), workers, workers.students,
                                  workers.hindsight, workers.teacher)
    assert not any(event[0] == 'send' and event[2] == 'update' for event in workers.events)


def test_duplicate_target_is_rejected_before_update():
    def corrupt(conn, message):
        if conn.role == 'student' and message['op'] == 'batch':
            message['records'][1]['index'] = message['records'][0]['index']
    workers = _PipelineHarness(corrupt=corrupt)
    with pytest.raises(RuntimeError, match='Duplicate target'):
        persistent.pipeline_round(_args(global_batch_prompts=4), workers, workers.students,
                                  workers.hindsight, workers.teacher)


class _ByteBuffer:
    def __init__(self):
        self.data = None

    def send_bytes(self, data):
        assert isinstance(data, bytes)
        self.data = data

    def recv_bytes(self):
        return self.data


def test_memory_channel_round_trip_is_independent_of_shared_tensor_storage():
    raw = _ByteBuffer()
    channel = persistent.MemoryChannel(raw)
    original = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    channel.send({'round': 4, 'hidden': original})
    original.add_(100)
    received = channel.recv()
    torch.testing.assert_close(received['hidden'], torch.arange(12, dtype=torch.bfloat16).reshape(3, 4))
    assert received['round'] == 4
    assert received['hidden'].untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
    assert not received['hidden'].is_shared()


def test_first_run_in_existing_empty_directory_publishes_manifest_before_workers(tmp_path, monkeypatch):
    data = tmp_path/'train.jsonl'
    data.write_text('{}\n')
    output = tmp_path/'run'
    output.mkdir()
    args = _args(cpu=True, dtype='float32', train_data=str(data), output_dir=str(output))
    closed = []

    class AbortWorkers:
        def __init__(self, timeout):
            pass

        def launch(self, *args, **kwargs):
            assert (output/'run_config.json').is_file()
            raise RuntimeError('intentional launch stop')

        def close(self):
            closed.append(True)

    monkeypatch.setattr(persistent, 'Workers', AbortWorkers)
    with pytest.raises(RuntimeError, match='intentional launch stop'):
        persistent.run_persistent(args)
    assert json.loads((output/'run_config.json').read_text())['train_sha256']
    assert closed == [True]


def _resident_fixture():
    model = _student()
    records, frozen_head, teacher_head, tokenizer = _records(copy.deepcopy(model).requires_grad_(False).eval(),
                                                           'ren_opd', include_empty=True)
    # Rank 0 has both active trajectories; rank 1 gets only an empty trajectory
    # plus a padding dummy, which must still participate in every DDP reduction.
    records = [records[1], records[0], records[2]]
    args = _args(cpu=True, dtype='float32', global_batch_prompts=3, lora_rank=0,
                 max_sequence_tokens=128, max_prompt_tokens=64, learning_rate=1e-4,
                 logit_chunk_size=2, top_k=3)
    step = training.DistillationStep(model, frozen_head, teacher_head, tokenizer, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.)
    return args, model, records, step, optimizer


def _resident_ddp_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    args, model, records, step, optimizer = _resident_fixture()
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    try:
        wrapped = DDP(step, broadcast_buffers=False)
        dummy = dict(causal_prompt_ids=[3, 4], response_ids=[5], positions=[],
                     snapshot_round=0, dummy=True)
        metrics = persistent.update_records(step, wrapped, optimizer, records[rank::2], dummy, args, rank, 2)
        if rank == 0:
            torch.save({'state': model.state_dict(), 'optimizer': optimizer.state_dict(), 'metrics': metrics}, output)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason='PyTorch distributed unavailable')
def test_resident_ddp_update_matches_nonempty_global_mean_with_empty_and_dummy_rank(tmp_path):
    args, model, records, step, optimizer = _resident_fixture()
    reference = sum(step([record]) for record in records)/2
    reference.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    output = tmp_path/'distributed.pt'
    torch.multiprocessing.spawn(_resident_ddp_worker,
        args=(str(tmp_path/'gloo_init'), str(output)), nprocs=2, join=True)
    actual = torch.load(output, weights_only=True)
    for index, (name, parameter) in enumerate(model.named_parameters()):
        expected_state = optimizer.state[parameter]
        received_state = actual['optimizer']['state'][index]
        torch.testing.assert_close(received_state['exp_avg'], expected_state['exp_avg'], atol=4e-9, rtol=3e-4)
        torch.testing.assert_close(received_state['exp_avg_sq'], expected_state['exp_avg_sq'], atol=1e-11, rtol=3e-4)
        meaningful = parameter.grad.abs() > 1e-6
        torch.testing.assert_close(actual['state'][name][meaningful], parameter.detach()[meaningful], atol=2e-6, rtol=3e-5)
        assert torch.isfinite(actual['state'][name]).all()
    assert actual['metrics']['supervised_trajectories'] == 2
    assert actual['metrics']['trajectories'] == 3
    assert actual['metrics']['reasoning_tokens'] == 6
    assert actual['metrics']['forward_kl'] == pytest.approx(reference.item(), abs=2e-7, rel=2e-5)


@pytest.mark.parametrize('missing', ['snapshot', 'target'])
def test_update_records_rejects_stale_snapshot_or_missing_teacher_target(missing):
    args, model, records, step, optimizer = _resident_fixture()
    if missing == 'snapshot':
        records[0]['snapshot_round'] = 9
        expected = 'Stale Student'
    else:
        del records[0]['teacher_scored']
        expected = 'Missing Teacher target'
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    with pytest.raises(RuntimeError, match=expected):
        persistent.update_records(step, step, optimizer, records, records[0], args, 0, 1)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, original[name])


def test_collected_cpu_frozen_cache_is_normal_tensor_and_supports_backward(monkeypatch):
    model = _student()
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    args = _args(cpu=True, dtype='float32', global_batch_prompts=1, lora_rank=0,
                 max_sequence_tokens=128, max_prompt_tokens=64, max_new_tokens=3,
                 logit_chunk_size=2, top_k=3)
    rows = [{'id': 'tiny', 'messages': [{'role': 'user', 'content': 'problem'}], 'gold_answer': '7'}]
    monkeypatch.setattr(training, 'build_prompt_views', lambda *args, **kwargs: {
        'causal_prompt_ids': [3, 4], 'hindsight_prompt_ids': [3, 4, 20]})
    monkeypatch.setattr(training, 'reasoning_token_mask', lambda *args, **kwargs: [True, True, False])

    def generate(**kwargs):
        response = torch.tensor([[6, 7, 2]], dtype=torch.long)
        return torch.cat((kwargs['input_ids'], response), dim=1)

    monkeypatch.setattr(model, 'generate', generate)
    batches = list(persistent.collect_batches(model, tokenizer, rows, args, 0, 1))
    assert len(batches) == 1
    records, payload = batches[0]
    hidden = records[0]['student_hidden']
    assert hidden.device.type == 'cpu'
    assert not hidden.is_inference()
    assert not hidden.requires_grad
    assert 'hindsight_prompt_ids' not in records[0]
    assert 'student_hidden' not in payload[0]
    records[0].update(correction_ids=torch.tensor([[6, -1], [7, -1]]),
                      teacher_probs=torch.tensor([[.2, 0.], [.15, 0.]]), teacher_scored=True)
    frozen_head = copy.deepcopy(model.get_output_embeddings()).requires_grad_(False)
    step = training.DistillationStep(model, frozen_head, None, tokenizer, args)
    loss = step(records)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in model.parameters())
