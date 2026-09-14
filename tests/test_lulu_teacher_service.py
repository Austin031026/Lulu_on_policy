"""Exact scoring, answer isolation and persistent Teacher lifecycle regressions."""
from types import SimpleNamespace

import pytest
import torch

from lulu import teacher_service, training


def _model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.set_num_threads(1)
    torch.manual_seed(713)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=31, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
        attention_dropout=0., tie_word_embeddings=False,
        pad_token_id=0, bos_token_id=1, eos_token_id=2)).eval().requires_grad_(False)


def _args(method='ren_opd', batch=2, chunk=2):
    return SimpleNamespace(method=method, score_batch_size=batch, logit_chunk_size=chunk,
                           max_sequence_tokens=128, cpu=True, dtype='float32')


def _request(round_index=0, request_id='0:0'):
    return {'op': 'score', 'round': round_index, 'request_id': request_id, 'records': [
        {'causal_prompt_ids': [3, 4], 'response_ids': [6, 7, 8, 9], 'positions': [0, 1, 3],
         'correction_ids': torch.tensor([[0, 6, -1], [30, 2, 7], [-1, -1, -1]])},
        {'causal_prompt_ids': [3, 5, 10], 'response_ids': [12, 13], 'positions': [],
         'correction_ids': torch.empty((0, 3), dtype=torch.long)},
        {'causal_prompt_ids': [3, 5, 11], 'response_ids': [12, 13, 14], 'positions': [0, 2],
         'correction_ids': torch.tensor([[4, 1, 2], [7, -1, 3]])},
    ]}


@pytest.mark.parametrize('method', ['ren_opd', 'causal_topk', 'union_topk'])
@pytest.mark.parametrize('batch,chunk', [(1, 1), (2, 2), (8, 32)])
def test_selected_probabilities_equal_dense_teacher_softmax(method, batch, chunk):
    model, tok, args = _model(), SimpleNamespace(pad_token_id=0), _args(method, batch, chunk)
    records = teacher_service.sanitize_score_request(_request(), method, vocab_size=31)['records']
    actual = teacher_service.score_records(model, tok, records, args)
    with torch.inference_mode():
        hidden = training.selected_hidden(model, tok, records, 'causal_prompt_ids', args)
        for record, states, result in zip(records, hidden, actual):
            probabilities = model.get_output_embeddings()(states).float().softmax(-1)
            ids = record['correction_ids']
            expected = probabilities.gather(-1, ids.clamp_min(0)).masked_fill(ids < 0, 0)
            torch.testing.assert_close(result['teacher_probs'], expected, atol=1e-7, rtol=1e-6)
            assert result['teacher_probs'].device.type == 'cpu'
            assert result['teacher_probs'].dtype == torch.float32
            assert not result['teacher_probs'].requires_grad
            assert result['teacher_scored'] is True


def test_vanilla_hidden_reconstructs_teacher_distribution():
    model, tok, args = _model(), SimpleNamespace(pad_token_id=0), _args('vanilla_opd')
    request = _request()
    for record in request['records']:
        del record['correction_ids']
    records = teacher_service.sanitize_score_request(request, args.method)['records']
    actual = teacher_service.score_records(model, tok, records, args)
    with torch.inference_mode():
        expected = training.selected_hidden(model, tok, records, 'causal_prompt_ids', args)
    for wanted, result in zip(expected, actual):
        torch.testing.assert_close(result['teacher_hidden'], wanted)
        assert 'teacher_probs' not in result


@pytest.mark.parametrize('field', ['answer', 'gold_answer', 'hindsight_prompt_ids', 'student_hidden', 'question'])
def test_teacher_rejects_extra_fields_instead_of_silently_receiving_gold(field):
    request = _request()
    request['records'][0][field] = 'private'
    with pytest.raises(ValueError, match='forbidden fields'):
        teacher_service.sanitize_score_request(request, 'ren_opd')


@pytest.mark.parametrize('change', [
    lambda r: r.update(round=-1),
    lambda r: r.update(answer='private'),
    lambda r: r['records'][0].update(positions=[0, 0, 3]),
    lambda r: r['records'][0].update(positions=[0, 1, 4]),
    lambda r: r['records'][0].update(causal_prompt_ids=[]),
    lambda r: r['records'][0].update(correction_ids=torch.zeros(3, 3)),
    lambda r: r['records'][0].update(correction_ids=torch.full((3, 3), -2)),
    lambda r: r['records'][0].update(correction_ids=torch.full((3, 3), 31)),
    lambda r: r['records'][0].update(causal_prompt_ids=[31]),
])
def test_teacher_validates_shapes_positions_and_ids(change):
    request = _request()
    change(request)
    with pytest.raises(ValueError):
        teacher_service.sanitize_score_request(request, 'ren_opd', vocab_size=31)


def test_teacher_refuses_silent_context_truncation():
    with pytest.raises(ValueError, match='refusing truncation'):
        teacher_service.sanitize_score_request(_request(), 'ren_opd', max_sequence_tokens=5)


class _Connection:
    def __init__(self, requests):
        self.requests, self.replies, self.closed = iter(requests), [], False

    def recv(self):
        return next(self.requests)

    def send(self, value):
        self.replies.append(value)

    def close(self):
        self.closed = True


def _mock_loader(monkeypatch):
    calls = []
    model = _model()

    def load(args, world, device):
        calls.append((world, device))
        return model, SimpleNamespace(pad_token_id=0)

    monkeypatch.setattr(teacher_service, '_load_teacher', load)
    # The production worker owns its subprocess environment. Keep tests local.
    for name in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
        monkeypatch.setenv(name, 'test')
    return model, calls


def test_resident_teacher_loads_once_across_updates_and_echoes_versions(monkeypatch):
    _, calls = _mock_loader(monkeypatch)
    conn = _Connection([_request(), _request(1, '1:0'), {'op': 'stop'}])
    teacher_service.teacher_worker(_args(), 0, 1, [], '127.0.0.1', 12345, conn)
    assert len(calls) == 1
    assert [r['op'] for r in conn.replies] == ['ready', 'scored', 'scored', 'stopped']
    assert [(r['round'], r['request_id']) for r in conn.replies if r['op'] == 'scored'] == [(0, '0:0'), (1, '1:0')]
    assert conn.closed


def test_vanilla_head_transferred_once_at_ready(monkeypatch):
    model, calls = _mock_loader(monkeypatch)
    conn = _Connection([_request(), _request(1, '1:0'), {'op': 'stop'}])
    teacher_service.teacher_worker(_args('vanilla_opd'), 0, 1, [], '127.0.0.1', 12345, conn)
    torch.testing.assert_close(conn.replies[0]['teacher_head']['weight'], model.get_output_embeddings().weight)
    assert all('teacher_head' not in result for result in conn.replies[1:])
    assert len(calls) == 1


@pytest.mark.parametrize('second, message', [(_request(), 'duplicate request_id'), (_request(0, 'late'), 'stale scoring round')])
def test_duplicate_or_stale_requests_report_error_and_exit(monkeypatch, second, message):
    _mock_loader(monkeypatch)
    first = _request(1, 'new') if message == 'stale scoring round' else _request()
    conn = _Connection([first, second])
    with pytest.raises(RuntimeError, match=message):
        teacher_service.teacher_worker(_args(), 0, 1, [], '127.0.0.1', 12345, conn)
    assert conn.replies[-1]['op'] == 'error'
    assert message in conn.replies[-1]['error']
    assert conn.replies[-1]['request_id'] == second['request_id']
    assert conn.closed


def test_native_tp_two_gpu_matches_dense_model_across_rounds(tmp_path):
    """Opt-in hardware regression: LULU_TEST_TP_GPUS=6,7 pytest ... -k native_tp."""
    import multiprocessing as mp
    import os
    import socket
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    ids = os.environ.get('LULU_TEST_TP_GPUS', '').split(',')
    if len(ids) != 2 or not all(ids):
        pytest.skip('Set LULU_TEST_TP_GPUS to two idle visible CUDA IDs for native TP validation')
    model = _model()
    model.save_pretrained(tmp_path)
    vocabulary = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[UNK]': 3,
                  **{str(i): i for i in range(4, 31)}}
    core = Tokenizer(WordLevel(vocabulary, unk_token='[UNK]'))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, pad_token='[PAD]',
        bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]')
    tokenizer.save_pretrained(tmp_path)
    args = _args()
    args.cpu, args.model, args.teacher_model, args.worker_timeout = False, str(tmp_path), str(tmp_path), 90
    request = _request()
    expected = teacher_service.score_records(model, tokenizer, request['records'], args)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    ctx = mp.get_context('spawn')
    parent, child = ctx.Pipe()
    processes = [ctx.Process(target=teacher_service.teacher_worker,
        args=(args, rank, 2, ids, '127.0.0.1', port, child if rank == 0 else None)) for rank in range(2)]
    try:
        for process in processes:
            process.start()
        child.close()
        assert parent.poll(90), [p.exitcode for p in processes]
        ready = parent.recv()
        assert ready['op'] == 'ready', ready
        assert ready['backend'] == 'transformers_tp'
        for round_index in (0, 1):
            request.update(round=round_index, request_id=f'tp2:{round_index}')
            parent.send(request)
            assert parent.poll(60), [p.exitcode for p in processes]
            actual = parent.recv()
            assert actual['op'] == 'scored', actual
            assert actual['round'] == round_index
            for wanted, result in zip(expected, actual['records']):
                torch.testing.assert_close(result['teacher_probs'], wanted['teacher_probs'], atol=2e-6, rtol=1e-5)
        parent.send({'op': 'stop'})
        assert parent.poll(30)
        assert parent.recv()['op'] == 'stopped'
        for process in processes:
            process.join(30)
        assert [p.exitcode for p in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(5)
        parent.close()


@pytest.mark.parametrize('local_output', [True, False])
@pytest.mark.parametrize('already_replicated', [True, False])
def test_native_rowwise_reduction_is_completed_before_subsequent_dispatch(monkeypatch, local_output, already_replicated):
    from transformers.integrations.tensor_parallel import ALL_PARALLEL_STYLES, RowwiseParallel
    from torch.distributed.tensor import Replicate, Partial
    original = ALL_PARALLEL_STYLES['rowwise']
    monkeypatch.setitem(ALL_PARALLEL_STYLES, 'rowwise', original)
    teacher_service.install_synchronous_rowwise_tp()
    replacement = ALL_PARALLEL_STYLES['rowwise']
    # monkeypatch restores the registry even if an assertion fails; production
    # registration occurs only inside the isolated Teacher subprocess.
    assert isinstance(replacement, RowwiseParallel)
    assert replacement.partition_tensor.__func__ is RowwiseParallel.partition_tensor
    layout = (Replicate(),)
    calls = []

    class Reduction:
        placements = layout if already_replicated else (Partial(),)

        def redistribute(self, *, placements, async_op):
            assert placements == layout
            assert async_op is False
            calls.append('reduce_completed')
            self.placements = placements
            return self

        def to_local(self):
            if not already_replicated:
                assert calls == ['reduce_completed']
            calls.append('local_output')
            return torch.tensor([2., 3.])

    output = Reduction()
    actual = replacement._prepare_output_fn(layout, local_output, SimpleNamespace(), output, None)
    if local_output:
        torch.testing.assert_close(actual, torch.tensor([2., 3.]))
    else:
        assert actual is output
    assert calls.count('reduce_completed') == int(not already_replicated)


def _synchronous_rowwise_gloo_worker(rank, rendezvous, output):
    """Exercise the actual native TP hooks on 8k activations without a GPU/LM."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard
    from transformers.integrations.tensor_parallel import ALL_PARALLEL_STYLES

    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    try:
        mesh = init_device_mesh('cpu', (2,))
        teacher_service.install_synchronous_rowwise_tp()
        weights = torch.linspace(-.01, .01, 1024*32).reshape(1024, 32)
        inputs = torch.linspace(0., .5, 8192*32).reshape(1, 8192, 32)
        linear = torch.nn.Linear(32, 1024, bias=False)
        local_weights = weights[:, rank*16:(rank+1)*16].contiguous()
        linear.weight = torch.nn.Parameter(DTensor.from_local(local_weights, mesh, (Shard(1),),
                                                              run_check=False), requires_grad=False)
        native_style = ALL_PARALLEL_STYLES['rowwise']
        native_style.prepare_module_tp(linear, mesh)
        with torch.inference_mode():
            actual = linear(inputs[..., rank*16:(rank+1)*16].contiguous())
            expected = torch.nn.functional.linear(inputs, weights)
        assert not isinstance(actual, DTensor)
        assert tuple(actual.shape) == (1, 8192, 1024)
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
        if rank == 0:
            torch.save({'max_abs_error': float((actual-expected).abs().max()),
                        'shape': list(actual.shape), 'ranks': 2, 'device': actual.device.type}, output)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason='PyTorch distributed unavailable')
def test_native_synchronous_rowwise_gloo_8192_token_projection_equals_dense(tmp_path):
    output = tmp_path/'rowwise_8192.pt'
    torch.multiprocessing.spawn(_synchronous_rowwise_gloo_worker,
        args=(str(tmp_path/'gloo_init'), str(output)), nprocs=2, join=True)
    result = torch.load(output, weights_only=True)
    assert result['device'] == 'cpu'
    assert result['ranks'] == 2
    assert result['shape'] == [1, 8192, 1024]
    assert result['max_abs_error'] < 2e-7
