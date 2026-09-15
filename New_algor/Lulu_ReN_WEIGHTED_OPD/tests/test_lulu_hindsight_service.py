"""Snapshot consistency and exact sparse support for resident H scoring."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from lulu import hindsight_service as service
from lulu import training


def _student(lora=False):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.set_num_threads(1)
    torch.manual_seed(72)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=31, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
        attention_dropout=.2, tie_word_embeddings=False, pad_token_id=0,
        bos_token_id=1, eos_token_id=2))
    if lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4,
            target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                torch.nn.init.normal_(parameter, std=.03)
    return model


def _args(method='ren_opd', chunk=2):
    return SimpleNamespace(method=method, max_sequence_tokens=128, top_k=4,
        logit_chunk_size=chunk, score_batch_size=2, cpu=True, model='tiny',
        lora_rank=0, initial_checkpoint=None)


def _records(model):
    records = [
        {'causal_prompt_ids': [3, 4], 'hindsight_prompt_ids': [3, 4, 20, 21],
         'response_ids': [6, 7, 8, 9, 10], 'positions': [0, 1, 2, 4]},
        {'causal_prompt_ids': [3, 5, 11], 'hindsight_prompt_ids': [3, 5, 11, 24],
         'response_ids': [12, 13, 14], 'positions': [0, 2]},
        {'causal_prompt_ids': [3, 4], 'hindsight_prompt_ids': [3, 4, 20],
         'response_ids': [2], 'positions': []},
    ]
    tok = SimpleNamespace(pad_token_id=0)
    with torch.no_grad():
        for record, hidden in zip(records, training.selected_hidden(model, tok, records,
                                                                    'causal_prompt_ids', _args())):
            record['causal_topk_ids'] = training.base_model(model).get_output_embeddings()(hidden).topk(4).indices
    return records, tok


@pytest.mark.parametrize('method', ['ren_opd', 'ren_graft', 'union_topk', 'opsd', 'causal_topk', 'vanilla_opd'])
@pytest.mark.parametrize('chunk', [1, 20])
def test_hindsight_results_equal_dense_forward_without_exposing_prompts(method, chunk):
    model = _student().eval().requires_grad_(False)
    records, tok = _records(model)
    result = service.score_hindsight(model, tok, records, _args(method, chunk))
    with torch.no_grad():
        dense_h = training.selected_hidden(model, tok, records, 'hindsight_prompt_ids', _args())
        for record, hidden, actual in zip(records, dense_h, result):
            assert not {'hindsight_prompt_ids', 'gold_answer', 'response_ids'} & actual.keys()
            if method == 'vanilla_opd':
                assert actual == {}
            elif method == 'opsd':
                torch.testing.assert_close(actual['student_hidden'], hidden)
                assert actual['student_hidden'].device.type == 'cpu'
            else:
                cs = record['causal_topk_ids']
                hs = training.base_model(model).get_output_embeddings()(hidden).topk(4).indices
                if method == 'ren_opd':
                    torch.testing.assert_close(actual['recognition_ids'], hs)
                    assert actual['recognition_ids'].device.type == 'cpu'
                else:
                    novel = hs.masked_fill(hs.unsqueeze(-1).eq(cs.unsqueeze(-2)).any(-1), -1)
                    expected = cs if method == 'causal_topk' else torch.cat((cs, novel), -1) if method == 'union_topk' else novel
                    torch.testing.assert_close(actual['correction_ids'], expected)
                    assert actual['correction_ids'].device.type == 'cpu'


@pytest.mark.parametrize('lora', [False, True])
def test_sync_tracks_current_student_parameters_and_frozen_predictions(lora):
    student = _student(lora=lora)
    names = tuple(name for name, parameter in student.named_parameters() if parameter.requires_grad)
    resident = copy.deepcopy(student).requires_grad_(False).eval()
    records, tok = _records(resident)
    with torch.no_grad():
        for name, parameter in student.named_parameters():
            if name in names:
                parameter.add_(torch.randn_like(parameter) * .04)
    snapshot = {name: p.detach().cpu().clone() for name, p in student.named_parameters() if name in names}
    service.sync_snapshot(resident, snapshot, names)
    student.eval()
    expected = training.selected_hidden(student, tok, records, 'hindsight_prompt_ids', _args())
    actual = training.selected_hidden(resident, tok, records, 'hindsight_prompt_ids', _args())
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want)
    assert all(not p.requires_grad for p in resident.parameters())
    assert not resident.training
    with torch.no_grad():
        for value in snapshot.values():
            value.add_(1)
    # Copying leaves no dependency on controller-owned snapshot tensor storage.
    for name, p in resident.named_parameters():
        torch.testing.assert_close(p, dict(student.named_parameters())[name])


def test_bad_sync_is_rejected_before_partial_parameter_copy():
    model = _student().eval()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    state = {n: p.clone() + 1 for n, p in before.items()}
    state[next(reversed(state))] = torch.zeros(1)
    with pytest.raises(ValueError, match='shape/dtype'):
        service.sync_snapshot(model, state, tuple(before))
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name])
    with pytest.raises(ValueError, match='parameter mismatch'):
        service.sync_snapshot(model, {}, tuple(before))
    with pytest.raises(ValueError, match='parameter mismatch'):
        service.sync_snapshot(model, {**before, 'extra': torch.zeros(1)}, tuple(before))


class _ScriptedPipe:
    def __init__(self, requests):
        self.requests, self.responses, self.closed = iter(requests), [], False

    def recv(self):
        return next(self.requests)

    def send(self, message):
        self.responses.append(message)

    def close(self):
        self.closed = True


def test_worker_loads_once_and_synchronizes_each_round(monkeypatch):
    model = _student(lora=True)
    names = tuple(n for n, p in model.named_parameters() if p.requires_grad)
    states = [{n: p.detach().clone() for n, p in model.named_parameters() if n in names}]
    states.append({n: p + .025 for n, p in states[0].items()})
    records, tok = _records(model.eval())
    calls = []
    monkeypatch.setattr(service, 'load_student', lambda *a, **k: calls.append(k) or model)
    monkeypatch.setattr(service, 'load_tokenizer', lambda *a: tok)
    a = _args('opsd')
    a.lora_rank, a.initial_checkpoint = 2, '/initial'
    pipe = _ScriptedPipe([
        {'op': 'sync', 'round': 0, 'state': states[0]},
        {'op': 'score', 'round': 0, 'request_id': 'a', 'records': records},
        {'op': 'sync', 'round': 1, 'state': states[1]},
        {'op': 'score', 'round': 1, 'request_id': 'b', 'records': records},
        {'op': 'stop'},
    ])
    service.hindsight_worker(a, [], pipe)
    assert len(calls) == 1
    assert pipe.closed
    assert [m['op'] for m in pipe.responses] == ['ready', 'synced', 'scored', 'synced', 'scored', 'stopped']
    assert pipe.responses[0]['sync_names'] == names
    assert pipe.responses[2]['round'] == 0 and pipe.responses[4]['round'] == 1
    assert not torch.allclose(pipe.responses[2]['records'][0]['student_hidden'],
                              pipe.responses[4]['records'][0]['student_hidden'])
    assert model.config.attention_dropout == 0


@pytest.mark.parametrize('score_round', [None, 0, 2])
def test_worker_refuses_missing_or_stale_snapshot(monkeypatch, score_round):
    model = _student()
    records, tok = _records(model.eval())
    state = {n: p.detach().clone() for n, p in model.named_parameters()}
    monkeypatch.setattr(service, 'load_student', lambda *a, **k: model)
    monkeypatch.setattr(service, 'load_tokenizer', lambda *a: tok)
    requests = [] if score_round is None else [{'op': 'sync', 'round': 1, 'state': state}]
    requests.append({'op': 'score', 'round': score_round or 0, 'request_id': 7, 'records': records})
    pipe = _ScriptedPipe(requests)
    with pytest.raises(ValueError, match='synchronized Student round'):
        service.hindsight_worker(_args(), [], pipe)
    assert pipe.responses[-1]['op'] == 'error'
    assert pipe.responses[-1]['request_id'] == 7
    assert pipe.closed


def test_score_rejects_invalid_causal_support_and_positions():
    model = _student().eval().requires_grad_(False)
    records, tok = _records(model)
    records[0]['causal_topk_ids'][0, 0] = -1
    with pytest.raises(ValueError, match='vocabulary ID'):
        service.score_hindsight(model, tok, records, _args())
    records[0]['causal_topk_ids'][0, 0] = 2
    records[0]['positions'] = [0, 0, 2, 4]
    with pytest.raises(ValueError, match='unique and increasing'):
        service.score_hindsight(model, tok, records, _args())
