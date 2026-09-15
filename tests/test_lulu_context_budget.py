"""Long response configuration and exact prefix indexing without model compute."""
from types import SimpleNamespace
import pytest
import torch
from lulu import training


def test_default_8192_response_budget_and_conservative_batches():
    args = training.parser().parse_args(['--train-data', 'unused', '--output-dir', 'unused'])
    training.validate_args(args)
    assert args.max_new_tokens == 8192
    assert args.max_prompt_tokens == 4096
    assert args.max_sequence_tokens == 16384
    assert (args.rollout_batch_size, args.score_batch_size, args.train_micro_batch_size) == (4, 1, 1)
    assert args.save_every == 20
    assert args.lora_rank == 0
    assert args.kl_direction == 'forward'
    assert args.kl_diagnostics is False
    assert 0.05 in args.kl_diagnostic_thresholds
    args.max_sequence_tokens = 8192
    with pytest.raises(ValueError, match='max_prompt_tokens.*max_new_tokens'):
        training.validate_args(args)


def test_reverse_kl_requires_an_explicit_target_and_enables_shadow_diagnostics():
    args = training.parser().parse_args([
        '--train-data', 'unused', '--output-dir', 'unused', '--kl-direction', 'reverse'])
    training.validate_args(args)
    assert training.kl_diagnostics_enabled(args)

    args.method = 'ren_weighted_opd'
    with pytest.raises(ValueError, match='reverse KL is not defined'):
        training.validate_args(args)


def test_pre_kl_direction_manifest_gets_forward_compatible_defaults():
    old = {'backend': 'persistent', 'method': 'ren_opd'}
    normalized = training.with_kl_defaults(old)
    assert normalized['kl_direction'] == 'forward'
    assert normalized['kl_diagnostics'] is False
    assert normalized['kl_diagnostic_thresholds'] == training.DEFAULT_KL_DIAGNOSTIC_THRESHOLDS


def test_last_reasoning_token_at_8192_response_limit_uses_exact_prefix():
    # A tiny stand-in checks layout/indexing only, with no attention or GPU work.
    class Decoder:
        def __call__(self, **batch):
            length = batch['input_ids'].shape[1]
            return SimpleNamespace(last_hidden_state=torch.arange(length).reshape(1, length, 1))
    class Model:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))
        def get_decoder(self):
            return Decoder()
    record = {'causal_prompt_ids': [3]*4096, 'hindsight_prompt_ids': [4]*4096,
              'response_ids': [5]*8192, 'positions': [0, 4095, 8191]}
    for view in ('causal_prompt_ids', 'hindsight_prompt_ids'):
        selected = training.selected_hidden(Model(), SimpleNamespace(pad_token_id=0), [record], view,
                                             SimpleNamespace(max_sequence_tokens=16384))[0]
        assert selected[:, 0].tolist() == [4095, 8190, 12286]
        with pytest.raises(ValueError, match='exceeds max_sequence_tokens'):
            training.selected_hidden(Model(), SimpleNamespace(pad_token_id=0), [record], view,
                                      SimpleNamespace(max_sequence_tokens=8192))
