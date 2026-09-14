"""Recovery and retention behavior without loading an LLM or using a GPU."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from lulu.checkpoints import CheckpointManager


class FakeModel:
    def __init__(self, value=0, fail=False):
        self.value = value
        self.fail = fail

    def save_pretrained(self, path, *, safe_serialization):
        assert safe_serialization
        (Path(path) / "weights.json").write_text(json.dumps({"value": self.value}))
        if self.fail:
            raise RuntimeError("interrupted weight write")


class FakeTokenizer:
    def save_pretrained(self, path):
        (Path(path) / "tokenizer.json").write_text('{}')


class FakeOptimizer:
    def __init__(self, step):
        self.step = step

    def state_dict(self):
        return {"state": {0: {"step": torch.tensor(self.step), "exp_avg": torch.tensor([.5])}}}


def save(manager, step, **kwargs):
    return manager.save(FakeModel(step), FakeTokenizer(), FakeOptimizer(step), step,
                        {"completed_rounds": step, "model": "tiny"}, **kwargs)


def names(manager):
    return sorted(p.name for p in manager.directory.iterdir()
                  if p.is_dir() and not p.is_symlink())


def test_default_cadence_latest_weights_and_optimizer_survive_reload(tmp_path):
    manager = CheckpointManager(tmp_path)
    for step in range(43):
        save(manager, step)
    assert names(manager) == ['step_000000', 'step_000020', 'step_000040', 'step_000042']
    assert manager.latest_link.is_symlink()
    resumed = CheckpointManager(tmp_path)
    latest = resumed.latest_path()
    assert latest.name == 'step_000042'
    assert json.loads((latest / 'weights.json').read_text()) == {'value': 42}
    optimizer = torch.load(latest / 'optimizer.pt', weights_only=True)
    assert optimizer['state'][0]['step'].item() == 42
    assert resumed.latest_metadata()['completed_updates'] == 42
    manifest = json.loads((tmp_path / 'latest.json').read_text())
    assert manifest['completed_rounds'] == 42
    assert manifest['checkpoint'] == str(latest)


def test_failed_model_save_preserves_latest_and_retry_recovers(tmp_path):
    manager = CheckpointManager(tmp_path, save_every=3)
    first = save(manager, 0)
    with pytest.raises(RuntimeError, match='interrupted'):
        manager.save(FakeModel(1, fail=True), FakeTokenizer(), None, 1)
    assert manager.latest_path() == first
    save(manager, 1)
    assert manager.latest_metadata()['completed_updates'] == 1
    assert not (manager.directory / 'step_000001.tmp').exists()


def test_interruption_before_pointer_publication_can_retry_orphan(tmp_path):
    manager = CheckpointManager(tmp_path)
    first = save(manager, 0)
    with patch.object(manager, '_publish', side_effect=RuntimeError('publish failed')):
        with pytest.raises(RuntimeError, match='publish failed'):
            save(manager, 1)
    assert manager.latest_path() == first
    assert (manager.directory / 'step_000001').is_dir()
    save(manager, 1)
    assert manager.latest_metadata()['completed_updates'] == 1


def test_resume_uses_committed_pointer_if_json_update_was_interrupted(tmp_path):
    manager = CheckpointManager(tmp_path)
    save(manager, 0)
    from lulu import training
    original = training.atomic_json

    def interrupted_json(path, value):
        if Path(path) == tmp_path / 'latest.json':
            raise RuntimeError('manifest update failed')
        return original(path, value)

    with patch.object(training, 'atomic_json', side_effect=interrupted_json):
        with pytest.raises(RuntimeError, match='manifest update failed'):
            save(manager, 1)
    assert json.loads((tmp_path / 'latest.json').read_text())['completed_updates'] == 0
    resumed = CheckpointManager(tmp_path)
    assert resumed.latest_metadata()['completed_updates'] == 1
    save(resumed, 2)
    assert names(resumed) == ['step_000000', 'step_000002']


def test_retained_final_and_old_cadence_remain_when_cadence_changes(tmp_path):
    manager = CheckpointManager(tmp_path, save_every=3)
    for step in range(5):
        save(manager, step, final=step == 4)
    manager = CheckpointManager(tmp_path, save_every=10)
    save(manager, 5)
    save(manager, 6)
    assert names(manager) == ['step_000000', 'step_000003', 'step_000004', 'step_000006']
    assert json.loads((manager.directory / 'step_000004' / 'lulu_state.json').read_text())['checkpoint_manager']['final']


def test_cleanup_does_not_touch_foreign_directories_or_symlinks(tmp_path):
    manager = CheckpointManager(tmp_path)
    foreign = manager.directory / 'step_000005'
    foreign.mkdir()
    (foreign / 'lulu_state.json').write_text('{}')
    unrelated = manager.directory / 'external'
    unrelated.mkdir()
    (manager.directory / 'step_000006').symlink_to(unrelated, target_is_directory=True)
    save(manager, 0)
    save(manager, 1)
    save(manager, 2)
    assert foreign.is_dir() and unrelated.is_dir()
    assert (manager.directory / 'step_000006').is_symlink()
    with pytest.raises(ValueError, match='Not a valid'):
        save(manager, 5)
    with pytest.raises(FileExistsError, match='symlink'):
        save(manager, 6)


def test_committed_checkpoint_cannot_be_overwritten(tmp_path):
    manager = CheckpointManager(tmp_path)
    save(manager, 0)
    save(manager, 1)
    with pytest.raises(ValueError, match='must increase'):
        save(manager, 1)
    with pytest.raises(ValueError, match='must increase'):
        save(manager, 0)
    assert manager.latest_metadata()['completed_updates'] == 1


@pytest.mark.parametrize('target', ['step_999999', '../external'])
def test_broken_or_external_latest_is_rejected(tmp_path, target):
    manager = CheckpointManager(tmp_path)
    (tmp_path / 'external').mkdir()
    manager.latest_link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match='Broken|outside'):
        manager.latest_path()


@pytest.mark.parametrize('cadence', [0, -1, 1.5, True])
def test_invalid_save_cadence_rejected(tmp_path, cadence):
    with pytest.raises(ValueError, match='positive integer'):
        CheckpointManager(tmp_path, cadence)


def test_empty_manager_and_inconsistent_update_metadata(tmp_path):
    manager = CheckpointManager(tmp_path)
    assert manager.latest_path() is None
    assert manager.latest_metadata() is None
    with pytest.raises(ValueError, match='disagrees'):
        manager.save(FakeModel(), FakeTokenizer(), None, 1, {'completed_updates': 2})
