"""Atomic latest checkpoints with bounded archival retention.

Every completed update writes a recoverable ``latest`` checkpoint. ``save_every``
controls how many of those checkpoints are retained as history, not how often
weights or optimizer state are written. Only directories marked by this manager
are eligible for removal.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import uuid


_STEP_NAME = re.compile(r"step_(\d{6,})\Z")
_OWNER = "lulu.persistent.v1"


class CheckpointManager:
    """Save into ``output_root/checkpoints`` and publish an atomic latest link.

    The symlink is the commit record; ``latest.json`` is a convenient manifest.
    Readers use the checkpoint's own metadata, so an interruption between the
    symlink and JSON updates cannot send a resumed run to an older checkpoint.
    Calls must be serialized (normally only Student rank zero calls ``save``).
    """

    def __init__(self, output_root, save_every: int = 20):
        if isinstance(save_every, bool) or not isinstance(save_every, int) or save_every < 1:
            raise ValueError("save_every must be a positive integer")
        self.root = Path(output_root).expanduser().resolve()
        self.directory = self.root / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.latest_link = self.directory / "latest"

    def _state(self, path: Path) -> dict:
        with (path / "lulu_state.json").open(encoding="utf-8") as stream:
            state = json.load(stream)
        match = _STEP_NAME.fullmatch(path.name)
        marker = state.get("checkpoint_manager", {})
        if (not match or marker.get("owner") != _OWNER
                or state.get("completed_updates") != int(match.group(1))):
            raise ValueError(f"Not a valid LuLu managed checkpoint: {path}")
        return state

    def latest_path(self) -> Path | None:
        """Return the committed checkpoint, rejecting broken or foreign links."""
        if not self.latest_link.is_symlink():
            if self.latest_link.exists():
                raise ValueError(f"Expected a managed symlink: {self.latest_link}")
            return None
        try:
            path = self.latest_link.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Broken latest checkpoint link: {self.latest_link}") from exc
        if path.parent != self.directory or not path.is_dir():
            raise ValueError(f"Latest checkpoint points outside its directory: {path}")
        self._state(path)
        return path

    def latest_metadata(self) -> dict | None:
        """Read authoritative state from the committed checkpoint itself."""
        path = self.latest_path()
        if path is None:
            return None
        return {**self._state(path), "checkpoint": str(path)}

    def _publish(self, path: Path, state: dict) -> None:
        from .training import atomic_json

        temporary_link = self.latest_link.with_name(f".latest.{uuid.uuid4().hex}.tmp")
        try:
            temporary_link.symlink_to(path.name, target_is_directory=True)
            os.replace(temporary_link, self.latest_link)
        finally:
            temporary_link.unlink(missing_ok=True)
        atomic_json(self.root / "latest.json", {**state, "checkpoint": str(path)})

    def _prune(self, current: Path) -> None:
        for path in self.directory.iterdir():
            if path == current or path.is_symlink() or not path.is_dir():
                continue
            if not _STEP_NAME.fullmatch(path.name):
                continue
            try:
                state = self._state(path)
            except (OSError, ValueError, TypeError, AttributeError):
                # Existing user checkpoints and unrelated directories are untouched.
                continue
            if not state["checkpoint_manager"].get("retained", False):
                shutil.rmtree(path)

    def save(self, model, tokenizer, optimizer, step: int,
             metadata: dict | None = None, *, final: bool = False) -> Path:
        """Write latest and retain initial, periodic, and explicit final snapshots.

        A failed write leaves the previously committed latest untouched. A complete
        orphan from an interrupted publication can be replaced on a later retry;
        an already committed update can never be overwritten.
        """
        from .training import save_checkpoint

        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        state = dict(metadata or {})
        if "completed_updates" in state and state["completed_updates"] != step:
            raise ValueError("metadata completed_updates disagrees with step")
        previous = self.latest_metadata()
        if previous is not None and step <= previous["completed_updates"]:
            raise ValueError("Checkpoint updates must increase; committed updates cannot be overwritten")
        state["completed_updates"] = step
        state["checkpoint_manager"] = {
            "owner": _OWNER,
            "save_every": self.save_every,
            "retained": step == 0 or step % self.save_every == 0 or bool(final),
            "final": bool(final),
        }
        path = self.directory / f"step_{step:06d}"
        if path.is_symlink():
            raise FileExistsError(f"Refusing to replace checkpoint symlink: {path}")
        if path.exists():
            # Since step exceeds the committed update, a correctly marked existing
            # directory is an unpublished orphan, not a checkpoint a reader uses.
            self._state(path)
            shutil.rmtree(path)
        save_checkpoint(model, tokenizer, path, optimizer=optimizer, metadata=state)
        self._publish(path, state)
        self._prune(path)
        return path
