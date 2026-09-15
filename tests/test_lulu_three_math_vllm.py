import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_three_math_vllm.py"
spec = importlib.util.spec_from_file_location("lulu_three_math_vllm", SCRIPT)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
MONITOR = SCRIPT.with_name("monitor_three_math_vllm.py")
monitor_spec = importlib.util.spec_from_file_location("lulu_three_math_monitor", MONITOR)
monitor = importlib.util.module_from_spec(monitor_spec)
monitor_spec.loader.exec_module(monitor)


class ThreeMathVllmTests(unittest.TestCase):
    def test_pass1_pass4_and_rollout_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scored.jsonl"
            values = {0: [False, True, False, False], 1: [True, True, False, False]}
            rows = [
                {"problem_index": problem, "rollout_index": rollout, "is_correct": correct}
                for problem, outcomes in values.items()
                for rollout, correct in enumerate(outcomes)
            ]
            evaluation.write_jsonl(path, rows)
            result = evaluation.summarize(path, 4)
            self.assertEqual(result["problems"], 2)
            self.assertEqual(result["rollouts"], 8)
            self.assertEqual(result["pass@1"], 0.5)
            self.assertEqual(result["pass@4"], 1.0)
            self.assertEqual(result["rollout_accuracy"], 3 / 8)

    def test_missing_or_duplicate_rollout_indices_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scored.jsonl"
            evaluation.write_jsonl(path, [
                {"problem_index": 0, "rollout_index": index, "is_correct": False}
                for index in (0, 1, 1, 3)
            ])
            with self.assertRaisesRegex(ValueError, "rollout indices"):
                evaluation.summarize(path, 4)

    def test_adapter_digest_changes_with_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "adapter_config.json").write_text(json.dumps({"r": 16}))
            weights = root / "adapter_model.safetensors"
            weights.write_bytes(b"first")
            first = evaluation.checkpoint_digest(root)
            weights.write_bytes(b"second")
            self.assertNotEqual(first, evaluation.checkpoint_digest(root))

    def test_progress_snapshot_counts_rollout_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "eval_plan.json").write_text(json.dumps({
                "num_rollouts": 4, "max_examples": 2, "checkpoint_name": "step20"
            }))
            shard = root / "math500/step20/shards/shard_00.jsonl"
            evaluation.write_jsonl(shard, [{"x": 1}, {"x": 2}, {"x": 3}])
            done, rows = monitor.snapshot(root)
            self.assertFalse(done)
            self.assertIn("math500: 3/8 rollouts (37.5%)", rows)
            self.assertIn("overall: 3/24 rollouts (12.5%)", rows)


if __name__ == "__main__":
    unittest.main()
