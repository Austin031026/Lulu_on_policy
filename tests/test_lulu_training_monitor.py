import json
from pathlib import Path
import tempfile
import unittest

from scripts.monitor_lulu_training import duration, snapshot


class TrainingMonitorTests(unittest.TestCase):
    def test_progress_and_recent_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metrics").mkdir()
            (root / "run_config.json").write_text(json.dumps({
                "rounds": 5, "lora_rank": 0, "kl_direction": "forward",
                "pointwise_kl_clip": 0.05, "kl_diagnostics": True,
            }))
            (root / "latest.json").write_text(json.dumps({
                "completed_updates": 2, "checkpoint": str(root / "checkpoints/step_000002")
            }))
            for index, seconds in enumerate((100, 140)):
                (root / "metrics" / f"round_{index:04d}.json").write_text(json.dumps([{
                    "completed_updates": index + 1, "round_seconds": seconds,
                    "forward_kl": 0.1 / (index + 1), "grad_norm": 0.2,
                }]))
            report, complete = snapshot(root, recent_window=2, bar_width=10)
            self.assertFalse(complete)
            self.assertIn("40.00%", report)
            self.assertIn("全部已完成轮平均：2m 00s", report)
            self.assertIn("预计剩余：6m 00s", report)
            self.assertIn("训练模式：full_parameter", report)
            self.assertIn("pointwise clip=0.05", report)

    def test_duration(self):
        self.assertEqual(duration(3661), "1h 01m 01s")
        self.assertEqual(duration(None), "n/a")


if __name__ == "__main__":
    unittest.main()
