"""CPU tests for reusable Lulu evaluation, including a real saved PEFT student."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_lulu.py"
spec = importlib.util.spec_from_file_location("lulu_evaluation_tests", SCRIPT)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "data.parquet").touch()
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"benchmarks": {
            name: {"full": "data.parquet", "probe": "data.parquet", "full_examples": 10,
                   "probe_examples": 4, "scorer": "choice" if name in ("mmlu_pro", "gpqa_diamond") else "math"}
            for name in evaluation.DEFAULT_BENCHMARKS
        }}))

    def args(self, *extra):
        return evaluation.argument_parser().parse_args([
            "--data-manifest", str(self.manifest), "--gpus", "2,5",
            "--output-dir", str(self.root / "out"), *extra])

    def test_default_suite_includes_general_reasoning_and_relative_paths(self):
        plan = evaluation.build_plan(self.args())
        self.assertEqual(plan["models"], [{"name": "base", "model": "Qwen/Qwen3-1.7B",
                                                  "checkpoint_type": "full"}])
        self.assertEqual({b["name"] for b in plan["benchmarks"]}, set(evaluation.DEFAULT_BENCHMARKS))
        self.assertIn("mmlu_pro", [b["name"] for b in plan["benchmarks"]])
        self.assertIn("gpqa_diamond", [b["name"] for b in plan["benchmarks"]])
        self.assertTrue(all(b["path"] == str((self.root / "data.parquet").resolve())
                            for b in plan["benchmarks"]))
        self.assertFalse((self.root / "out").exists())

    def test_default_shared_framework_and_output_are_workspace_relative(self):
        plan = evaluation.build_plan(self.args())
        framework = SCRIPT.parents[1]
        self.assertEqual(Path(plan["soraka_root"]), framework)
        self.assertEqual(Path(plan["parser_path"]), framework / "scripts" / "benchmark_parser.py")
        self.assertEqual(evaluation.argument_parser().parse_args([]).output_dir,
                         str(evaluation.DEFAULT_OUTPUT_ROOT / "evaluation"))

    def test_shared_framework_override_controls_parser_and_plan(self):
        alternate = self.root / "other-framework"
        (alternate / "scripts").mkdir(parents=True)
        parser = alternate / "scripts" / "benchmark_parser.py"
        parser.write_text("# dry-run fixture\n")
        with patch.dict(os.environ, {"LULU_SORAKA_ROOT": str(alternate)}):
            from_env = evaluation.build_plan(self.args())
            self.assertEqual(from_env["soraka_root"], str(alternate.resolve()))
            self.assertEqual(from_env["parser_path"], str(parser.resolve()))
            original = SCRIPT.parents[1]
            explicit = evaluation.build_plan(self.args("--soraka-root", str(original)))
            self.assertEqual(explicit["soraka_root"], str(original))
            self.assertEqual(explicit["parser_path"], str(original / "scripts/benchmark_parser.py"))

    def test_launcher_preserves_caller_relative_paths_from_arbitrary_directory(self):
        (self.root / "student").mkdir()
        env = dict(os.environ)
        for key in ("DATA_MANIFEST", "EVAL_DATA", "BENCHMARKS", "CHECKPOINT",
                    "FULL_CHECKPOINT", "LORA_CHECKPOINT", "LCB_REPO", "S2T_MATH_PARSER",
                    "S2T_PARSER", "OUTPUT_DIR", "LULU_OUTPUT_ROOT", "LULU_SORAKA_ROOT"):
            env.pop(key, None)
        env.update(PYTHON_BIN=sys.executable, PYTHON="/missing/python", GPUS="cpu",
                   DATA_MANIFEST="manifest.json", CHECKPOINT="student", OUTPUT_DIR="eval-output",
                   DRY_RUN="1", INCLUDE_BASE="0", BATCH_SIZE="16", PYTHONPATH="")
        completed = subprocess.run(["bash", str(SCRIPT.parents[1] / "runs" / "eval_lulu.sh")],
                                   cwd=self.root, env=env, check=True, text=True, capture_output=True)
        plan = json.loads(completed.stdout)
        self.assertEqual(plan["output_dir"], str((self.root / "eval-output").resolve()))
        self.assertEqual(plan["models"], [{"name": "lulu",
                                                  "model": str((self.root / "student").resolve()),
                                                  "checkpoint_type": "auto"}])
        self.assertEqual(plan["batch_size"], 16)
        self.assertTrue(all(b["path"] == str((self.root / "data.parquet").resolve())
                            for b in plan["benchmarks"]))
        self.assertFalse((self.root / "eval-output").exists())

    def test_named_checkpoints_base_control_and_generation_flags(self):
        adapter = self.root / "adapter"
        full = self.root / "full"
        adapter.mkdir(); full.mkdir()
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")
        (full / "config.json").write_text("{}")
        plan = evaluation.build_plan(self.args("--lora-checkpoint", f"ren={adapter}",
                    "--full-checkpoint", f"opd={full}", "--include-base", "--no-thinking",
                    "--max-examples", "3"))
        self.assertEqual([m["name"] for m in plan["models"]], ["base", "opd", "ren"])
        self.assertEqual([m["checkpoint_type"] for m in plan["models"]],
                         ["full", "full", "lora"])
        self.assertFalse(plan["thinking"])
        self.assertTrue(all(b["expected_examples"] == 3 for b in plan["benchmarks"]))

    def test_checkpoint_names_cannot_escape_output_directory_or_collide(self):
        for value in ("../escape=x", "base=x"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluation.build_plan(self.args("--checkpoint", value, "--include-base"))

    def test_explicit_checkpoint_types_reject_mismatched_local_directories(self):
        full = self.root / "full"
        lora = self.root / "lora"
        full.mkdir(); lora.mkdir()
        (full / "config.json").write_text("{}")
        (lora / "adapter_config.json").write_text("{}")
        (lora / "adapter_model.safetensors").write_bytes(b"weights")
        self.assertEqual(evaluation.resolve_checkpoint_type(str(full), "full"), "full")
        self.assertEqual(evaluation.resolve_checkpoint_type(str(lora), "lora"), "lora")
        with self.assertRaisesRegex(ValueError, "declared full-model checkpoint"):
            evaluation.resolve_checkpoint_type(str(lora), "full")
        with self.assertRaisesRegex(ValueError, "declared LoRA checkpoint"):
            evaluation.resolve_checkpoint_type(str(full), "lora")

    def test_direct_benchmark_overrides_manifest_count(self):
        plan = evaluation.build_plan(self.args("--benchmark", f"mmlu_pro={self.root / 'data.parquet'}"))
        self.assertEqual(len(plan["benchmarks"]), 1)
        self.assertIsNone(plan["benchmarks"][0]["expected_examples"])

    def test_selected_missing_benchmark_fails_before_model_load(self):
        with self.assertRaisesRegex(ValueError, "missing from manifest"):
            evaluation.build_plan(self.args("--benchmarks", "missing"))

    def test_device_detection_preserves_scheduler_visibility(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,GPU-example"}):
            self.assertEqual(evaluation.resolve_devices("auto"), ["3", "GPU-example"])
        for value in ("", "-1", "0,0", "cpu,0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluation.resolve_devices(value)

    def test_lcb_requires_official_scorer_and_complete_split(self):
        data = json.loads(self.manifest.read_text())
        data["benchmarks"]["livecodebench"] = {
            "full": "data.parquet", "scorer": "livecodebench",
            "livecodebench": {"release_version": "v6"}}
        self.manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "official evaluator"):
            evaluation.build_plan(self.args("--benchmarks", "all"))
        with self.assertRaisesRegex(ValueError, "complete split"):
            evaluation.build_plan(self.args("--benchmarks", "all", "--lcb-repo", str(self.root), "--max-examples", "1"))


class EvaluationSummaryTests(unittest.TestCase):
    @staticmethod
    def row(index, reward):
        return {"prompt_index": index, "reward": reward, "response_tokens": 10 + index,
                "prompt_tokens": 5, "hit_cap": index == 1}

    def test_scores_and_paired_rescue_degradation_use_all_matching_examples(self):
        base = [self.row(0, 0.0), self.row(1, 1.0), self.row(2, 0.0)]
        candidate = [self.row(0, 1.0), self.row(1, 0.0), self.row(2, 1.0)]
        summary = evaluation.summarize_rows(candidate)
        self.assertAlmostEqual(summary["accuracy"], 2 / 3)
        self.assertEqual(summary["mean_response_tokens"], 11)
        self.assertAlmostEqual(summary["hit_cap_fraction"], 1 / 3)
        compared = evaluation.paired_comparison(base, candidate)
        self.assertEqual(compared["rescues"], 2)
        self.assertEqual(compared["degradations"], 1)
        self.assertAlmostEqual(compared["accuracy_delta"], 1 / 3)
        with self.assertRaisesRegex(ValueError, "indices must match"):
            evaluation.paired_comparison(base, candidate[:-1])

    def test_unscored_rows_are_not_treated_as_incorrect(self):
        summary = evaluation.summarize_rows([self.row(0, None)])
        self.assertEqual(summary["scored_examples"], 0)
        self.assertIsNone(summary["accuracy"])

    def test_bundled_runner_and_parser_are_present_and_choice_parser_scores(self):
        scripts = SCRIPT.parent
        runner = scripts / "evaluate_plain_model.py"
        parser = scripts / "benchmark_parser.py"
        self.assertTrue(runner.is_file())
        self.assertTrue(parser.is_file())
        spec = importlib.util.spec_from_file_location("bundled_benchmark_parser", parser)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(
            module.score_prediction("Answer: C", "__CHOICE__C", scorer="choice")["reward"],
            1.0,
        )

    def test_livecodebench_helpers_use_explicit_shared_framework(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {"soraka_root": str(root / "shared"), "lcb_python": "python",
                    "lcb_repo": str(root / "official"), "lcb_processes": 2}
            benchmark = {"path": str(root / "data.parquet"),
                         "livecodebench": {"release_version": "v6"}}
            with patch.object(evaluation.subprocess, "run") as run:
                scored = evaluation.score_livecodebench(plan, benchmark, root / "raw")
            self.assertEqual(scored, root / "raw/scored")
            self.assertEqual(run.call_args_list[0].args[0][1],
                             str((root / "shared/scripts/export_livecodebench_custom.py").resolve()))
            self.assertEqual(run.call_args_list[1].kwargs["cwd"], str(root / "official"))
            self.assertEqual(run.call_args_list[2].args[0][1],
                             str((root / "shared/scripts/inject_livecodebench_rewards.py").resolve()))

    def test_merge_requires_complete_exact_disjoint_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "shard-000.jsonl").write_text(json.dumps(self.row(0, 1)) + "\n")
            with self.assertRaisesRegex(ValueError, "incomplete or stale"):
                evaluation.load_complete_rows(root, 2, 2)
            (root / "shard-001.jsonl").write_text(json.dumps(self.row(1, 0)) + "\n")
            self.assertEqual(len(evaluation.load_complete_rows(root, 2, 2)), 2)
            with self.assertRaisesRegex(ValueError, "coverage"):
                evaluation.load_complete_rows(root, 2, 3)
            (root / "shard-001.jsonl").write_text(json.dumps(self.row(0, 0)) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate or incorrectly"):
                evaluation.load_complete_rows(root, 2, 2)


@unittest.skipUnless(importlib.util.find_spec("transformers") and importlib.util.find_spec("peft"),
                     "model integration needs the project Python environment")
class SavedStudentIntegrationTests(unittest.TestCase):
    def test_hf_and_lora_checkpoint_loading_and_batched_general_reasoning(self):
        import torch
        import pyarrow as pa
        import pyarrow.parquet as pq
        from peft import LoraConfig, PeftModel, get_peft_model
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            adapter = root / "adapter"
            vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3,
                     "user": 4, "assistant": 5, "question": 6, "long": 7,
                     "<think>": 8, "A": 9, "B": 10, "answer": 11, "is": 12}
            backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<bos>",
                        eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")
            tokenizer.chat_template = ("{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}"
                                       "assistant {% if enable_thinking %}<think>{% endif %}")
            model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=16,
                    intermediate_size=24, num_hidden_layers=1, num_attention_heads=2,
                    num_key_value_heads=2, max_position_embeddings=64,
                    eos_token_id=1, bos_token_id=2, pad_token_id=0))
            model.save_pretrained(base)
            tokenizer.save_pretrained(base)
            trained = get_peft_model(AutoModelForCausalLM.from_pretrained(base),
                                    LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM"))
            trained.save_pretrained(adapter)
            tokenizer.save_pretrained(adapter)
            with self.assertRaisesRegex(ValueError, "declared full-model checkpoint"):
                evaluation.load_model_assets(str(adapter), dtype=torch.float32, device="cpu",
                                             thinking=False, checkpoint_type="full")
            with self.assertRaisesRegex(ValueError, "declared LoRA checkpoint"):
                evaluation.load_model_assets(str(base), dtype=torch.float32, device="cpu",
                                             thinking=False, checkpoint_type="lora")
            loaded, wrapped = evaluation.load_model_assets(
                str(adapter), dtype=torch.float32, device="cpu", thinking=False,
                checkpoint_type="lora")
            self.assertIsInstance(loaded, PeftModel)
            self.assertFalse(loaded.training)
            self.assertNotIn("<think>", wrapped.apply_chat_template(
                [{"role": "user", "content": "question"}], tokenize=False, add_generation_prompt=True))
            self.assertEqual(loaded.generation_config.repetition_penalty, 1.0)
            plain, thinking_tokenizer = evaluation.load_model_assets(
                str(base), dtype=torch.float32, device="cpu", thinking=True,
                checkpoint_type="full")
            self.assertNotIsInstance(plain, PeftModel)
            self.assertIn("<think>", thinking_tokenizer.apply_chat_template(
                [{"role": "user", "content": "question"}], tokenize=False, add_generation_prompt=True))
            data = root / "mmlu.parquet"
            pq.write_table(pa.Table.from_pylist([
                {"prompt": [{"role": "user", "content": text}], "data_source": "mmlu_pro",
                 "reward_model": {"ground_truth": "__CHOICE__A"}}
                for text in ("question", "long long question", "long question")]), data)
            args = evaluation.argument_parser().parse_args([
                "--benchmark", f"mmlu_pro={data}", "--lora-checkpoint", f"ren={adapter}",
                "--gpus", "cpu", "--batch-size", "2", "--max-response-tokens", "2",
                "--max-prompt-tokens", "32", "--dtype", "float32", "--no-thinking",
                "--output-dir", str(root / "out")])
            plan = evaluation.build_plan(args)
            runner = evaluation.make_runner(plan, "cpu")
            try:
                for sid in range(2):
                    runner.evaluate_shard(model_id=str(adapter), input_parquet=str(data),
                        output=str(root / "out" / f"shard-{sid:03d}.jsonl"), shard_id=sid, num_shards=2)
                rows = evaluation.load_complete_rows(root / "out", 2, 3)
                self.assertEqual([r["prompt_index"] for r in rows], [0, 1, 2])
                self.assertEqual({r["data_source"] for r in rows}, {"mmlu_pro"})
                self.assertTrue(all(r["reward"] in (0.0, 1.0) for r in rows))
                self.assertTrue(all(r["response_tokens"] <= 2 for r in rows))
                self.assertGreater(rows[1]["prompt_tokens"], rows[0]["prompt_tokens"])
                self.assertEqual(evaluation.summarize_rows(rows)["scored_examples"], 3)
            finally:
                runner.close()
            # Exercise the real process launcher and final baseline comparison,
            # using a CPU worker and local tiny models only.
            plan["output_dir"] = str(root / "suite")
            plan["models"].insert(0, {"name": "base", "model": str(base)})
            # Start both orchestrator and worker from an unrelated directory,
            # with no inherited project PYTHONPATH and only relative user paths.
            completed = subprocess.run([
                sys.executable, str(SCRIPT), "--benchmark", "mmlu_pro=mmlu.parquet",
                "--lora-checkpoint", "ren=adapter", "--include-base", "--model", "base",
                "--gpus", "cpu", "--batch-size", "2", "--max-response-tokens", "2",
                "--max-prompt-tokens", "32", "--dtype", "float32", "--no-thinking",
                "--output-dir", "suite"], cwd=root, env={**os.environ, "PYTHONPATH": ""},
                text=True, capture_output=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr +
                             "\n" + "\n".join(p.read_text() for p in (root / "suite").glob("worker-*.log")))
            worker_plan = json.loads((root / "suite/eval_plan.json").read_text())
            self.assertEqual(worker_plan["soraka_root"], plan["soraka_root"])
            self.assertEqual(worker_plan["models"][1]["model"], str(adapter))
            report = json.loads((root / "suite/summary.json").read_text())
            self.assertEqual(set(report["models"]), {"base", "ren"})
            metrics = report["models"]["ren"]["benchmarks"]["mmlu_pro"]
            self.assertEqual(metrics["examples"], 3)
            self.assertEqual(metrics["vs_base"]["paired_examples"], 3)
            with self.assertRaisesRegex(ValueError, "must be empty"):
                evaluation.run_suite(plan)


if __name__ == "__main__":
    unittest.main()
