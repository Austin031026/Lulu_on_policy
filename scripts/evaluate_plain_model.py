#!/usr/bin/env python3
"""Self-contained batched Transformers evaluator for Lulu checkpoints."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


def load_parser(path: str | Path):
    location = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location("lulu_benchmark_parser", location)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import benchmark parser: {location}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlainModelRunner:
    """Generate and score one response per benchmark prompt."""

    def __init__(self, *, parser_path, batch_size=8, max_response_tokens=8192,
                 max_prompt_tokens=4096, dtype="bfloat16"):
        self.parser_path = str(parser_path)
        self.parser = load_parser(parser_path)
        self.batch_size = int(batch_size)
        self.max_response_tokens = int(max_response_tokens)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.dtype = dtype
        self.model_id = None
        self.model = None
        self.tokenizer = None

    def load_model(self, model_id, **kwargs):  # pragma: no cover - overridden by LuluRunner
        raise NotImplementedError

    def close(self):
        model, self.model = self.model, None
        self.tokenizer = None
        self.model_id = None
        if model is not None:
            del model
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _messages(row: dict[str, Any]):
        prompt = row.get("prompt")
        if isinstance(prompt, str):
            return prompt
        if not isinstance(prompt, list):
            raise ValueError("benchmark row must contain prompt as text or chat messages")
        return [{"role": str(item["role"]), "content": str(item["content"])}
                for item in prompt]

    def _prompt_text(self, row):
        prompt = self._messages(row)
        if isinstance(prompt, str):
            return prompt
        return self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

    def _score(self, text, row, scorer):
        reward_model = row.get("reward_model") or {}
        reference = reward_model.get("ground_truth")
        if reference is None:
            return 0.0, {"error": "missing reward_model.ground_truth"}
        result = self.parser.score_prediction(
            text, str(reference), scorer=scorer,
            data_source=str(row.get("data_source") or ""),
        )
        return float(result["reward"]), result

    def _trim_generation(self, token_ids):
        eos = self.tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple, set)) else [eos])
        eos_ids.discard(None)
        for index, token_id in enumerate(token_ids):
            if token_id in eos_ids:
                return token_ids[:index + 1], True
        return token_ids, False

    def evaluate_shard(self, *, model_id, input_parquet, output, shard_id,
                       num_shards, max_examples=0, store_text=False,
                       progress_every=8, scorer="math"):
        import pyarrow.parquet as pq
        import torch

        self.load_model(model_id)
        all_rows = pq.read_table(input_parquet).to_pylist()
        if max_examples:
            all_rows = all_rows[:int(max_examples)]
        rows = [row for row in all_rows
                if int(row["prompt_index"]) % int(num_shards) == int(shard_id)]
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        completed = 0
        with destination.open("w", encoding="utf-8") as handle:
            for start in range(0, len(rows), self.batch_size):
                batch_rows = rows[start:start + self.batch_size]
                prompt_texts = [self._prompt_text(row) for row in batch_rows]
                encoded = self.tokenizer(
                    prompt_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.max_prompt_tokens,
                )
                model_device = next(self.model.parameters()).device
                encoded = {key: value.to(model_device) for key, value in encoded.items()}
                input_width = int(encoded["input_ids"].shape[1])
                with torch.inference_mode():
                    generated = self.model.generate(
                        **encoded, max_new_tokens=self.max_response_tokens,
                        do_sample=False, num_beams=1,
                    )
                for offset, row in enumerate(batch_rows):
                    raw_ids = generated[offset, input_width:].detach().cpu().tolist()
                    token_ids, stopped = self._trim_generation(raw_ids)
                    response = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                    reward, details = self._score(response, row, scorer)
                    result = {
                        "prompt_index": int(row["prompt_index"]),
                        "id": str(row.get("id", row["prompt_index"])),
                        "data_source": row.get("data_source"),
                        "reward": reward,
                        "prompt_tokens": int(encoded["attention_mask"][offset].sum().item()),
                        "response_tokens": len(token_ids),
                        "hit_cap": not stopped and len(raw_ids) >= self.max_response_tokens,
                        "score_details": details,
                    }
                    if store_text:
                        result["response"] = response
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                completed += len(batch_rows)
                if completed == len(rows) or completed % max(1, int(progress_every)) == 0:
                    percent = 100.0 * completed / len(rows) if rows else 100.0
                    print(f"[lulu-eval][PROGRESS] shard={shard_id} "
                          f"completed={completed}/{len(rows)} percent={percent:.1f}%",
                          flush=True)
        print(f"[lulu-eval][DONE] shard={shard_id} rows={len(rows)} output={destination}",
              flush=True)

