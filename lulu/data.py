"""Data preparation and leak-free prompt/response alignment for ReN-OPD.

This module has no heavyweight imports at import time. In particular,
reading local JSONL and constructing masks do not need torch, datasets, or a GPU.
The common evaluator Parquet export requires pyarrow.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_DATASET = "BytedTsinghua-SIA/DAPO-Math-17k"
DEFAULT_CACHE_DIR = "/pfss/mlde/workspaces/mlde_wsp_Eco_Inference/huggingface_cache/datasets"
HINDSIGHT_CONTEXT = (
    "For this reasoning task, the verified final answer is provided below as "
    "additional context. Continue solving the original problem step by step.\n"
    "<verified_final_answer>\n{answer}\n</verified_final_answer>"
)


def _column(row: Mapping[str, Any], name: str) -> Any:
    # A literal column name wins over interpreting a dotted nested path.
    if name in row:
        return row[name]
    value: Any = row
    for key in name.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"Missing required column {name!r}")
        value = value[key]
    return value


def _decode_dapo_prompt(text: str) -> str:
    """Undo DAPO's extra escape layer without treating LaTeX \\neq as a newline."""
    # Only singly escaped newlines are line breaks; doubled backslashes belong
    # to LaTeX. Do the newline pass before removing the extra LaTeX slash.
    text = re.sub(r"(?<!\\)\\[nr]", "\n", text)
    return text.replace("\\\\", "\\")


def _messages(value: Any, *, decode_dapo_escapes: bool = False) -> list[dict[str, str]]:
    if isinstance(value, str):
        value = [{"role": "user", "content": value}]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Question must be a nonempty string or a chat-message list")
    result = []
    for message in value:
        if not isinstance(message, Mapping):
            raise ValueError("Every prompt message must contain role and content")
        role, content = message.get("role"), message.get("content")
        if role not in {"system", "user"}:
            raise ValueError("Lulu question prompts accept only system/user messages; no answer messages")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Prompt message content must be nonempty text")
        if decode_dapo_escapes:
            content = _decode_dapo_prompt(content)
        result.append({"role": role, "content": content.strip()})
    if result[-1]["role"] != "user":
        raise ValueError("Question prompt must end with a user message")
    return result


def question_key(messages: Sequence[Mapping[str, str]]) -> str:
    """Group identical user questions despite whitespace/system-prompt differences."""
    content = "\n".join(m["content"] for m in messages if m["role"] == "user")
    canonical = " ".join(unicodedata.normalize("NFC", content).split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_record(
    row: Mapping[str, Any], *, question_column: str | None = None,
    answer_column: str | None = None, decode_dapo_escapes: bool = False,
) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError("Each dataset row must be an object")
    if question_column is None:
        question_column = next((k for k in ("messages", "prompt", "question", "problem") if k in row), None)
    if question_column is None:
        raise ValueError("Cannot infer question column; provide --question-column")
    if answer_column is None:
        answer_column = next((k for k in ("gold_answer", "answer", "final_answer") if k in row), None)
        if answer_column is None and isinstance(row.get("reward_model"), Mapping):
            answer_column = "reward_model.ground_truth"
    if answer_column is None:
        raise ValueError("Cannot infer gold-answer column; provide --answer-column")
    messages = _messages(
        _column(row, question_column),
        decode_dapo_escapes=decode_dapo_escapes or row.get("data_source") == "math_dapo",
    )
    answer = _column(row, answer_column)
    if isinstance(answer, (list, tuple)) and len(answer) == 1:
        answer = answer[0]
    if answer is None or isinstance(answer, (dict, list, tuple, bool)):
        raise ValueError("Gold answer must be a nonempty scalar, not a solution/message structure")
    if isinstance(answer, float) and not math.isfinite(answer):
        raise ValueError("Gold answer must be finite")
    answer = str(answer).strip()
    if not answer or answer.lower() in {"null", "none", "nan"}:
        raise ValueError("Gold answer is missing or empty")
    key = question_key(messages)
    return {"id": key, "messages": messages, "gold_answer": answer}


def split_unique_questions(
    rows: Iterable[Mapping[str, Any]], *, dev_size: int | float = 256, seed: int = 42,
    train_limit: int | None = None, question_column: str | None = None,
    answer_column: str | None = None, decode_dapo_escapes: bool = False,
    gold_conflict_policy: str = "drop",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Deduplicate before splitting; drop contradictory groups and reject goldless rows.

    Hash ordering makes the split independent of source row order and repeats.
    The published DAPO train split repeats its 17,917 questions 100 times.
    """
    if gold_conflict_policy not in {"drop", "error"}:
        raise ValueError("gold_conflict_policy must be drop or error")
    unique: dict[str, dict[str, Any]] = {}
    conflicting: set[str] = set()
    count = 0
    # DAPO's repeats are adjacent in full-dataset copies. Memoize normalization
    # for repeated raw questions, avoiding 1.8M repeated regex/hash operations.
    raw_cache: dict[str, tuple[str, str]] = {}
    if callable(getattr(rows, "iter", None)):
        # Hugging Face batch formatting amortizes Arrow-to-Python conversion;
        # keep only one small batch resident in addition to the unique map.
        def batched_records():
            for batch in rows.iter(batch_size=4096):
                columns = list(batch)
                for values in zip(*(batch[column] for column in columns)):
                    yield dict(zip(columns, values))
        records = batched_records()
    else:
        records = rows
    for count, row in enumerate(records, start=1):
        try:
            raw_key = None
            if (question_column is None and answer_column is None
                    and "messages" not in row and "prompt" in row and "reward_model" in row
                    and not any(key in row for key in ("gold_answer", "answer", "final_answer"))):
                raw_key = json.dumps(row["prompt"], ensure_ascii=False, sort_keys=True)
                raw_answer = _column(row, answer_column or "reward_model.ground_truth")
                cached = raw_cache.get(raw_key)
                if cached is not None and raw_answer == cached[1]:
                    continue
            item = normalize_record(
                row, question_column=question_column, answer_column=answer_column,
                decode_dapo_escapes=decode_dapo_escapes,
            )
            previous = unique.get(item["id"])
            if previous is not None and previous["gold_answer"] != item["gold_answer"]:
                if gold_conflict_policy == "error":
                    raise ValueError(f"Conflicting gold answers for question {item['id'][:16]}")
                conflicting.add(item["id"])
                del unique[item["id"]]
            if previous is None and item["id"] not in conflicting:
                unique[item["id"]] = item
            if raw_key is not None:
                raw_cache[raw_key] = (item["id"], item["gold_answer"])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid source row {count - 1}: {exc}") from exc
    ordered = sorted(unique.values(), key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).digest())
    if isinstance(dev_size, float):
        if not 0 <= dev_size < 1:
            raise ValueError("Fractional dev_size must be in [0, 1)")
        n_dev = int(len(ordered) * dev_size)
    else:
        n_dev = int(dev_size)
    if n_dev < 0 or n_dev >= len(ordered):
        raise ValueError(f"dev_size={dev_size} leaves no training questions among {len(ordered)} unique questions")
    if train_limit is not None and train_limit <= 0:
        raise ValueError("train_limit must be positive")
    dev, train = ordered[:n_dev], ordered[n_dev:]
    if train_limit is not None:
        train = train[:train_limit]
    return train, dev, {"source_rows": count, "unique_questions": len(ordered) + len(conflicting),
                        "duplicate_rows_removed": count - len(ordered) - len(conflicting),
                        "conflicting_questions_removed": len(conflicting),
                        "usable_unique_questions": len(ordered),
                        "train_questions": len(train), "dev_questions": len(dev)}


def load_source_dataset(
    dataset: str = DEFAULT_DATASET, *, dataset_config: str | None = None,
    split: str = "train", cache_dir: str | None = DEFAULT_CACHE_DIR,
    offline: bool = True,
) -> Iterable[Mapping[str, Any]]:
    path = Path(dataset).expanduser()
    if path.is_file() and path.suffix.lower() in {".jsonl", ".json"}:
        if path.suffix.lower() == ".jsonl":
            def iter_jsonl():
                with path.open(encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if line.strip():
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError as exc:
                                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            return iter_jsonl()
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get(split, value.get("data"))
        if not isinstance(value, list):
            raise ValueError("Local JSON must contain a list, a data list, or a named split list")
        return value
    import datasets
    from datasets import DownloadConfig, load_dataset, load_from_disk
    if path.is_dir() and ((path / "dataset_info.json").exists() or (path / "dataset_dict.json").exists()):
        result = load_from_disk(str(path))
        return result[split] if isinstance(result, datasets.DatasetDict) else result
    if path.is_file():
        if path.suffix.lower() not in {".parquet", ".csv", ".arrow"}:
            raise ValueError(f"Unsupported local data extension: {path.suffix}")
        if path.suffix.lower() == ".arrow":
            return datasets.Dataset.from_file(str(path))
        return load_dataset(path.suffix[1:].lower(), data_files=str(path), split="train", cache_dir=cache_dir)
    old_offline = datasets.config.HF_DATASETS_OFFLINE
    try:
        datasets.config.HF_DATASETS_OFFLINE = offline
        return load_dataset(dataset, name=dataset_config, split=split, cache_dir=cache_dir,
                            download_config=DownloadConfig(local_files_only=offline))
    finally:
        datasets.config.HF_DATASETS_OFFLINE = old_offline


def export_eval_parquet(
    rows: Sequence[Mapping[str, Any]], path: str | Path, *, data_source: str = "math",
) -> dict[str, Any]:
    """Export held-out causal prompts in the shared plain-model evaluator schema."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("id", pa.string()), ("prompt_index", pa.int64()),
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
        ("extra_info", pa.struct([("index", pa.int64()), ("question_id", pa.string()),
                                  ("dataset_name", pa.string())])),
    ])
    values = []
    for index, row in enumerate(rows):
        item = normalize_record(row, question_column="messages", answer_column="gold_answer")
        identity = str(row.get("id", item["id"]))
        values.append({
            "id": identity, "prompt_index": index, "data_source": data_source,
            "prompt": item["messages"],
            "reward_model": {"ground_truth": item["gold_answer"], "style": "rule"},
            "extra_info": {"index": index, "question_id": identity, "dataset_name": "lulu_dev"},
        })
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(values, schema=schema), destination)
    return {"path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "examples": len(values), "data_source": data_source}


def prepare_dataset(
    output_dir: str | Path, *, dataset: str = DEFAULT_DATASET,
    dataset_config: str | None = None, split: str = "train",
    cache_dir: str | None = DEFAULT_CACHE_DIR, offline: bool = True,
    question_column: str | None = None, answer_column: str | None = None,
    dev_size: int | float = 256, train_limit: int | None = None, seed: int = 42,
    gold_conflict_policy: str = "drop", eval_data_source: str = "math",
) -> dict[str, Any]:
    source = load_source_dataset(dataset, dataset_config=dataset_config, split=split,
                                 cache_dir=cache_dir, offline=offline)
    train, dev, counts = split_unique_questions(
        source, dev_size=dev_size, seed=seed, train_limit=train_limit,
        question_column=question_column, answer_column=answer_column,
        decode_dapo_escapes="dapo" in dataset.lower(),
        gold_conflict_policy=gold_conflict_policy,
    )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, rows in (("train", train), ("dev", dev)):
        path = destination / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        outputs[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    outputs["dev_eval"] = export_eval_parquet(dev, destination / "dev_eval.parquet", data_source=eval_data_source)
    manifest = {
        "schema_version": 1, "dataset": dataset, "dataset_config": dataset_config,
        "source_split": split,
        "source_fingerprint": getattr(source, "_fingerprint", None),
        "source_cache_files": getattr(source, "cache_files", []),
        "question_column": question_column,
        "answer_column": answer_column, "eval_data_source": eval_data_source,
        "seed": seed, "dev_size": dev_size,
        "train_limit": train_limit, "split_policy": "deduplicate normalized user questions; seeded hash order",
        "goldless_policy": "reject", "gold_conflict_policy": gold_conflict_policy, "counts": counts,
        "outputs": outputs,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def load_prepared_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            item = normalize_record(raw, question_column="messages", answer_column="gold_answer")
            if item["id"] in seen:
                raise ValueError(f"Duplicate question in prepared data at {path}:{line_number}")
            seen.add(item["id"])
            # Prepared/user-supplied IDs are traceability labels; split safety
            # always relies on normalized question content, never these labels.
            item["id"] = str(raw.get("id", item["id"]))
            rows.append(item)
    return rows


def build_prompt_views(
    tokenizer: Any, messages: Sequence[Mapping[str, str]], gold_answer: str,
    *, enable_thinking: bool = True,
) -> dict[str, Any]:
    """Only the hindsight user context receives gold; neither causal nor teacher does."""
    causal = _messages(messages)
    if gold_answer is None or not str(gold_answer).strip():
        raise ValueError("A nonempty gold answer is required for hindsight")
    hindsight = [dict(message) for message in causal]
    # Extend the final user turn, preserving templates that disallow consecutive
    # user messages. The original question is byte-identical in both views.
    hindsight[-1]["content"] += "\n\n" + HINDSIGHT_CONTEXT.format(answer=str(gold_answer).strip())
    result: dict[str, Any] = {}
    for name, view in (("causal", causal), ("hindsight", hindsight)):
        text = tokenizer.apply_chat_template(view, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=enable_thinking)
        ids = tokenizer.encode(text, add_special_tokens=False)
        result[f"{name}_prompt"] = text
        result[f"{name}_prompt_ids"] = list(ids)
    return result


_ANSWER_START = re.compile(
    r"\\(?:boxed|fbox)\s*\{|(?:^|\n)\s*(?:\*\*)?(?:final\s+answer|answer)\s*(?:\*\*)?\s*[:：]"
    r"|\b(?:the\s+)?(?:final\s+)?answer\s*(?:is\b|[:：])",
    re.IGNORECASE,
)


def reasoning_token_mask(
    tokenizer: Any, response_ids: Sequence[int], *, prompt_ids: Sequence[int] | None = None,
) -> list[bool]:
    """Return a response-only ReN mask, conservatively excluding answer leakage.

    Only explicit thinking spans qualify. An open <think> span is valid when
    rollout is truncated. With no thinking marker/prefill, all positions are
    excluded. The first boxed/final-answer marker ends eligibility, even inside
    <think>. Exact sampled IDs are never re-tokenized: generated BPE sequences
    need not equal the canonical tokenization of their decoded text.
    """
    ids = [int(token) for token in response_ids]
    if not ids:
        return []
    decode = lambda value: tokenizer.decode(value, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    text = decode(ids)
    prompt = decode(list(prompt_ids)) if prompt_ids is not None else ""
    # Only an assistant-generation prefill at the end counts; a user question
    # mentioning <think> must not turn a plain answer into eligible reasoning.
    prefilled = re.search(r"<think>\s*$", prompt) is not None
    opening = text.find("<think>")
    if prefilled:
        start = 0
    elif opening >= 0:
        start = opening + len("<think>")
    else:
        return [False] * len(ids)
    closing = text.find("</think>", start)
    end = closing if closing >= 0 else len(text)
    answer = _ANSWER_START.search(text, start, end)
    if answer is not None:
        end = answer.start()

    lengths: dict[int, int] = {0: 0, len(ids): len(text)}
    def boundary(position: int, *, strict: bool) -> int:
        # Prefix decoding is O(log N) calls per boundary and preserves sampled
        # token segmentation, including tokens that straddle an answer marker.
        low, high = 0, len(ids) + 1
        while low < high:
            middle = (low + high) // 2
            if middle == len(ids) + 1:
                length = len(text) + 1
            else:
                if middle not in lengths:
                    lengths[middle] = len(decode(ids[:middle]))
                length = lengths[middle]
            reached = length > position if strict else length >= position
            if reached:
                high = middle
            else:
                low = middle + 1
        return low
    first = boundary(start, strict=False)
    stop = min(len(ids), boundary(end, strict=True) - 1)
    excluded = set(getattr(tokenizer, "all_special_ids", []))
    for marker in ("<think>", "</think>"):
        marker_ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(marker_ids) == 1:
            excluded.add(int(marker_ids[0]))
    return [first <= index < stop and token not in excluded for index, token in enumerate(ids)]
