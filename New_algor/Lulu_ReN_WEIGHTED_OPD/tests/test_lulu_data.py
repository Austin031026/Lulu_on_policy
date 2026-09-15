import json

import pytest

from lulu.data import (
    build_prompt_views, export_eval_parquet, load_prepared_jsonl, normalize_record, prepare_dataset,
    question_key, reasoning_token_mask, split_unique_questions,
)


class CharacterTokenizer:
    """Simple tokenizer with structural tokens and controllable sampled BPE."""
    pieces = {1000: "<think>", 1001: "</think>", 1002: "<eos>", 1003: "ab", 1004: "z\\boxed{"}
    all_special_ids = [1002]

    def encode(self, text, add_special_tokens=False):
        result = []
        while text:
            for token, piece in self.pieces.items():
                if text.startswith(piece):
                    result.append(token)
                    text = text[len(piece):]
                    break
            else:
                result.append(ord(text[0]))
                text = text[1:]
        return result

    def decode(self, ids, **kwargs):
        return "".join(self.pieces.get(i, chr(i)) for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is True
        assert kwargs["add_generation_prompt"] is True
        return "".join(f"<{m['role']}>{m['content']}\n" for m in messages) + "<assistant>\n"


def selected(tokenizer, text, **kwargs):
    ids = tokenizer.encode(text)
    mask = reasoning_token_mask(tokenizer, ids, **kwargs)
    return tokenizer.decode([token for token, take in zip(ids, mask) if take])


def test_dapo_schema_escapes_and_zero_answer():
    row = {
        "data_source": "math_dapo",
        "prompt": [{"role": "user", "content": r"Question\nCompute \\neq and \\frac{1}{2}."}],
        "reward_model": {"ground_truth": "0"},
    }
    item = normalize_record(row)
    assert item["messages"][0]["content"] == "Question\nCompute \\neq and \\frac{1}{2}."
    assert item["gold_answer"] == "0"
    assert item["id"] == question_key(item["messages"])


def test_explicit_columns_and_reject_missing_gold():
    item = normalize_record({"query": "Q?", "label": {"value": 0}}, question_column="query", answer_column="label.value")
    assert item["gold_answer"] == "0"
    for answer in (None, "", "nan", [], {"answer": "2"}):
        with pytest.raises(ValueError):
            normalize_record({"question": "Q?", "answer": answer})
    with pytest.raises(ValueError, match="no answer messages"):
        normalize_record({"messages": [{"role": "assistant", "content": "Gold solution"}], "answer": "2"})


def test_split_deduplicates_before_partition_and_is_order_independent():
    rows = [{"question": f"Question {i}", "answer": str(i)} for i in range(12)]
    rows.append({"question": " Question   1 ", "answer": "1"})
    train, dev, stats = split_unique_questions(rows * 3, dev_size=3, seed=7)
    train_reversed, dev_reversed, _ = split_unique_questions(reversed(rows * 3), dev_size=3, seed=7)
    assert stats["unique_questions"] == 12
    assert stats["duplicate_rows_removed"] == 27
    assert len(train) == 9 and len(dev) == 3
    assert {r["id"] for r in train}.isdisjoint(r["id"] for r in dev)
    assert [r["id"] for r in train] == [r["id"] for r in train_reversed]
    assert [r["id"] for r in dev] == [r["id"] for r in dev_reversed]
    with pytest.raises(ValueError, match="Conflicting gold"):
        split_unique_questions(rows + [{"question": "Question 1", "answer": "wrong"}], dev_size=3, gold_conflict_policy="error")


def test_local_preparation_manifest_and_round_trip(tmp_path):
    pytest.importorskip("pyarrow")
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps({"question": f"Q{i}", "answer": i}) + "\n" for i in range(8)))
    manifest = prepare_dataset(tmp_path / "prepared", dataset=str(source), dev_size=2, train_limit=3)
    assert manifest["counts"] == {"source_rows": 8, "unique_questions": 8, "duplicate_rows_removed": 0, "conflicting_questions_removed": 0, "usable_unique_questions": 8, "train_questions": 3, "dev_questions": 2}
    assert len(load_prepared_jsonl(manifest["outputs"]["train"]["path"])) == 3
    assert len(load_prepared_jsonl(manifest["outputs"]["dev"]["path"])) == 2
    assert len(manifest["outputs"]["train"]["sha256"]) == 64
    assert (tmp_path / "prepared" / "manifest.json").is_file()


def test_gold_is_only_in_hindsight_and_response_ids_align():
    tokenizer = CharacterTokenizer()
    messages = [{"role": "user", "content": "Compute something."}]
    original = json.dumps(messages)
    views = build_prompt_views(tokenizer, messages, "unique-gold-7349")
    assert "unique-gold-7349" not in views["causal_prompt"]
    assert "unique-gold-7349" in views["hindsight_prompt"]
    assert json.dumps(messages) == original
    # Sampled [a,b] differs from encode('ab'), which is a single canonical token.
    response = [1000, ord("a"), ord("b"), 1001]
    for name in ("causal", "hindsight"):
        joined = views[name + "_prompt_ids"] + response
        assert joined[len(views[name + "_prompt_ids"]):] == response
    assert reasoning_token_mask(tokenizer, response) == [False, True, True, False]


def test_reasoning_mask_excludes_answer_and_structural_tokens():
    tokenizer = CharacterTokenizer()
    assert selected(tokenizer, "<think>work</think>Answer: 3<eos>") == "work"
    assert selected(tokenizer, "<think>work\\boxed{3} after</think>") == "work"
    assert selected(tokenizer, "<think>work\nFinal Answer: 3</think>") == "work"
    assert selected(tokenizer, "<think>work\n**Answer:** 3</think>") == "work"
    assert selected(tokenizer, "<think>work. The answer is 3</think>") == "work. "


def test_truncated_and_missing_reasoning_are_safe():
    tokenizer = CharacterTokenizer()
    assert selected(tokenizer, "<think>unfinished work") == "unfinished work"
    assert selected(tokenizer, "answer without explicit thinking") == ""
    assert selected(tokenizer, "answer", prompt_ids=tokenizer.encode("<user>Explain <think> tags.<assistant>")) == ""
    assert selected(tokenizer, "work</think>answer", prompt_ids=tokenizer.encode("<assistant><think>")) == "work"
    assert selected(tokenizer, "<think>x<eos></think>answer") == "x"
    assert reasoning_token_mask(tokenizer, []) == []


def test_mask_excludes_sampled_token_straddling_final_answer_boundary():
    tokenizer = CharacterTokenizer()
    assert selected(tokenizer, "<think>workz\\boxed{3}</think>") == "work"


def test_conflicting_questions_are_removed_entirely():
    rows = [{"question": "Conflicting", "answer": answer} for answer in ("6", "64", "6")]
    rows += [{"question": "Valid", "answer": "1"}]
    train, dev, stats = split_unique_questions(rows, dev_size=0)
    assert len(train) == 1 and train[0]["messages"][0]["content"] == "Valid"
    assert not dev
    assert stats["conflicting_questions_removed"] == 1
    assert stats["unique_questions"] == 2


def test_dapo_fast_path_respects_explicit_gold_precedence():
    rows = [{"prompt": [{"role": "user", "content": "Q"}],
             "reward_model": {"ground_truth": "1"}, "gold_answer": answer}
            for answer in ("1", "2")]
    rows.append({"question": "Valid", "answer": "1"})
    train, _, stats = split_unique_questions(rows, dev_size=0)
    assert len(train) == 1 and train[0]["messages"][0]["content"] == "Valid"
    assert stats["conflicting_questions_removed"] == 1


def test_dev_parquet_matches_shared_evaluator_fields_without_hindsight(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    rows = [normalize_record({"question": "Compute two plus two.", "answer": "GOLD_UNIQUE_4"}),
            normalize_record({"question": "Compute three plus three.", "answer": "GOLD_UNIQUE_6"})]
    metadata = export_eval_parquet(rows, tmp_path / "dev_eval.parquet")
    exported = pq.read_table(metadata["path"]).to_pylist()
    assert len(exported) == metadata["examples"] == 2
    for index, (source, row) in enumerate(zip(rows, exported)):
        assert row["prompt_index"] == index
        assert row["extra_info"]["question_id"] == source["id"]
        assert row["data_source"] == "math"
        # These are exactly the keys consumed by PlainModelEvaluator.
        assert row["prompt"] == source["messages"]
        assert row["reward_model"]["ground_truth"] == source["gold_answer"]
        assert source["gold_answer"] not in json.dumps(row["prompt"])
    empty = export_eval_parquet([], tmp_path / "empty.parquet")
    assert pq.read_table(empty["path"]).schema == pq.read_table(metadata["path"]).schema
