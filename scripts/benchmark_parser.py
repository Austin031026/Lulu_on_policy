#!/usr/bin/env python3
"""Self-contained Lulu math and multiple-choice benchmark scoring."""
from __future__ import annotations

import json
import re

_MATH = None


def _math_parser():
    global _MATH
    if _MATH is None:
        from math_verify import parse, verify
        from math_verify.parser import (
            ExprExtractionConfig,
            LatexExtractionConfig,
            StringExtractionConfig,
        )
        configs = [LatexExtractionConfig(), ExprExtractionConfig(), StringExtractionConfig()]
        _MATH = (parse, verify, configs)
    return _MATH


def _choice(text: str):
    matches = re.findall(
        r"(?i)(?:answer|choice)\s*(?:is|:)?\s*\(?([A-J])\)?|\\boxed\{\s*([A-J])\s*\}",
        text,
    )
    for pair in reversed(matches):
        value = pair[0] or pair[1]
        if value:
            return value.upper()
    standalone = re.findall(r"(?i)(?:^|\s)\(?([A-J])\)?(?:[.\s]|$)", text)
    return standalone[-1].upper() if standalone else None


def score_prediction(prediction: str, reference: str, *, scorer="math", data_source=""):
    if scorer == "choice" or reference.startswith("__CHOICE__"):
        gold = reference.removeprefix("__CHOICE__").strip().upper()
        parsed = _choice(prediction)
        correct = parsed == gold
        return {"reward": float(correct), "parsed_prediction": parsed,
                "matched_gold": gold if correct else None, "error": None}

    references = [reference]
    if reference.lstrip().startswith("["):
        try:
            decoded = json.loads(reference)
            if isinstance(decoded, list) and decoded:
                references = [str(item) for item in decoded]
        except json.JSONDecodeError:
            pass
    parse, verify, configs = _math_parser()
    try:
        parsed_prediction = parse(prediction, extraction_config=configs)
        for gold_text in references:
            parsed_gold = parse(gold_text, extraction_config=configs)
            correct = bool(parsed_prediction and parsed_gold and verify(parsed_gold, parsed_prediction))
            if correct:
                return {"reward": 1.0, "parsed_prediction": repr(parsed_prediction)[:2000],
                        "matched_gold": gold_text, "error": None}
        return {"reward": 0.0, "parsed_prediction": repr(parsed_prediction)[:2000],
                "matched_gold": None, "error": None}
    except Exception as exc:
        return {"reward": 0.0, "parsed_prediction": None,
                "matched_gold": None, "error": repr(exc)}


def extract_answer(text, data_name="", use_last_number=True):
    return text


def math_equal(prediction, reference, **kwargs):
    return score_prediction(str(prediction), str(reference))["reward"] == 1.0

