# Copyright 2026 Javier Marín
# SPDX-License-Identifier: Apache-2.0
"""Checks against the stored data. The test sets are rebuilt from FOLIO, which needs network access."""
from pathlib import Path

import pytest

from logicverifier import analysis as A
from logicverifier import data

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def sets():
    return data.load_test_sets(DATA / "test_sets")


def test_test_sets_rebuild_identically(tmp_path):
    try:
        validation = data.read_jsonl(data.download_folio("validation", tmp_path / "folio-validation.jsonl"))
    except OSError:
        pytest.skip("FOLIO cannot be downloaded")
    built, _ = data.build_test_sets(validation)
    for name, stories in built.items():
        assert stories == data.read_jsonl(DATA / "test_sets" / f"{name}.jsonl")


def test_judge_answers(sets):
    gpt = A.load_judge(DATA / "judges" / "gpt-6.1-sol.jsonl")
    labels = A.labels_of(sets)
    summary = A.summarize(A.pooled(gpt), labels)
    assert (summary.stories, summary.false_approvals, summary.approved_follow) == (801, 28, 129)


def test_appendix_b_folio_original(sets):
    """Translator v4 on FOLIO original: 6 false approvals with one reading and 3 with three."""
    stories = sets["folio_original"]
    readings = A.load_readings(DATA / "readings" / "v4" / "folio_original.json")
    rules = [A.Rule("every", 1), A.Rule("every", 3)]
    outcomes = A.evaluate(stories, readings, rules, processes=2)
    labels = {i: s["label"] for i, s in stories.items()}
    one, three = (A.summarize(outcomes[r], labels) for r in rules)
    assert (one.false_approvals, round(one.accuracy, 3), one.retained) == (6, 0.793, 83)
    assert (three.false_approvals, round(three.accuracy, 3), three.retained) == (3, 0.749, 96)
    assert not any(o.retained and o.decision == "True" and labels[i] != "True"
                   for r in rules for i, o in outcomes[r].items())
