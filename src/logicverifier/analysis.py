# Copyright 2026 Javier Marín
# SPDX-License-Identifier: Apache-2.0
"""Decision rules, measures, statistical tests and the estimate of the rate of false approvals.

Everything here runs on stored readings and judge answers, on a CPU.

Measures, as defined in the paper. A claim *follows* when its label is True. *False
approvals* are the share of claims that do not follow that are approved (PASS),
*coverage* the share of claims that follow that are approved, and *referred* the share of
all stories sent to review. A story is *retained* when, for the claim and every sentence
with a non empty reading set, some reading is equivalent to the expert formula.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

from scipy.stats import beta, binomtest

from .data import read_jsonl
from .logic import contains_equivalent, decide_reading_sets, directed_edits, equivalent, is_valid_reading
from .prompts import parse_judge

__all__ = [
    "Rule",
    "Outcome",
    "Summary",
    "load_readings",
    "load_judge",
    "evaluate",
    "summarize",
    "mcnemar",
    "bootstrap_difference",
    "clopper_pearson_upper",
    "estimate_bound",
    "validate_estimate",
    "outer_bound",
    "labels_of",
    "pooled",
    "plot_main_figure",
    "plot_summary_figure",
]

PASS, FAIL, REVIEW = "True", "False", "Uncertain"


class Rule(NamedTuple):
    """How the reading sets of a story are formed from its first ``readings`` readings.

    ``"every"`` keeps every valid reading (quantification), ``"most_frequent"`` keeps
    one reading of the largest class of equivalent readings (selection). ``edits`` adds
    the valid neighbours of each kept reading (see :func:`logic.directed_edits`).
    """

    kind: str
    readings: int
    edits: str = ""

    def __str__(self) -> str:
        name = {"every": "Every reading", "most_frequent": "Most frequent"}[self.kind]
        text = f"{name} of {self.readings}" if self.readings > 1 else "One reading (greedy)"
        return text + (f", with {self.edits} edits" if self.edits else "")


class Outcome(NamedTuple):
    decision: str           # PASS ("True"), FAIL ("False") or REVIEW ("Uncertain")
    retained: bool | None   # every faithful reading kept (None for judges)
    size: float | None      # mean number of readings kept per sentence


# ----------------------------------------------------------------------------- stored data

def load_readings(path: str | Path) -> dict[str, list]:
    """Readings of one translator on one test set: ``{story id: [reading 0, ..., reading 6]}``."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)["readings"]


def load_judge(path: str | Path) -> dict[str, dict[str, str]]:
    """Verdicts of a hosted judge: ``{set: {story id: "True" | "False" | "Uncertain"}}``."""
    verdicts: dict[str, dict[str, str]] = {}
    for answer in read_jsonl(path):
        verdicts.setdefault(answer["set"], {})[answer["id"]] = parse_judge(answer["text"])
    return verdicts


# ----------------------------------------------------------------------------- decision rules

def _most_frequent(readings: Sequence[str]) -> list[str]:
    """One reading of the largest class of equivalent formulas (selection).

    The distinct valid readings are compared in order of generation, each joins the first
    class whose representative Z3 finds equivalent within three seconds, and the first
    reading of the largest class is kept. A class is as large as the number of distinct
    formulas it holds, and ties go to the earliest class, the one of the greedy reading
    when it is valid.
    """
    classes: list[list[str]] = []
    for reading in dict.fromkeys(readings):
        for members in classes:
            if equivalent(members[0], reading, 3000) is True:
                members.append(reading)
                break
        else:
            classes.append([reading])
    if not classes:
        return []
    largest = max(len(members) for members in classes)
    return [next(members for members in classes if len(members) == largest)[0]]


def _reading_sets(story: dict, readings: Sequence[Sequence[str]], rule: Rule) -> list[list[str]]:
    n = len(story["premises"]) + 1
    vocab = story["vocab"]
    candidates = [[] for _ in range(n)]
    for reading in readings[:rule.readings]:
        for j in range(n):
            formula = reading[j] if reading and j < len(reading) else ""
            if is_valid_reading(formula, vocab):
                candidates[j].append(formula)
    if rule.kind == "most_frequent":
        sets = [_most_frequent(c) for c in candidates]
    else:
        sets = [list(dict.fromkeys(c)) for c in candidates]
    if rule.edits:
        for j in range(n):
            for formula in list(sets[j]):
                for neighbour in directed_edits(formula, rule.edits):
                    if is_valid_reading(neighbour, vocab) and neighbour not in sets[j]:
                        sets[j].append(neighbour)
    return sets


def _evaluate_story(rules: Sequence[Rule], task: tuple[dict, list]) -> dict[Rule, Outcome]:
    story, readings = task
    expert = story["premises-FOL"] + [story["conclusion-FOL"]]
    cache: dict = {}
    outcomes = {}
    for rule in rules:
        sets = _reading_sets(story, readings, rule)
        key = tuple(tuple(s) for s in sets)
        if key not in cache:
            decision = decide_reading_sets(sets[:-1], sets[-1])
            retained = bool(sets[-1]) and all(not s or contains_equivalent(s, g) for s, g in zip(sets, expert))
            cache[key] = (decision, retained)
        decision, retained = cache[key]
        outcomes[rule] = Outcome(decision, retained, sum(len(s) for s in sets) / len(sets))
    return outcomes


def evaluate(stories: dict[str, dict], readings: dict[str, list], rules: Sequence[Rule],
             processes: int | None = None) -> dict[Rule, dict[str, Outcome]]:
    """Decide every story under every rule. Returns ``{rule: {story id: outcome}}``."""
    ids = [i for i in stories if i in readings]
    tasks = [(stories[i], readings[i]) for i in ids]
    with Pool(processes) as pool:
        per_story = pool.map(partial(_evaluate_story, list(rules)), tasks, chunksize=1)
    return {rule: {i: result[rule] for i, result in zip(ids, per_story)} for rule in rules}


# ----------------------------------------------------------------------------- measures

@dataclass(frozen=True)
class Summary:
    stories: int
    follow: int                 # claims that follow (label True)
    not_follow: int             # claims that do not follow (False or Uncertain)
    correct: int                # decision equal to the label
    approved_follow: int        # PASS on a claim that follows
    false_approvals: int        # PASS on a claim that does not follow
    referred: int               # REVIEW
    retained: int | None
    mean_size: float | None

    @property
    def accuracy(self) -> float:
        return self.correct / self.stories

    @property
    def coverage(self) -> float:
        return self.approved_follow / self.follow

    @property
    def false_approval_rate(self) -> float:
        return self.false_approvals / self.not_follow

    @property
    def referred_rate(self) -> float:
        return self.referred / self.stories

    @property
    def retained_rate(self) -> float | None:
        return None if self.retained is None else self.retained / self.stories


def summarize(outcomes: dict[str, Outcome | str], labels: dict[str, str]) -> Summary:
    """Counts over the stories present in both ``outcomes`` and ``labels``. Outcomes may be
    plain decisions (for judges)."""
    ids = [i for i in labels if i in outcomes]
    decisions = {i: o.decision if isinstance(o, Outcome) else o for i, o in outcomes.items()}
    has_readings = all(isinstance(outcomes[i], Outcome) for i in ids)
    return Summary(
        stories=len(ids),
        follow=sum(labels[i] == PASS for i in ids),
        not_follow=sum(labels[i] != PASS for i in ids),
        correct=sum(decisions[i] == labels[i] for i in ids),
        approved_follow=sum(decisions[i] == PASS and labels[i] == PASS for i in ids),
        false_approvals=sum(decisions[i] == PASS and labels[i] != PASS for i in ids),
        referred=sum(decisions[i] == REVIEW for i in ids),
        retained=sum(outcomes[i].retained for i in ids) if has_readings else None,
        mean_size=sum(outcomes[i].size for i in ids) / len(ids) if has_readings else None,
    )


# ----------------------------------------------------------------------------- statistics

def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Two-sided exact McNemar test on paired outcomes (binomial test on the discordant pairs)."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    if only_a + only_b == 0:
        return 1.0
    return binomtest(only_a, only_a + only_b, 0.5).pvalue


def bootstrap_difference(a: Sequence[bool], b: Sequence[bool], resamples: int = 10_000, seed: int = 0,
                         level: float = 0.95) -> tuple[float, float, float]:
    """Difference of two paired rates, ``mean(a) - mean(b)``, with a percentile bootstrap
    interval that resamples the paired items. Returned in percentage points."""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(resamples):
        sample = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a[k] - b[k] for k in sample) / n)
    diffs.sort()
    tail = (1 - level) / 2
    low, high = diffs[int(tail * resamples)], diffs[int((1 - tail) * resamples) - 1]
    point = (sum(a) - sum(b)) / n
    return 100 * point, 100 * low, 100 * high


def clopper_pearson_upper(k: int, m: int, confidence: float = 0.95) -> float:
    """Exact one-sided upper confidence bound for a proportion of ``k`` in ``m``."""
    if k >= m:
        return 1.0
    return float(beta.ppf(confidence, k + 1, m - k))


def estimate_bound(outcomes: Iterable[Outcome], confidence: float = 0.95) -> tuple[float, int, int]:
    """Section 3.3: the share of approvals times the exact upper bound on the share of
    approvals that lost a faithful reading. ``retained`` stands in for the person who
    formalizes the sentences of the approved cases. Returns the bound as a share of all
    cases, the number of approvals that lost a reading and the number of approvals."""
    outcomes = list(outcomes)
    approved = [o for o in outcomes if o.decision == PASS]
    lost = sum(not o.retained for o in approved)
    if not approved:
        return 0.0, 0, 0
    return len(approved) / len(outcomes) * clopper_pearson_upper(lost, len(approved), confidence), lost, len(approved)


def validate_estimate(outcomes: dict[str, Outcome], labels: dict[str, str], groups: dict[str, object],
                      splits: int = 1000, seed: int = 0, confidence: float = 0.95) -> list[tuple[float, float]]:
    """Random splits into two halves that keep each group (a FOLIO story and its variants)
    on one side. The bound is computed on the approved cases of one half and the rate of
    false approvals is observed on the other, both as shares of all cases of their half.
    Returns ``[(bound, observed)]``, one pair per split."""
    rng = random.Random(seed)
    keys = sorted(set(groups.values()), key=str)
    results = []
    for _ in range(splits):
        shuffled = keys[:]
        rng.shuffle(shuffled)
        first = set(shuffled[:len(shuffled) // 2])
        half_a = [i for i in outcomes if groups[i] in first]
        half_b = [i for i in outcomes if groups[i] not in first]
        bound, _, _ = estimate_bound((outcomes[i] for i in half_a), confidence)
        observed = sum(outcomes[i].decision == PASS and labels[i] != PASS for i in half_b) / len(half_b)
        results.append((bound, observed))
    return results


def outer_bound(outcomes: Iterable[Outcome]) -> float:
    """``P(L)``: the share of all cases that lost some faithful reading."""
    outcomes = list(outcomes)
    return sum(not o.retained for o in outcomes) / len(outcomes)


# ----------------------------------------------------------------------------- pooling and figure

def labels_of(sets: dict[str, dict[str, dict]]) -> dict[tuple[str, str], str]:
    """Labels of every story, keyed by ``(set, story id)``."""
    return {(name, i): story["label"] for name, stories in sets.items() for i, story in stories.items()}


def pooled(by_set: dict[str, dict], keys: Iterable[tuple[str, str]] | None = None) -> dict[tuple[str, str], object]:
    """Join per-set results ``{set: {story id: x}}`` into ``{(set, story id): x}``,
    optionally restricted to some keys."""
    keys = None if keys is None else set(keys)
    return {(name, i): x for name, items in by_set.items() for i, x in items.items() if keys is None or (name, i) in keys}


def plot_main_figure(curves: dict[str, list[Summary]], judges_all: dict[str, Summary],
                     curves_shared: dict[str, list[Summary]], judges_shared: dict[str, Summary],
                     splits: list[tuple[float, float]], path: str | Path | None = None):
    """Figure 1 of the paper. ``curves`` maps a rule name to its summaries with one to
    seven readings; the judges are single points."""
    import matplotlib.pyplot as plt

    colours = {"Most frequent": "#eb6834", "Every reading": "#2a78d6", "Every reading, with edits": "#1baf7a"}
    markers = {"Most frequent": "s", "Every reading": "o", "Every reading, with edits": "^"}
    judge_markers = {"gpt-6.1-sol": "D", "Claude Opus 5.5": "P"}
    grey = "#52514e"
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.9), gridspec_kw={"wspace": 0.35})

    for name, summaries in curves.items():
        style = dict(color=colours[name], marker=markers[name], ms=4, lw=1.3, mec="white", mew=0.5, label=name)
        axes[0].plot([s.mean_size for s in summaries], [100 * s.retained_rate for s in summaries], **style)
        axes[1].plot([s.coverage for s in summaries], [100 * s.false_approval_rate for s in summaries], **style)
    for name, summaries in curves_shared.items():
        axes[2].plot([s.coverage for s in summaries], [100 * s.false_approval_rate for s in summaries],
                     color=colours[name], marker=markers[name], ms=4, lw=1.3, mec="white", mew=0.5, label=name)
    for axis, judges in ((axes[1], judges_all), (axes[2], judges_shared)):
        for name, s in judges.items():
            axis.plot([s.coverage], [100 * s.false_approval_rate], marker=judge_markers[name], ms=5.5, color=grey,
                      ls="none", label=f"{name} as judge")
    bounds, observed = [100 * b for b, _ in splits], [100 * o for _, o in splits]
    edges = [x / 4 for x in range(int(4 * max(bounds + observed)) + 2)]
    axes[3].hist(observed, bins=edges, color=colours["Every reading"], alpha=0.5, label="Observed, other half")
    axes[3].hist(bounds, bins=edges, color=colours["Most frequent"], alpha=0.5, label="Bound, estimation half")

    axes[0].set(xlabel="Readings kept per sentence", ylabel="Stories retaining every\nfaithful reading (%)")
    axes[1].set(xlabel="Coverage", ylabel="False approvals (%)")
    axes[2].set(xlabel="Coverage, shared subset", ylabel="False approvals (%)")
    axes[3].set(xlabel="Share of all cases (%)", ylabel="Random splits")
    for axis, letter in zip(axes, "abcd"):
        axis.set_title(f"({letter})", loc="left", fontsize=9)
        axis.grid(True, color="#e6e6e3", lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="best")
    if path is not None:
        fig.savefig(path, bbox_inches="tight")
    return fig


def plot_summary_figure(rows: list[tuple[str, str, Summary]], path: str | Path | None = None):
    """The figure at the top of the README: wrong and correct approvals of each system.

    ``rows`` are ``(label, kind, summary)`` from top to bottom, ``kind`` being ``"judge"``,
    ``"majority"`` or ``"every"``.
    """
    import matplotlib.pyplot as plt

    colours = {"judge": "#8a8986", "majority": "#eb6834", "every": "#2a78d6"}
    ink, muted = "#1f1f1e", "#6b6a67"
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.55 * len(rows) + 1.4), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    y = list(range(len(rows)))[::-1]
    panels = (
        ("Wrong approvals", "of the claims the document does not support", lambda s: s.false_approvals,
         lambda s: s.not_follow, "fewer is better"),
        ("Correct approvals", "of the claims the document supports", lambda s: s.approved_follow,
         lambda s: s.follow, "the rest go to a person"),
    )
    for axis, (title, subtitle, count, total, note) in zip(axes, panels):
        values = [100 * count(s) / total(s) for _, _, s in rows]
        axis.barh(y, values, height=0.62, color=[colours[k] for _, k, _ in rows], edgecolor="white", linewidth=2)
        top = max(values)
        for yi, value, (_, _, s) in zip(y, values, rows):
            axis.text(value + top * 0.02, yi, f"{value:.1f}%  ({count(s)} of {total(s)})", va="center",
                      color=ink, fontsize=9)
        axis.set_xlim(0, top * 1.55)
        axis.set_title(f"{title}\n", loc="left", fontsize=11, fontweight="bold", color=ink)
        axis.text(0, 1.02, f"{subtitle}, {note}", transform=axis.transAxes, color=muted, fontsize=9)
        axis.set_xticks([])
        for side in ("top", "right", "bottom"):
            axis.spines[side].set_visible(False)
        axis.spines["left"].set_color("#d9d8d5")
        axis.tick_params(left=False)
    axes[0].set_yticks(y, [label for label, _, _ in rows], color=ink)
    if path is not None:
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    return fig
