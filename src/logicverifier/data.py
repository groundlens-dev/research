# Copyright 2026 Javier Marín
# SPDX-License-Identifier: Apache-2.0
"""Test sets and training mixtures.

Test sets (Section 4 of the paper). FOLIO original keeps the validation stories whose
expert formulas reproduce the label under Z3. FOLIO with new names renames every named
individual and shuffles the premises. FOLIO counterfactual also removes one premise and
recomputes the label with Z3. No text or label is written by a person or a model.

Training mixtures (Appendix A). Translator v2 learns from FOLIO training stories and DGT
sentences formalized by the base model. Translator v3 adds synthetic stories, ContractNLI
clauses and renamed FOLIO stories. Translator v4 adds stories over real vocabularies in
six languages.
"""

from __future__ import annotations

import collections
import io
import json
import random
import re
import unicodedata
import urllib.request
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path

from .logic import VARIABLES, FOLError, decide, parse, vocabulary
from .prompts import (CONTROL_STORY_INSTRUCTION, FREE_INSTRUCTION, TRANSLATOR_INSTRUCTION, formula_lines,
                      sentence_prompt, story_block, translator_prompt)

__all__ = [
    "LANGUAGES",
    "SETS",
    "read_jsonl",
    "write_jsonl",
    "download_folio",
    "build_test_sets",
    "load_test_sets",
    "prepare_dgt",
    "common_sense_questions",
    "contractnli_cases",
    "dgt_formula_is_faithful",
    "build_v2_mixture",
    "build_v3_mixture",
    "build_v4_mixture",
]

LANGUAGES = ["en", "es", "fr", "de", "nl", "it"]

#: The three test sets, in the order of the paper.
SETS = {"folio_original": "FOLIO original",
        "folio_new_names": "FOLIO with new names",
        "folio_counterfactual": "FOLIO counterfactual"}

#: FOLIO is read at a fixed commit of its repository.
FOLIO_COMMIT = "5d7bb84c7edab3fb358e057d2807f19cf5cf5e2d"
FOLIO_URL = "https://raw.githubusercontent.com/Yale-LILY/FOLIO/{commit}/data/v0.0/folio-{split}.jsonl"

#: Replacement names, mostly gender neutral so that pronouns in the text still fit.
NEW_NAMES = ["Quinn", "Rowan", "Sasha", "Robin", "Avery", "Morgan", "Kai", "Noor", "Ari", "Emery", "Jules", "Remy",
             "Sage", "Tenzin", "Ashby", "Lumi", "Marlo", "Oakley", "Perrin", "Reese", "Skyler", "Toby", "Wren", "Zephyr",
             "Indigo", "Hollis", "Frankie", "Ellis", "Darcy", "Blair"]


# ----------------------------------------------------------------------------- files

def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_folio(split: str, path: str | Path) -> Path:
    """Download one FOLIO split (``"train"`` or ``"validation"``) at the pinned commit."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FOLIO_URL.format(commit=FOLIO_COMMIT, split=split), path)
    return path


# ----------------------------------------------------------------------------- test sets

def _named_constants(formulas: Sequence[str]) -> set[str]:
    """Lowercase arguments that are not variables, as candidates for renaming."""
    found = set()
    for formula in formulas:
        for args in re.findall(r"\(([^()]*)\)", formula):
            for arg in args.split(","):
                arg = arg.strip()
                if re.fullmatch(r"[a-z][A-Za-z0-9]*", arg) and arg not in VARIABLES:
                    found.add(arg)
    return found


def _text_form(constant: str) -> str:
    """``jamesCameron`` → ``James Cameron``."""
    parts = re.findall(r"[a-z]+|[A-Z][a-z]*|\d+", constant)
    return " ".join(p[:1].upper() + p[1:] for p in parts)


def _rename(premises, premise_formulas, conclusion, conclusion_formula, rng: random.Random):
    """Give a new name to every individual that is named in the text, in the text and
    in the formulas at the same time."""
    texts = premises + [conclusion]
    pool = [n for n in NEW_NAMES if not any(n in t for t in texts)]
    rng.shuffle(pool)
    mapping = {}
    for constant in sorted(_named_constants(premise_formulas + [conclusion_formula])):
        written = _text_form(constant)
        if not re.search(r"\b" + re.escape(written) + r"\b", " ".join(texts)):
            continue
        if not pool:
            break
        mapping[constant] = (written, pool.pop())

    def in_text(text):
        for written, new in mapping.values():
            text = re.sub(r"\b" + re.escape(written) + r"\b", new, text)
        return text

    def in_formula(formula):
        for constant, (_, new) in mapping.items():
            formula = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(constant) + r"(?![A-Za-z0-9_])", new.lower(), formula)
        return formula

    renamed = {constant: new for constant, (_, new) in mapping.items()}
    return ([in_text(t) for t in premises], [in_formula(f) for f in premise_formulas],
            in_text(conclusion), in_formula(conclusion_formula), renamed)


def _ascii(identifier: str) -> str:
    decomposed = unicodedata.normalize("NFKD", identifier)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.replace("ł", "l").replace("Ł", "L").encode("ascii", "ignore").decode()


def _normalise(formula: str) -> str:
    """Write every name so that the vocabulary extraction and the grammar accept it:
    no space before an argument list (``Mammal (x)``) and ASCII letters only."""
    formula = re.sub(r"([A-ZÀ-Ž][A-Za-z0-9_À-ž]*)\s+\(", r"\1(", formula)
    return re.sub(r"[A-Za-z0-9_]*[^\x00-\x7F∀∃¬∧∨⊕→↔≠\s(),=][\w]*",
                  lambda m: _ascii(m.group(0)) if any(ch.isalpha() for ch in m.group(0)) else m.group(0), formula)


def _story(story_id, premises, premise_formulas, conclusion, conclusion_formula, label, **extra) -> dict:
    """A test story with its vocabulary, extracted from the normalised expert formulas."""
    normal_premises = [_normalise(f) for f in premise_formulas]
    normal_conclusion = _normalise(conclusion_formula)
    if (normal_premises, normal_conclusion) != (list(premise_formulas), conclusion_formula):
        if decide(normal_premises, normal_conclusion) != label:
            raise ValueError(f"normalising the formulas of {story_id} changes its label")
    return {"id": story_id, "premises": premises, "premises-FOL": normal_premises, "conclusion": conclusion,
            "conclusion-FOL": normal_conclusion, "label": label,
            "vocab": vocabulary(normal_premises + [normal_conclusion]), **extra}


def build_test_sets(validation: Sequence[dict], seed: int = 0, controls_per_story: int = 1,
                    max_changed_per_story: int = 3) -> tuple[dict[str, list[dict]], dict]:
    """Build the three test sets from the FOLIO validation split.

    Returns the sets, keyed as in :data:`SETS`, and a report with the counts.
    """
    rng = random.Random(seed)
    stats = collections.Counter()
    original, new_names, counterfactual = [], [], []
    for idx, row in enumerate(validation):
        P, F, C, G = row["premises"], row["premises-FOL"], row["conclusion"], row["conclusion-FOL"]
        if len(P) != len(F) or len(P) < 2:
            stats["skipped_shape"] += 1
            continue
        if decide(F, G) != row["label"]:
            stats["skipped_formulas_disagree_with_label"] += 1
            continue
        stats["stories_used"] += 1
        original.append(_story(f"o{idx}", P, F, C, G, row["label"], source_idx=idx, kind="original"))

        P2, F2, C2, G2, renamed = _rename(P, F, C, G, rng)
        order = list(range(len(P2)))
        rng.shuffle(order)
        P2, F2 = [P2[j] for j in order], [F2[j] for j in order]
        if decide(F2, G2) == row["label"]:
            new_names.append(_story(f"s{idx}", P2, F2, C2, G2, row["label"], source_idx=idx, kind="surface",
                                    renamed=renamed))

        changed, unchanged = [], []
        for i in range(len(P)):
            label = decide(F[:i] + F[i + 1:], G)
            if label != "Error":
                (changed if label != row["label"] else unchanged).append((i, label))
        rng.shuffle(changed)
        rng.shuffle(unchanged)
        picks = ([("drop_changed", x) for x in changed[:max_changed_per_story]]
                 + [("drop_control", x) for x in unchanged[:controls_per_story]])
        for kind, (i, label) in picks:
            P3, F3, C3, G3, renamed = _rename(P[:i] + P[i + 1:], F[:i] + F[i + 1:], C, G, rng)
            order = list(range(len(P3)))
            rng.shuffle(order)
            P3, F3 = [P3[j] for j in order], [F3[j] for j in order]
            if decide(F3, G3) != label:
                stats["skipped_recheck"] += 1
                continue
            counterfactual.append(_story(f"d{idx}_{i}", P3, F3, C3, G3, label, orig_label=row["label"],
                                         source_idx=idx, dropped_premise=i, kind=kind, renamed=renamed))
            stats[kind] += 1
            stats[f"{kind}:{row['label']}->{label}"] += 1
    sets = {"folio_original": original, "folio_new_names": new_names, "folio_counterfactual": counterfactual}
    report = {name: len(rows) for name, rows in sets.items()}
    report["counterfactual_labels"] = dict(collections.Counter(s["label"] for s in counterfactual))
    report.update(stats)
    return sets, report


def load_test_sets(directory: str | Path) -> dict[str, dict[str, dict]]:
    """The three test sets as ``{set: {story id: story}}``."""
    return {name: {s["id"]: s for s in read_jsonl(Path(directory) / f"{name}.jsonl")} for name in SETS}


# ----------------------------------------------------------------------------- DGT

_DGT_PAIRS = {"es": "en-es", "fr": "en-fr", "de": "de-en", "nl": "en-nl", "it": "en-it"}
_DGT_URL = "https://object.pouta.csc.fi/OPUS-DGT/{version}/moses/{pair}.txt.zip"
_MODAL = re.compile(r"\b(shall|must|may|should)\b", re.I)
_NEGATION_OR_NUMBER = re.compile(r"\b(not|no|neither|nor|unless|except|without|never)\b", re.I)
_SEPARATORS = re.compile(r"[\s.,'  ]")


def _dgt_candidate(english: str) -> bool:
    words = english.split()
    return (8 <= len(words) <= 30 and bool(_MODAL.search(english))
            and bool(_NEGATION_OR_NUMBER.search(english) or re.search(r"\d", english))
            and not any(c in english for c in "[]{};|") and english.count("(") <= 2)


def _numbers(text: str) -> list[str]:
    return sorted((_SEPARATORS.sub("", t).lstrip("0") or "0") for t in re.findall(r"\d[\d\s.,'  ]*\d|\d", text))


def _misaligned(row: dict) -> str | None:
    english = row["en"]
    for lang in LANGUAGES[1:]:
        other = row[lang]
        if _numbers(other) != _numbers(english):
            return "numbers"
        if not 0.6 <= len(other) / max(1, len(english)) <= 1.8:
            return "length"
        if english[:1].isupper() and other[:1].islower():
            return "fragment"
        if english[-1:] in ".:;" and other[-1:] in ".:;" and english[-1] != other[-1]:
            return "fragment"
    return None


def prepare_dgt(zip_dir: str | Path, n_test: int = 300, n_train: int = 1500, min_similarity: float = 0.75,
                max_embedded: int = 9000) -> tuple[list[dict], list[dict], dict]:
    """Sentences of the DGT translation memory present in six languages, with alignment checks.

    A row is kept when every language has a single translation, the same numbers, a
    comparable length, does not look like a fragment, and its LaBSE embedding is close to
    the English one. Returns disjoint test and training rows and a report. Needs
    ``sentence-transformers`` and, ideally, a GPU.
    """
    zip_dir = Path(zip_dir)
    zip_dir.mkdir(parents=True, exist_ok=True)
    candidates, conflicts = None, {}
    for lang, pair in _DGT_PAIRS.items():
        archive = zip_dir / f"DGT.{pair}.zip"
        if not archive.exists():
            for version in ("v2019", "v4", "v3"):
                try:
                    urllib.request.urlretrieve(_DGT_URL.format(version=version, pair=pair), archive)
                    break
                except OSError:
                    continue
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            english_file = next(n for n in names if n.endswith(".en"))
            other_file = next(n for n in names if n.endswith("." + lang))
            found, duplicates = {}, set()
            with z.open(english_file) as a, z.open(other_file) as b:
                for line_en, line_xx in zip(io.TextIOWrapper(a, encoding="utf-8"), io.TextIOWrapper(b, encoding="utf-8")):
                    english, other = line_en.strip(), line_xx.strip()
                    if (candidates is None and _dgt_candidate(english)) or (candidates is not None and english in candidates):
                        if english in found and found[english] != other:
                            duplicates.add(english)
                        found.setdefault(english, other)
        for english in duplicates:
            found.pop(english, None)
        conflicts[lang] = len(duplicates)
        if candidates is None:
            candidates = {en: {"en": en, lang: text} for en, text in found.items()}
        else:
            candidates = {en: {**candidates[en], lang: found[en]} for en in candidates if en in found}

    rows = [r for r in candidates.values() if all(r.get(lang) for lang in LANGUAGES)]
    random.Random(0).shuffle(rows)
    reasons = collections.Counter()
    passed = []
    for row in rows:
        reason = _misaligned(row)
        reasons[reason or "ok"] += 1
        if reason is None:
            passed.append(row)

    from sentence_transformers import SentenceTransformer
    import torch
    encoder = SentenceTransformer("sentence-transformers/LaBSE", device="cuda" if torch.cuda.is_available() else "cpu")
    pool = passed[:max_embedded]
    embeddings = encoder.encode([r[lang] for r in pool for lang in LANGUAGES], batch_size=256,
                                convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    embeddings = embeddings.view(len(pool), len(LANGUAGES), -1)
    similarity = (embeddings[:, 1:, :] * embeddings[:, :1, :]).sum(-1).min(-1).values.tolist()
    kept = [row for row, s in zip(pool, similarity) if s >= min_similarity]
    test, train = kept[:n_test], kept[n_test:n_test + n_train]
    report = {"candidates": len(rows), "conflicting_duplicates_dropped": conflicts, "simple_checks": dict(reasons),
              "embedded": len(pool), "min_sim_threshold": min_similarity, "kept_after_embedding": len(kept),
              "test": len(test), "train": len(train)}
    return test, train, report


_NUMBER_TOKEN = re.compile(r"\d[\d\s.,]*\d|\d")
_QUANTITY = re.compile(r"(\d[\d\s.,]*\d|\d)\s*(%|per\s*cent|EUR|euro|days?|weeks?|months?|years?|hours?|tonnes?|kg|mg|g\b|km|m2|litres?)", re.I)
_NEGATION_WORD = re.compile(r"\b(not|no|never|neither|nor|without|cannot|none)\b", re.I)
_WORD_NUMBERS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
_WORD_NUMBERS.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
                      "ninety": "90", "hundred": "100"})


def _plain_numbers(text: str) -> list[str]:
    """Numbers of an English sentence as plain digits (``200 000`` → ``200000``)."""
    out = []
    for match in _NUMBER_TOKEN.finditer(text):
        token = re.sub(r"\s", "", match.group(0).strip()).replace(",", "").rstrip(".")
        if re.fullmatch(r"\d+(\.\d+)?", token):
            out.append(token if "." in token else token.lstrip("0") or "0")
    return out


def dgt_formula_is_faithful(english: str, formula: str) -> tuple[bool, str]:
    """Checks that need no reference formula: the formula parses, invents no number,
    keeps every quantity of the sentence, and negates exactly when the sentence does."""
    try:
        parse(formula)
    except FOLError:
        return False, "parse"
    in_text = _plain_numbers(english) + [_WORD_NUMBERS[w.lower()] for w in re.findall(r"[A-Za-z]+", english)
                                         if w.lower() in _WORD_NUMBERS]
    in_formula = [n if "." in n else n.lstrip("0") or "0"
                  for n in re.findall(r"(?<![A-Za-z_])\d+(?:\.\d+)?(?![A-Za-z_])", formula)]
    if any(n not in in_text for n in in_formula):
        return False, "invented_number"
    quantities = [_plain_numbers(m.group(1))[0] for m in _QUANTITY.finditer(english) if _plain_numbers(m.group(1))]
    if any(q not in in_formula for q in quantities):
        return False, "missing_quantity"
    if bool(_NEGATION_WORD.search(english)) != ("¬" in formula):
        return False, "polarity"
    return True, "ok"


# ----------------------------------------------------------------------------- common sense

#: Everyday Wikidata classes whose members become the "is X a kind of Y?" questions.
_WIKIDATA_CLASSES = ["Q42889", "Q11436", "Q34379", "Q41176", "Q39546", "Q12136", "Q12140", "Q2095", "Q40050", "Q728",
                     "Q11460", "Q349", "Q14745", "Q1364", "Q11004", "Q6256", "Q515", "Q11344", "Q8142", "Q34770", "Q634"]


def common_sense_questions(n: int = 300, seed: int = 0) -> list[dict]:
    """Yes or no questions "is X a kind of Y?" with labels in the six languages, from
    Wikidata. Each positive question is followed, when one is found, by a negative one
    about the same item and another class. The answers depend on the live Wikidata, so
    the stored file, not a new run, is the one used in the paper."""
    import time
    import urllib.parse

    endpoint = "https://query.wikidata.org/sparql"

    def sparql(query):
        for attempt in range(4):
            try:
                body = urllib.parse.urlencode({"query": query, "format": "json"}).encode()
                request = urllib.request.Request(endpoint, data=body, headers={
                    "User-Agent": "GroundLensResearch/0.1", "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/sparql-results+json"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    return json.loads(response.read())
            except OSError:
                time.sleep(5 * (attempt + 1))
        return None

    def labels(var):
        return "\n".join(f'?{var} rdfs:label ?{var}_{lang} . FILTER(lang(?{var}_{lang})="{lang}")' for lang in LANGUAGES)

    class_labels, members = {}, {}
    for cls in _WIKIDATA_CLASSES:
        found = sparql(f"SELECT ?c {' '.join('?c_' + l for l in LANGUAGES)} WHERE {{ VALUES ?c {{ wd:{cls} }} {labels('c')} }}")
        if not found or not found["results"]["bindings"]:
            continue
        binding = found["results"]["bindings"][0]
        class_labels[cls] = {lang: binding["c_" + lang]["value"] for lang in LANGUAGES}
        found = sparql(f"""SELECT DISTINCT ?item ?sl {' '.join('?item_' + l for l in LANGUAGES)} WHERE {{
          {{ ?item wdt:P279 ?x . ?x wdt:P279* wd:{cls} . }} UNION {{ ?item wdt:P31 ?x . ?x wdt:P279* wd:{cls} . }}
          ?item wikibase:sitelinks ?sl . FILTER(?sl > 40)
          {labels('item')}
        }} ORDER BY DESC(?sl) LIMIT 20""")
        if not found:
            continue
        items = []
        for binding in found["results"]["bindings"]:
            item_labels = {lang: binding["item_" + lang]["value"] for lang in LANGUAGES}
            if item_labels["en"].lower() != class_labels[cls]["en"].lower():
                items.append((binding["item"]["value"].split("/")[-1], item_labels))
        members[cls] = items

    rng = random.Random(seed)
    positives = [(cls, item, item_labels) for cls, items in members.items() for item, item_labels in items]
    rng.shuffle(positives)
    questions = []
    for cls, item, item_labels in positives:
        if len(questions) >= n:
            break
        questions.append({"item": item, "class": cls, "answer": True, "x": item_labels, "y": class_labels[cls]})
        others = [k for k in members if k != cls]
        rng.shuffle(others)
        for other in others[:3]:
            answer = sparql(f"ASK {{ wd:{item} (wdt:P31|wdt:P279)/wdt:P279* wd:{other} }}")
            if answer is not None and answer.get("boolean") is False:
                questions.append({"item": item, "class": other, "answer": False, "x": item_labels, "y": class_labels[other]})
                break
    return questions


# ----------------------------------------------------------------------------- ContractNLI

def contractnli_cases(train_json: dict, max_cases: int = 3000, max_clause_words: int = 70) -> tuple[list[dict], dict]:
    """Cases from the ContractNLI training split: the clauses that support an Entailment or
    a Contradiction become the premises and the hypothesis becomes the claim."""
    hypotheses = {key: value["hypothesis"] for key, value in train_json["labels"].items()}
    cases, reasons = [], collections.Counter()
    for document in train_json["documents"]:
        text, spans = document["text"], document["spans"]
        for hypothesis_id, annotation in document["annotation_sets"][0]["annotations"].items():
            if annotation["choice"] not in ("Entailment", "Contradiction"):
                reasons["not_mentioned"] += 1
                continue
            clauses = [re.sub(r"\s+", " ", text[spans[i][0]:spans[i][1]]).strip(" ;,") for i in annotation["spans"]]
            clauses = [c for c in clauses if c]
            if not clauses or len(clauses) > 3:
                reasons["too_many_or_no_clauses"] += 1
                continue
            if any(len(c.split()) > max_clause_words for c in clauses):
                reasons["clause_too_long"] += 1
                continue
            cases.append({"id": f"{document['id']}_{hypothesis_id}", "premises": clauses,
                          "conclusion": hypotheses[hypothesis_id],
                          "label": "True" if annotation["choice"] == "Entailment" else "False"})
            reasons["kept"] += 1
    random.Random(0).shuffle(cases)
    return cases[:max_cases], dict(reasons)


# ----------------------------------------------------------------------------- training mixtures

def _example(premises, premise_formulas, conclusion, conclusion_formula, source, extra_predicates=()) -> dict:
    """A story in the zero-shot vocabulary format used to train and run v3 and v4. The
    vocabulary line may list predicates that the story does not use."""
    vocab = vocabulary(list(premise_formulas) + [conclusion_formula], decimals=False)
    for name in extra_predicates:
        vocab["preds"].setdefault(name, 1)
    story = {"premises": premises, "conclusion": conclusion, "vocab": vocab}
    return {"prompt": translator_prompt(story), "target": formula_lines(premise_formulas, conclusion_formula),
            "src": source}


def build_v2_mixture(folio_train: Sequence[dict], dgt_train: Sequence[dict], dgt_formulas: Sequence[str]) -> tuple[list[dict], dict]:
    """Format training of translator v2.

    FOLIO training stories (their last premise plays the claim) are repeated three times,
    and each FOLIO sentence appears once on its own. A DGT formula written by the base
    model for the English sentence is kept if it passes :func:`dgt_formula_is_faithful`
    and becomes the answer for the same sentence in all six languages.
    """
    examples, stories, seen = [], {}, set()
    for row in folio_train:
        stories.setdefault(row["story_id"], row)
    n_stories = n_sentences = 0
    for row in stories.values():
        P, F = row["premises"], row["premises-FOL"]
        if len(P) < 2 or len(P) != len(F):
            continue
        try:
            for formula in F:
                parse(formula)
        except FOLError:
            continue
        prompt = CONTROL_STORY_INSTRUCTION + story_block(P[:-1], P[-1]) + "FOL:\n"
        target = formula_lines(F[:-1], F[-1])
        examples += [{"prompt": prompt, "target": target, "src": "folio_story"}] * 3
        n_stories += 1
        for sentence, formula in zip(P, F):
            if sentence not in seen:
                seen.add(sentence)
                examples.append({"prompt": sentence_prompt(sentence, False), "target": " " + formula, "src": "folio_sent"})
                n_sentences += 1
    reasons, kept = collections.Counter(), 0
    for row, formula in zip(dgt_train, dgt_formulas):
        ok, reason = dgt_formula_is_faithful(row["en"], formula)
        reasons[reason] += 1
        if ok:
            kept += 1
            for lang in LANGUAGES:
                examples.append({"prompt": sentence_prompt(row[lang], False), "target": " " + formula, "src": f"dgt_{lang}"})
    random.Random(0).shuffle(examples)
    report = {"folio_stories": n_stories, "folio_sentences": n_sentences, "dgt_kept": kept,
              "dgt_reasons": dict(reasons), "total": len(examples)}
    return examples, report


def _suspicious(formula: str) -> bool:
    """FOLIO training formulas with an evident error: ∃x (A → B), or unreadable."""
    try:
        tree = parse(formula)
    except (FOLError, RecursionError):
        return True

    def walk(node):
        if not isinstance(node, tuple):
            return False
        if node[0] == "∃" and isinstance(node[2], tuple) and node[2][0] == "→":
            return True
        return any(walk(x) for x in node[1:] if isinstance(x, tuple))

    return walk(tree)


def _both_or_neither_as_xor(text: str, formula: str) -> bool:
    return bool(re.search(r"either (both )?.+ and .+ or neither", text, re.I)) and "∧" not in formula and "⊕" in formula


def build_v3_mixture(synthetic: Sequence[dict], contract: Sequence[dict], folio_train: Sequence[dict],
                     v2_mixture: Sequence[dict], max_dgt: int = 2500) -> tuple[list[dict], dict]:
    """Training mixture of translator v3, all stories in the zero-shot vocabulary format.

    ``synthetic`` are stories from :func:`logicverifier.synthetic.domain_stories`, ``contract``
    the verified ContractNLI examples, and the DGT examples are taken from the v2 mixture.
    FOLIO training stories with suspicious formulas are left out; the others appear as they
    are and once more with new names and shuffled premises.
    """
    rng = random.Random(0)
    examples, stats = [], collections.Counter()
    for story in synthetic:
        examples.append(_example(story["premises"], story["premises-FOL"], story["conclusion"],
                                 story["conclusion-FOL"], "synth", story.get("distractors") or ()))
        stats["synth"] += 1
    for example in contract:
        examples.append({"prompt": example["prompt"], "target": example["target"], "src": "contract"})
        stats["contract"] += 1

    names = ["Quinn", "Rowan", "Sasha", "Robin", "Avery", "Morgan", "Kai", "Noor", "Ari", "Emery", "Jules", "Remy",
             "Sage", "Tenzin", "Wren"]
    seen = set()
    for row in folio_train:
        all_premises, all_formulas = row["premises"], row["premises-FOL"]
        if len(all_premises) != len(all_formulas) or len(all_premises) < 3 or tuple(all_premises) in seen:
            continue
        seen.add(tuple(all_premises))
        P, F, C, G = all_premises[:-1], all_formulas[:-1], all_premises[-1], all_formulas[-1]
        label = decide(F, G)
        if label == "Error":
            stats["folio_inconsistent_or_unparsable"] += 1
            continue
        if any(_suspicious(f) for f in F + [G]) or any(_both_or_neither_as_xor(t, f) for t, f in zip(P + [C], F + [G])):
            stats["folio_suspicious"] += 1
            continue
        examples.append(_example(P, F, C, G, "folio"))
        stats["folio"] += 1

        constants = sorted(vocabulary(F + [G], decimals=False)["consts"])
        pool = [n for n in names if not any(n in t for t in P + [C])]
        rng.shuffle(pool)
        mapping = {}
        for constant in constants:
            written = " ".join(w[:1].upper() + w[1:] for w in re.findall(r"[a-z]+|[A-Z][a-z]*|\d+", constant))
            if pool and re.search(r"\b" + re.escape(written) + r"\b", " ".join(P + [C])):
                mapping[constant] = (written, pool.pop())

        def in_text(text):
            for written, new in mapping.values():
                text = re.sub(r"\b" + re.escape(written) + r"\b", new, text)
            return text

        def in_formula(formula):
            for constant, (_, new) in mapping.items():
                formula = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(constant) + r"(?![A-Za-z0-9_])", new.lower(), formula)
            return formula

        order = list(range(len(P)))
        rng.shuffle(order)
        P2, F2 = [in_text(P[j]) for j in order], [in_formula(F[j]) for j in order]
        if decide(F2, in_formula(G)) == label:
            examples.append(_example(P2, F2, in_text(C), in_formula(G), "folio_aug"))
            stats["folio_aug"] += 1

    dgt = [e for e in v2_mixture if e.get("src", "").startswith("dgt")]
    rng.shuffle(dgt)
    for example in dgt[:max_dgt]:
        examples.append({"prompt": example["prompt"], "target": example["target"], "src": example["src"]})
        stats["dgt"] += 1
    rng.shuffle(examples)
    return examples, {"total": len(examples), **stats}


def build_v4_mixture(v3_mixture: Sequence[dict], vocabulary_stories: Sequence[dict],
                     free_share: float = 0.2) -> tuple[list[dict], dict]:
    """Training mixture of translator v4: the v3 mixture plus stories over real
    vocabularies in six languages. One example in five loses its vocabulary line, so that
    the translator keeps working when no vocabulary is given."""
    examples = [dict(e) for e in v3_mixture]
    stats = collections.Counter(e.get("src", "?") for e in examples)
    for story in vocabulary_stories:
        examples.append(_example(story["premises"], story["premises-FOL"], story["conclusion"], story["conclusion-FOL"],
                                 f"vocab_{story['source']}_{story['lang']}", story.get("distractors", ())))
        stats[f"vocab_{story['lang']}"] += 1
    rng = random.Random(1)
    for example in examples:
        prompt = example["prompt"]
        body = prompt[len(TRANSLATOR_INSTRUCTION):]
        if prompt.startswith(TRANSLATOR_INSTRUCTION) and body.startswith("Vocabulary.") and rng.random() < free_share:
            example["prompt"] = FREE_INSTRUCTION + body.split("\n", 1)[1]
            example["src"] += "_free"
            stats["free_format"] += 1
    random.Random(0).shuffle(examples)
    return examples, {"total": len(examples), **stats}
