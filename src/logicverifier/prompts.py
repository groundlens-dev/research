# Copyright 2026 Javier Marín
# SPDX-License-Identifier: Apache-2.0
"""Prompts, decoding grammars and answer parsing.

The open translator and the hosted judges receive text built by the same functions, so
every system sees the same stories. The exact strings matter for reproduction: they are
the ones listed in Appendix A of the paper.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .logic import GRAMMAR, vocabulary

__all__ = [
    "FEW_SHOT_STORIES",
    "TRANSLATOR_INSTRUCTION",
    "FREE_INSTRUCTION",
    "JUDGE_SYSTEM",
    "story_block",
    "vocabulary_line",
    "formula_lines",
    "translator_prompt",
    "free_translator_prompt",
    "translator_grammar",
    "free_grammar",
    "parse_translation",
    "judge_prompt",
    "parse_judge",
    "CONTROL_STORY_INSTRUCTION",
    "control_story_prompt",
    "SENTENCE_GRAMMAR",
    "sentence_prompt",
    "COMMON_SENSE",
]

#: Three solved stories written for this work (none comes from FOLIO), checked with Z3.
FEW_SHOT_STORIES = [
    {"premises": ["All drugs that contain codeine require a prescription.",
                  "Dolorex is a drug that contains codeine.",
                  "No product that requires a prescription can be sold in supermarkets."],
     "conclusion": "Dolorex can be sold in supermarkets.",
     "premises-FOL": ["∀x ((Drug(x) ∧ ContainsCodeine(x)) → RequiresPrescription(x))",
                      "Drug(dolorex) ∧ ContainsCodeine(dolorex)",
                      "∀x (RequiresPrescription(x) → ¬SoldInSupermarkets(x))"],
     "conclusion-FOL": "SoldInSupermarkets(dolorex)",
     "label": "False"},
    {"premises": ["Every employee of the bank either works in Madrid or works in Lisbon.",
                  "Ana is an employee of the bank.",
                  "Ana does not work in Lisbon.",
                  "Some people who work in Madrid speak German."],
     "conclusion": "Ana speaks German.",
     "premises-FOL": ["∀x (BankEmployee(x) → (WorksInMadrid(x) ⊕ WorksInLisbon(x)))",
                      "BankEmployee(ana)",
                      "¬WorksInLisbon(ana)",
                      "∃x (WorksInMadrid(x) ∧ SpeaksGerman(x))"],
     "conclusion-FOL": "SpeaksGerman(ana)",
     "label": "Uncertain"},
    {"premises": ["Every certified pilot has passed a medical examination.",
                  "Lena is a certified pilot."],
     "conclusion": "Lena has passed a medical examination.",
     "premises-FOL": ["∀x (CertifiedPilot(x) → PassedMedicalExam(x))", "CertifiedPilot(lena)"],
     "conclusion-FOL": "PassedMedicalExam(lena)",
     "label": "True"},
]

TRANSLATOR_INSTRUCTION = (
    "Translate each premise and the conclusion into first-order logic.\n"
    "Use ∀ ∃ ¬ ∧ ∨ ⊕ → ↔, variables x y z, and only the predicates and constants listed in the vocabulary. "
    "Write one formula per line, numbered as in the input, and the conclusion on the line starting with C.\n\n")

FREE_INSTRUCTION = (
    "Translate each premise and the conclusion into first-order logic.\n"
    "Use ∀ ∃ ¬ ∧ ∨ ⊕ → ↔, variables x y z, predicate names in English CamelCase and constants in lowercase. "
    "Use the same predicate for the same idea everywhere.\n\n")

JUDGE_SYSTEM = "You answer logic questions with exactly one word: True, False or Uncertain."

_JUDGE_INSTRUCTION = (
    "Decide whether the conclusion follows from the premises, using only the premises.\n"
    "Answer True if the premises imply the conclusion, False if the premises imply that the conclusion is false, "
    "and Uncertain otherwise. Answer with one word.\n\n")


# ----------------------------------------------------------------------------- translator

def story_block(premises: Sequence[str], conclusion: str) -> str:
    numbered = "".join(f"{i + 1}. {p}\n" for i, p in enumerate(premises))
    return f"Premises:\n{numbered}Conclusion: {conclusion}\n"


def vocabulary_line(vocab: dict) -> str:
    predicates = ", ".join(f"{name}/{arity}" for name, arity in sorted(vocab["preds"].items()))
    constants = ", ".join(vocab["consts"]) or "none"
    return f"Vocabulary. Predicates (name/number of arguments): {predicates}. Constants: {constants}.\n"


def formula_lines(premise_formulas: Sequence[str], claim_formula: str) -> str:
    """The answer format: one numbered formula per premise and the claim on line C."""
    numbered = "".join(f"{i + 1}. {f}\n" for i, f in enumerate(premise_formulas))
    return f"{numbered}C. {claim_formula}"


def translator_prompt(story: dict, few_shot: bool = False) -> str:
    """Prompt with the story's vocabulary. Translator v2 is run with the three solved
    stories (``few_shot=True``); translators v3 and v4 without them."""
    prompt = TRANSLATOR_INSTRUCTION
    if few_shot:
        for shot in FEW_SHOT_STORIES:
            shot_vocab = vocabulary(shot["premises-FOL"] + [shot["conclusion-FOL"]], decimals=False)
            prompt += vocabulary_line(shot_vocab) + story_block(shot["premises"], shot["conclusion"]) + "FOL:\n"
            prompt += formula_lines(shot["premises-FOL"], shot["conclusion-FOL"]) + "\n\n"
    return prompt + vocabulary_line(story["vocab"]) + story_block(story["premises"], story["conclusion"]) + "FOL:\n"


def free_translator_prompt(premises: Sequence[str], conclusion: str, few_shot: bool = True) -> str:
    """Prompt without a vocabulary: the translator chooses its own names. Used only to
    build training data (ContractNLI pseudo-labels)."""
    prompt = FREE_INSTRUCTION
    if few_shot:
        for shot in FEW_SHOT_STORIES:
            prompt += story_block(shot["premises"], shot["conclusion"]) + "FOL:\n"
            prompt += formula_lines(shot["premises-FOL"], shot["conclusion-FOL"]) + "\n\n"
    return prompt + story_block(premises, conclusion) + "FOL:\n"


def _story_root(n_premises: int) -> str:
    lines = " ".join(f'"{i + 1}. " formula "\\n"' for i in range(n_premises))
    return f'root ::= {lines} "C. " formula'


def translator_grammar(story: dict) -> str:
    """Decoding grammar for one story: one line per premise and a claim line, with the
    predicates and constants restricted to the story's vocabulary."""
    vocab = story["vocab"]
    predicates = " | ".join(f'"{p}"' for p in sorted(vocab["preds"], key=len, reverse=True))
    constants = " | ".join(f'"{c}"' for c in sorted(vocab["consts"], key=len, reverse=True))
    number = '[0-9]+ ("." [0-9]+)?'
    grammar = re.sub(r"pred ::= .*", f"pred ::= {predicates}", GRAMMAR)
    grammar = re.sub(r"const ::= .*", f"const ::= {constants + ' | ' if constants else ''}{number}", grammar)
    return _story_root(len(story["premises"])) + "\n" + grammar


def free_grammar(n_premises: int) -> str:
    return _story_root(n_premises) + "\n" + GRAMMAR


def parse_translation(text: str | None, n_premises: int) -> list[str]:
    """Split a translator answer into one formula per premise and the claim formula
    (last). Missing lines become empty strings."""
    lines: dict[str, str] = {}
    for line in (text or "").strip().split("\n"):
        match = re.match(r"\s*(\d+|C)\s*[.)]\s*(.*)", line)
        if match and match.group(1) not in lines:
            lines[match.group(1)] = match.group(2).strip().strip("`").strip()
    return [lines.get(str(i + 1), "") for i in range(n_premises)] + [lines.get("C", "")]


# ----------------------------------------------------------------------------- judge

def judge_prompt(story: dict) -> str:
    prompt = _JUDGE_INSTRUCTION
    for shot in FEW_SHOT_STORIES:
        prompt += story_block(shot["premises"], shot["conclusion"]) + f"Answer: {shot['label']}\n\n"
    return prompt + story_block(story["premises"], story["conclusion"]) + "Answer:"


def parse_judge(text: str | None) -> str:
    """The first of True, False or Uncertain in the answer. Anything else counts as Uncertain."""
    match = re.search(r"\b(True|False|Uncertain)\b", text or "")
    return match.group(1) if match else "Uncertain"


# ----------------------------------------------------------------------------- controls outside FOLIO
# Used by the format training of translator v2 and by the controls of Appendix C.

CONTROL_STORY_INSTRUCTION = (
    "Translate the premises and the conclusion into first-order logic.\n"
    "Use ∀ ∃ ¬ ∧ ∨ ⊕ → ↔. Predicate names in English CamelCase, constants in lowercase, variables x y z. "
    "Use the same predicate for the same idea everywhere.\n\n")


def control_story_prompt(premises: Sequence[str], conclusion: str, few_shot: bool = True) -> str:
    """FOLIO without a vocabulary, with the first two solved stories as examples."""
    prompt = CONTROL_STORY_INSTRUCTION
    if few_shot:
        for shot in FEW_SHOT_STORIES[:2]:
            prompt += story_block(shot["premises"], shot["conclusion"]) + "FOL:\n"
            prompt += formula_lines(shot["premises-FOL"], shot["conclusion-FOL"]) + "\n\n"
    return prompt + story_block(premises, conclusion) + "FOL:\n"


_SENTENCE_INSTRUCTION = (
    "Translate the legal sentence into first-order logic. The sentence may be in any EU language. "
    "Always use English predicate names in CamelCase, lowercase constants, variables x y z, "
    "and write numbers as plain digits.\n\n")

_SENTENCE_SHOTS = [
    ("Member States shall not grant aid exceeding EUR 200 000.",
     "∀x (MemberState(x) → ¬∃y (Aid(y) ∧ Grants(x, y) ∧ Exceeds(y, 200000)))"),
    ("The period referred to in paragraph 1 shall not exceed 30 days.",
     "∀x (PeriodInParagraph(x) → ¬ExceedsDays(x, 30))"),
    ("Each operator shall keep the records for at least five years.",
     "∀x (Operator(x) → KeepsRecordsAtLeastYears(x, 5))"),
    ("Products containing more than 0.5 % of the substance may not be placed on the market.",
     "∀x ((Product(x) ∧ ContainsMoreThanPercent(x, 0.5)) → ¬MayBePlacedOnMarket(x))"),
]

SENTENCE_GRAMMAR = 'root ::= " " formula\n' + GRAMMAR


def sentence_prompt(sentence: str, few_shot: bool) -> str:
    """One DGT sentence, in any of the six languages."""
    prompt = _SENTENCE_INSTRUCTION
    if few_shot:
        for text, formula in _SENTENCE_SHOTS:
            prompt += f"Sentence: {text}\nFOL: {formula}\n\n"
    return prompt + f"Sentence: {sentence}\nFOL:"


#: Common-sense questions ("is X a kind of Y?") per language: template, yes words,
#: no words and two solved examples.
COMMON_SENSE = {
    "en": ('Question: Is "{x}" a kind of "{y}"? Answer "yes" or "no".\nAnswer:', ("yes",), ("no",),
           [("apple", "fruit", "yes"), ("hammer", "beverage", "no")]),
    "es": ('Pregunta: ¿«{x}» es un tipo de «{y}»? Responde «sí» o «no».\nRespuesta:', ("sí", "si"), ("no",),
           [("manzana", "fruta", "sí"), ("martillo", "bebida", "no")]),
    "fr": ('Question : « {x} » est-il un type de « {y} » ? Répondez « oui » ou « non ».\nRéponse :', ("oui",), ("non",),
           [("pomme", "fruit", "oui"), ("marteau", "boisson", "non")]),
    "de": ('Frage: Ist „{x}“ eine Art von „{y}“? Antworten Sie mit „ja“ oder „nein“.\nAntwort:', ("ja",), ("nein",),
           [("Apfel", "Obst", "ja"), ("Hammer", "Getränk", "nein")]),
    "nl": ('Vraag: Is "{x}" een soort "{y}"? Antwoord met "ja" of "nee".\nAntwoord:', ("ja",), ("nee",),
           [("appel", "fruit", "ja"), ("hamer", "drank", "nee")]),
    "it": ('Domanda: «{x}» è un tipo di «{y}»? Rispondi «sì» o «no».\nRisposta:', ("sì", "si"), ("no",),
           [("mela", "frutto", "sì"), ("martello", "bevanda", "no")]),
}
