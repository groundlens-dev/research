# Copyright 2026 Javier Marín
# SPDX-License-Identifier: Apache-2.0
from logicverifier.logic import (
    decide,
    decide_reading_sets,
    directed_edits,
    equivalent,
    is_valid_reading,
    parse,
    show,
    vocabulary,
)

SUPPLIERS = ["∀x (Supplier(x) → ¬PaysVat(x))", "Supplier(acme)"]


def test_decide_labels():
    assert decide(SUPPLIERS, "PaysVat(acme)") == "False"
    assert decide(SUPPLIERS, "¬PaysVat(acme)") == "True"
    assert decide(SUPPLIERS, "Owns(acme, acme)") == "Uncertain"
    assert decide(["P(a)", "¬P(a)"], "Q(a)") == "Error"


def test_precedence_and_round_trip():
    assert parse("A(x) ∧ B(x) → C(x)") == ("→", ("∧", ("pred", "A", ("x",)), ("pred", "B", ("x",))), ("pred", "C", ("x",)))
    formula = "∀x (Supplier(x) → (PaysVat(x) ⊕ IsExempt(x)))"
    assert equivalent(show(parse(formula)), formula) is True


def test_a_lost_faithful_reading_is_the_only_route_to_a_false_approval():
    faithful, wrong = "∀x (Supplier(x) → ¬PaysVat(x))", "∀x (Supplier(x) → PaysVat(x))"
    claim = ["PaysVat(acme)"]
    assert decide_reading_sets([[faithful, wrong], ["Supplier(acme)"]], claim) == "Uncertain"
    assert decide_reading_sets([[faithful], ["Supplier(acme)"]], claim) == "False"
    assert decide_reading_sets([[wrong], ["Supplier(acme)"]], claim) == "True"


def test_no_claim_reading_is_referred():
    assert decide_reading_sets([["P(a)"]], []) == "Uncertain"


def test_vocabulary_and_validity():
    vocab = vocabulary(["Owns(ana, y42.3billion)", "∀x (Owns(x, ana) → Rich(x))"])
    assert vocab == {"preds": {"Owns": 2, "Rich": 1}, "consts": ["ana", "y42.3billion"]}
    assert is_valid_reading("Rich(ana)", vocab)
    assert not is_valid_reading("Rich(bob)", vocab)
    assert not is_valid_reading("Owns(ana)", vocab)


def test_directed_edits():
    assert directed_edits("A(x) ⊕ B(x)", "xor") == ["(A(x) ∨ B(x))"]
    assert directed_edits("Owns(ana, bob)", "args") == ["Owns(bob, ana)"]
