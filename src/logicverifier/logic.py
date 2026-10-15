# Copyright 2026 Javier Marín
# SPDX-License-Identifier: Apache-2.0
"""First-order logic in FOLIO notation, decided by Z3.

Formulas use the symbols ∀ ∃ ¬ ∧ ∨ ⊕ → ↔ = ≠, predicates such as ``Owns(x, acme)``,
single-letter variables and lowercase constants. This module parses them, decides
whether a claim follows from premises, compares formulas for logical equivalence and
implements the decision over sets of candidate readings (Definition 1 of the paper).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import z3

__all__ = [
    "GRAMMAR",
    "VARIABLES",
    "FOLError",
    "parse",
    "show",
    "decide",
    "equivalent",
    "decide_reading_sets",
    "contains_equivalent",
    "vocabulary",
    "is_valid_reading",
    "signature",
    "directed_edits",
]

#: Grammar used to constrain decoding (GBNF). ``pred`` and ``const`` are replaced by
#: the names of each story's vocabulary in :func:`logicverifier.prompts.translator_grammar`.
GRAMMAR = r'''
formula ::= unary (" " binop " " unary)*
unary ::= "¬" unary | quant | atom | "(" formula ")"
quant ::= ("∀" | "∃") var " " unary
atom ::= pred "(" term (", " term)* ")"
pred ::= [A-Z] [a-zA-Z0-9]*
term ::= var | const
var ::= [xyzuvw]
const ::= [a-z] [a-zA-Z0-9_]+ | [0-9]+ ("." [0-9]+)?
binop ::= "∧" | "∨" | "→" | "↔" | "⊕"
'''

VARIABLES = frozenset("xyzuvw")

# Binary connectives from loosest to tightest binding. → and ↔ associate to the right.
PRECEDENCE = {"↔": 1, "→": 2, "⊕": 3, "∨": 4, "∧": 5}
_RIGHT_ASSOCIATIVE = {"→", "↔"}

_IDENT = r"[^\W\d][\w'\-]*(?:\.\d+\w*)?"
_NUMBER = r"\d+(?:\.\d+)?\w*"
_TOKEN = re.compile(r"\s*(∀|∃|¬|∧|∨|⊕|→|↔|≠|=|\(|\)|,|" + _IDENT + "|" + _NUMBER + ")")
_ATOM = re.compile(r"([A-Z][A-Za-z0-9_]*)\(([^()]*)\)")
_DEFAULT_TIMEOUT_MS = 5000


class FOLError(ValueError):
    """A formula that cannot be read."""


# ----------------------------------------------------------------------------- parsing

def _tokenize(text: str) -> list[str]:
    text = text.replace("->", "→").replace("<->", "↔").replace("¬ ", "¬").strip()
    pos, tokens = 0, []
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if not match or match.end() == pos:
            if text[pos].isspace():
                pos += 1
                continue
            raise FOLError(f"unexpected character {text[pos]!r} in {text!r}")
        tokens.append(match.group(1))
        pos = match.end()
    return tokens


class _Parser:
    """Recursive descent over the token list. Formulas become nested tuples:

    ``("¬", f)``, ``("∀", var, f)``, ``(op, left, right)``, ``("pred", name, args)``,
    ``("=", term, term)``, ``("prop", name)``. A term is a string or ``("fn", name, args)``.
    """

    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.i = 0

    def peek(self) -> str | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def eat(self, expected: str | None = None) -> str:
        token = self.peek()
        if token is None or (expected is not None and token != expected):
            raise FOLError(f"expected {expected!r}, found {token!r}")
        self.i += 1
        return token

    def formula(self, min_precedence: int = 1) -> tuple:
        left = self.unary()
        while self.peek() in PRECEDENCE and PRECEDENCE[self.peek()] >= min_precedence:
            op = self.eat()
            precedence = PRECEDENCE[op]
            right = self.formula(precedence if op in _RIGHT_ASSOCIATIVE else precedence + 1)
            left = (op, left, right)
        return left

    def unary(self) -> tuple:
        token = self.peek()
        if token == "¬":
            self.eat()
            return ("¬", self.unary())
        if token in ("∀", "∃"):
            self.eat()
            var = self.eat()
            if not re.fullmatch(r"[a-z]", var):
                raise FOLError(f"bad variable {var!r}")
            return (token, var, self.unary())
        if token == "(":
            self.eat()
            inner = self.formula()
            self.eat(")")
            return inner
        return self.atom()

    def arguments(self) -> tuple:
        self.eat("(")
        args = [self.term()]
        while self.peek() == ",":
            self.eat()
            args.append(self.term())
        self.eat(")")
        return tuple(args)

    def term(self):
        token = self.eat()
        if not (re.fullmatch(_IDENT, token) or re.fullmatch(_NUMBER, token)):
            raise FOLError(f"bad term {token!r}")
        if self.peek() == "(":
            return ("fn", token, self.arguments())
        return token

    def atom(self) -> tuple:
        if self.peek() is None:
            raise FOLError("unexpected end of formula")
        if self.i + 1 < len(self.tokens) and self.tokens[self.i + 1] in ("=", "≠"):
            left, op, right = self.term(), self.eat(), self.term()
            return ("=", left, right) if op == "=" else ("¬", ("=", left, right))
        name = self.eat()
        if not re.fullmatch(_IDENT, name):
            raise FOLError(f"bad predicate {name!r}")
        if self.peek() != "(":
            return ("prop", name)
        args = self.arguments()
        if self.peek() in ("=", "≠"):
            op, right = self.eat(), self.term()
            equality = ("=", ("fn", name, args), right)
            return equality if op == "=" else ("¬", equality)
        return ("pred", name, args)


def parse(text: str) -> tuple:
    """Parse a formula into a syntax tree. Raises :class:`FOLError`."""
    parser = _Parser(_tokenize(text))
    tree = parser.formula()
    if parser.peek() is not None:
        raise FOLError(f"unexpected {parser.peek()!r} in {text!r}")
    return tree


def _show_term(term) -> str:
    if isinstance(term, str):
        return term
    return f"{term[1]}({', '.join(_show_term(a) for a in term[2])})"


def show(tree: tuple) -> str:
    """Write a syntax tree back as a formula, with every binary connective in brackets."""
    kind = tree[0]
    if kind == "¬":
        return "¬" + show(tree[1])
    if kind in ("∀", "∃"):
        return f"{kind}{tree[1]} {show(tree[2])}"
    if kind in PRECEDENCE:
        return f"({show(tree[1])} {kind} {show(tree[2])})"
    if kind == "pred":
        return f"{tree[1]}({', '.join(_show_term(a) for a in tree[2])})"
    if kind == "=":
        return f"{_show_term(tree[1])} = {_show_term(tree[2])}"
    if kind == "prop":
        return tree[1]
    raise FOLError(f"unknown node {kind!r}")


# ----------------------------------------------------------------------------- Z3

_CONNECTIVES = {"∧": z3.And, "∨": z3.Or, "⊕": z3.Xor, "→": z3.Implies}


class _Z3Builder:
    """Translates syntax trees into Z3 terms. One builder per problem, so that
    premises and claim share their predicates and constants."""

    def __init__(self):
        self.sort = z3.DeclareSort("Obj")
        self.predicates: dict = {}
        self.constants: dict = {}
        self.propositions: dict = {}
        self.functions: dict = {}

    def term(self, term, env: dict):
        if isinstance(term, tuple):
            _, name, args = term
            key = (name, len(args))
            if key not in self.functions:
                self.functions[key] = z3.Function(f"f_{name}_{len(args)}", *([self.sort] * (len(args) + 1)))
            return self.functions[key](*[self.term(a, env) for a in args])
        if term in env:
            return env[term]
        if term not in self.constants:
            self.constants[term] = z3.Const(f"c_{term}", self.sort)
        return self.constants[term]

    def build(self, tree: tuple, env: dict | None = None):
        env = env or {}
        kind = tree[0]
        if kind == "¬":
            return z3.Not(self.build(tree[1], env))
        if kind in ("∀", "∃"):
            var = z3.Const(f"v_{tree[1]}_{len(env)}", self.sort)
            body = self.build(tree[2], {**env, tree[1]: var})
            return z3.ForAll([var], body) if kind == "∀" else z3.Exists([var], body)
        if kind in PRECEDENCE:
            left, right = self.build(tree[1], env), self.build(tree[2], env)
            if kind == "↔":
                return left == right
            return _CONNECTIVES[kind](left, right)
        if kind == "=":
            return self.term(tree[1], env) == self.term(tree[2], env)
        if kind == "prop":
            if tree[1] not in self.propositions:
                self.propositions[tree[1]] = z3.Bool(f"p_{tree[1]}")
            return self.propositions[tree[1]]
        if kind == "pred":
            name, args = tree[1], tree[2]
            key = (name, len(args))
            if key not in self.predicates:
                self.predicates[key] = z3.Function(f"P_{name}_{len(args)}", *([self.sort] * len(args)), z3.BoolSort())
            return self.predicates[key](*[self.term(a, env) for a in args])
        raise FOLError(f"unknown node {kind!r}")

    def formula(self, text: str):
        return self.build(parse(text))


def _check(constraints: Iterable, timeout_ms: int):
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    for constraint in constraints:
        solver.add(constraint)
    return solver.check()


def decide(premises: Sequence[str], claim: str, timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> str:
    """Decide a single reading of a story.

    Returns ``"True"`` if the premises entail the claim, ``"False"`` if they entail its
    negation, ``"Uncertain"`` otherwise, and ``"Error"`` if a formula cannot be read or
    the premises are inconsistent.
    """
    try:
        builder = _Z3Builder()
        P = [builder.formula(p) for p in premises]
        C = builder.formula(claim)
    except (FOLError, z3.Z3Exception, RecursionError):
        return "Error"
    if _check(P, timeout_ms) == z3.unsat:
        return "Error"
    if _check(P + [z3.Not(C)], timeout_ms) == z3.unsat:
        return "True"
    if _check(P + [C], timeout_ms) == z3.unsat:
        return "False"
    return "Uncertain"


def equivalent(a: str, b: str, timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> bool | None:
    """Whether two formulas are logically equivalent. ``None`` if unreadable or undecided."""
    try:
        builder = _Z3Builder()
        A, B = builder.formula(a), builder.formula(b)
    except (FOLError, z3.Z3Exception, RecursionError):
        return None
    result = _check([z3.Not(A == B)], timeout_ms)
    if result == z3.unsat:
        return True
    if result == z3.sat:
        return False
    return None


def decide_reading_sets(premise_sets: Sequence[Sequence[str]], claim_set: Sequence[str],
                        timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> str:
    """Decision over reading sets (Definition 1).

    Each source sentence enters as the disjunction of its readings, and sentences with
    no reading are left out. The claim is approved only if it follows under every one of
    its readings. Returns ``"True"`` (PASS), ``"False"`` (FAIL) or ``"Uncertain"`` (REVIEW),
    the last also when the claim has no reading or the premises are inconsistent.
    """
    builder = _Z3Builder()

    def build_all(readings):
        built = []
        for reading in readings:
            try:
                built.append(builder.formula(reading))
            except (FOLError, z3.Z3Exception, RecursionError):
                pass
        return built

    try:
        premises = []
        for readings in premise_sets:
            alternatives = build_all(readings)
            if alternatives:
                premises.append(z3.Or(*alternatives) if len(alternatives) > 1 else alternatives[0])
        claims = build_all(claim_set)
    except RecursionError:
        return "Uncertain"
    if not claims or _check(premises, timeout_ms) != z3.sat:
        return "Uncertain"
    if all(_check(premises + [z3.Not(c)], timeout_ms) == z3.unsat for c in claims):
        return "True"
    if all(_check(premises + [c], timeout_ms) == z3.unsat for c in claims):
        return "False"
    return "Uncertain"


def contains_equivalent(readings: Iterable[str], reference: str, timeout_ms: int = 3000) -> bool:
    """Whether some reading is logically equivalent to the reference formula."""
    return any(equivalent(r, reference, timeout_ms) is True for r in readings)


# ----------------------------------------------------------------------------- vocabulary

def vocabulary(formulas: Iterable[str], decimals: bool = True) -> dict:
    """Predicates (with their number of arguments) and constants used by some formulas.

    ``decimals`` also accepts constants such as ``y42.3billion``. The training mixtures
    were built with ``decimals=False``, which is kept for reproduction.
    """
    constant = r"[a-z][A-Za-z0-9_]*(?:\.\d+\w*)?" if decimals else r"[a-z][A-Za-z0-9_]*"
    predicates: dict[str, int] = {}
    constants: set[str] = set()
    for formula in formulas:
        for name, args in _ATOM.findall(formula):
            terms = [t.strip() for t in args.split(",") if t.strip()]
            predicates.setdefault(name, len(terms))
            constants.update(t for t in terms if re.fullmatch(constant, t) and t not in VARIABLES)
    return {"preds": predicates, "consts": sorted(constants)}


def is_valid_reading(formula: str, vocab: dict) -> bool:
    """A reading is kept only if it parses and uses the story's vocabulary with the right arities."""
    if not formula:
        return False
    try:
        parse(formula)
    except (FOLError, RecursionError):
        return False
    for name, args in _ATOM.findall(formula):
        terms = [t.strip() for t in args.split(",") if t.strip()]
        if vocab["preds"].get(name) != len(terms):
            return False
        for term in terms:
            if re.fullmatch(r"[a-z][A-Za-z0-9_]*", term) and term not in VARIABLES and term not in vocab["consts"]:
                return False
    return True


def signature(formula: str) -> dict | None:
    """Shape of a formula that does not depend on names: negations, quantifiers in
    order, connectives, arities and numbers. Used to compare translations of the same
    sentence in different languages."""
    try:
        tree = parse(formula)
    except FOLError:
        return None
    negations, quantifiers, connectives, arities, numbers = 0, [], {}, [], []

    def walk(node):
        nonlocal negations
        kind = node[0]
        if kind == "¬":
            negations += 1
            walk(node[1])
        elif kind in ("∀", "∃"):
            quantifiers.append(kind)
            walk(node[2])
        elif kind in PRECEDENCE:
            connectives[kind] = connectives.get(kind, 0) + 1
            walk(node[1])
            walk(node[2])
        elif kind == "pred":
            arities.append(len(node[2]))
            numbers.extend(a for a in node[2] if isinstance(a, str) and re.fullmatch(r"\d+(\.\d+)?", a))
        elif kind == "=":
            arities.append(2)

    walk(tree)
    return {"neg": negations, "quants": "".join(quantifiers), "conn": tuple(sorted(connectives.items())),
            "arity": tuple(sorted(arities)), "nums": tuple(sorted(numbers))}


# ----------------------------------------------------------------------------- directed edits

def _edits(tree: tuple, ops: str):
    kind = tree[0]
    if kind == "¬":
        for edited in _edits(tree[1], ops):
            yield ("¬", edited)
    elif kind in ("∀", "∃"):
        for edited in _edits(tree[2], ops):
            yield (kind, tree[1], edited)
    elif kind in PRECEDENCE:
        if "xor" in ops and kind in ("⊕", "∨"):
            yield ("∨" if kind == "⊕" else "⊕", tree[1], tree[2])
        for edited in _edits(tree[1], ops):
            yield (kind, edited, tree[2])
        for edited in _edits(tree[2], ops):
            yield (kind, tree[1], edited)
    elif kind == "pred" and "args" in ops and len(tree[2]) == 2 and tree[2][0] != tree[2][1]:
        yield ("pred", tree[1], (tree[2][1], tree[2][0]))


def directed_edits(formula: str, ops: str) -> list[str]:
    """Every formula one edit away. ``ops`` names the edits: ``"xor"`` exchanges ⊕ and ∨,
    ``"args"`` exchanges the two arguments of a binary predicate, ``"xor+args"`` both."""
    try:
        tree = parse(formula)
    except (FOLError, RecursionError):
        return []
    return [show(edited) for edited in _edits(tree, ops)]
