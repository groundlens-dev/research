# Copyright 2026 Javier Marín
# SPDX-License-Identifier: Apache-2.0
"""Synthetic stories whose labels are correct by construction.

The formulas of a story are written first and its sentences second, from templates, and
Z3 computes the label from the formulas. Two generators feed the training mixtures:

* :func:`domain_stories` (translator v3) covers ten domains in English, with the
  constructions where translators fail on real text: either/or, both-or-neither,
  neither/nor, only, unless, some, no, scope of quantifiers and negated facts.
* :func:`vocabulary_stories` (translator v4) uses real vocabularies (EuroVoc or Wikidata,
  FIBO, DPV) in six languages. Predicates are English names, sentences use the concept's
  label in each language, and every story includes true links of the hierarchy.
"""

from __future__ import annotations

import collections
import glob
import json
import random
import re
import subprocess
import time
import urllib.parse
import urllib.request
from collections.abc import Sequence
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from .logic import decide, parse

__all__ = ["DOMAINS", "domain_stories", "fetch_vocabulary_groups", "vocabulary_stories"]

_LANGUAGES = ["en", "es", "fr", "de", "nl", "it"]


# ----------------------------------------------------------------------------- English helpers

def _article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def _capital(text: str) -> str:
    return text[0].upper() + text[1:]


def _display(constant: str) -> str:
    """How a constant is written in a sentence: ``marta`` → ``Marta``."""
    return constant[0].upper() + constant[1:]


def _not_singular(verb: tuple) -> str:
    """Negation of a verb phrase in the third person singular."""
    if verb[1].startswith("is "):
        return "is not " + verb[1][3:]
    if verb[1].startswith("has "):
        return "does not have " + verb[1][4:]
    return "does not " + verb[2]


def _not_plural(verb: tuple) -> str:
    """Negation of a verb phrase in the plural."""
    if verb[2].startswith("are "):
        return "are not " + verb[2][4:]
    return "do not " + verb[2]


def _noun_phrase(label: str) -> str:
    """``debt instrument`` → ``a debt instrument``."""
    label = label if label.isupper() else label[:1].lower() + label[1:]
    return ("an " if label[0].lower() in "aeiou" else "a ") + label


def _bare(label: str) -> str:
    return label if label.isupper() else label[:1].lower() + label[1:]


_QUOTES = {"es": ("«", "»"), "fr": ("« ", " »"), "de": ("„", "“"), "nl": ("'", "'"), "it": ("«", "»")}


def _quote(lang: str, label: str) -> str:
    opening, closing = _QUOTES[lang]
    return f"{opening}{label}{closing}"


# ----------------------------------------------------------------------------- domain stories (v3)
# Each domain: nouns (Name, singular, plural), verb phrases (Name, third person singular,
# base form), relations between two individuals, and constants.
DOMAINS = {
 "banking": {"N": [("RetailCustomer", "retail customer", "retail customers"), ("JointAccount", "joint account", "joint accounts"),
                   ("Borrower", "borrower", "borrowers"), ("Guarantor", "guarantor", "guarantors"), ("Resident", "resident", "residents"),
                   ("CorporateClient", "corporate client", "corporate clients")],
             "V": [("PaysOverdraftFee", "pays an overdraft fee", "pay an overdraft fee"), ("PassesCreditCheck", "passes the credit check", "pass the credit check"),
                   ("HoldsSavingsAccount", "holds a savings account", "hold a savings account"), ("ReceivesStatement", "receives a monthly statement", "receive a monthly statement"),
                   ("DefaultsOnLoan", "defaults on a loan", "default on a loan"), ("UsesOnlineBanking", "uses online banking", "use online banking")],
             "R": [("Guarantees", "guarantees the loan of", "guarantee the loan of"), ("Advises", "advises", "advise")],
             "C": ["marta", "olsen", "kovac", "lindqvist", "acmeltd"]},
 "insurance": {"N": [("Policyholder", "policyholder", "policyholders"), ("Driver", "driver", "drivers"), ("Claimant", "claimant", "claimants"),
                     ("Smoker", "smoker", "smokers"), ("Homeowner", "homeowner", "homeowners"), ("Broker", "broker", "brokers")],
               "V": [("PaysPremium", "pays the annual premium", "pay the annual premium"), ("FilesClaim", "files a claim", "file a claim"),
                     ("HasNoClaimsBonus", "has a no-claims bonus", "have a no-claims bonus"), ("ReceivesPayout", "receives a payout", "receive a payout"),
                     ("NeedsMedicalExam", "needs a medical exam", "need a medical exam"), ("RenewsPolicy", "renews the policy", "renew the policy")],
               "R": [("Insures", "insures", "insure"), ("RepresentsClient", "represents", "represent")],
               "C": ["ines", "harlow", "petrov", "nakamura", "safehome"]},
 "health": {"N": [("Patient", "patient", "patients"), ("Nurse", "nurse", "nurses"), ("Outpatient", "outpatient", "outpatients"),
                  ("Pharmacist", "pharmacist", "pharmacists"), ("Minor", "minor", "minors"), ("Donor", "donor", "donors")],
            "V": [("TakesAnticoagulants", "takes anticoagulants", "take anticoagulants"), ("NeedsReferral", "needs a referral", "need a referral"),
                  ("GetsVaccinated", "gets vaccinated", "get vaccinated"), ("HasAllergy", "has an allergy", "have an allergy"),
                  ("SignsConsent", "signs a consent form", "sign a consent form"), ("PrescribesOpioids", "prescribes opioids", "prescribe opioids")],
            "R": [("Treats", "treats", "treat"), ("Supervises", "supervises", "supervise")],
            "C": ["amira", "boris", "chen", "dolores", "elias"]},
 "employment": {"N": [("Employee", "employee", "employees"), ("Contractor", "contractor", "contractors"), ("Manager", "manager", "managers"),
                      ("Intern", "intern", "interns"), ("PartTimer", "part-time worker", "part-time workers"), ("Apprentice", "apprentice", "apprentices")],
                "V": [("WorksRemotely", "works remotely", "work remotely"), ("ReceivesBonus", "receives a bonus", "receive a bonus"),
                      ("SignsNda", "signs a non-disclosure agreement", "sign a non-disclosure agreement"), ("TakesParentalLeave", "takes parental leave", "take parental leave"),
                      ("AttendsTraining", "attends the safety training", "attend the safety training"), ("OwnsShares", "owns company shares", "own company shares")],
                "R": [("ReportsTo", "reports to", "report to"), ("Mentors", "mentors", "mentor")],
                "C": ["farid", "greta", "hugo", "isabel", "jonas"]},
 "procurement": {"N": [("Supplier", "supplier", "suppliers"), ("Tender", "tender", "tenders"), ("Subcontractor", "subcontractor", "subcontractors"),
                       ("SmallBusiness", "small business", "small businesses"), ("ForeignBidder", "foreign bidder", "foreign bidders"), ("Buyer", "public buyer", "public buyers")],
                 "V": [("PaysVat", "pays VAT", "pay VAT"), ("SubmitsBid", "submits a bid", "submit a bid"), ("IsCertified", "is certified", "are certified"),
                       ("MeetsDeadline", "meets the deadline", "meet the deadline"), ("ProvidesGuarantee", "provides a bank guarantee", "provide a bank guarantee"),
                       ("WinsContract", "wins the contract", "win the contract")],
                 "R": [("Supplies", "supplies", "supply"), ("Evaluates", "evaluates", "evaluate")],
                 "C": ["norbuild", "valdez", "kimura", "ostrava", "luma"]},
 "data_protection": {"N": [("Controller", "data controller", "data controllers"), ("Processor", "data processor", "data processors"), ("DataSubject", "data subject", "data subjects"),
                           ("PublicAuthority", "public authority", "public authorities"), ("Startup", "startup", "startups"), ("Hospital", "hospital", "hospitals")],
                     "V": [("AppointsDpo", "appoints a data protection officer", "appoint a data protection officer"), ("ReportsBreach", "reports the breach", "report the breach"),
                           ("TransfersDataAbroad", "transfers data abroad", "transfer data abroad"), ("KeepsRecords", "keeps processing records", "keep processing records"),
                           ("UsesEncryption", "uses encryption", "use encryption"), ("RequestsErasure", "requests erasure", "request erasure")],
                     "R": [("ProcessesFor", "processes data for", "process data for"), ("Audits", "audits", "audit")],
                     "C": ["medisoft", "anna", "brightcloud", "carlos", "dataco"]},
 "tax": {"N": [("Taxpayer", "taxpayer", "taxpayers"), ("Pensioner", "pensioner", "pensioners"), ("Freelancer", "freelancer", "freelancers"),
               ("Charity", "charity", "charities"), ("NonResident", "non-resident", "non-residents"), ("Landlord", "landlord", "landlords")],
         "V": [("FilesReturn", "files a tax return", "file a tax return"), ("ClaimsDeduction", "claims the deduction", "claim the deduction"),
               ("PaysSurcharge", "pays a surcharge", "pay a surcharge"), ("IsExempt", "is exempt", "are exempt"),
               ("ReceivesRefund", "receives a refund", "receive a refund"), ("DeclaresRentalIncome", "declares rental income", "declare rental income")],
         "R": [("Employs", "employs", "employ"), ("Represents", "represents", "represent")],
         "C": ["pablo", "quentin", "rosa", "sven", "helpfund"]},
 "transport": {"N": [("Carrier", "carrier", "carriers"), ("Truck", "truck", "trucks"), ("Passenger", "passenger", "passengers"),
                     ("Pilot", "pilot", "pilots"), ("Ferry", "ferry", "ferries"), ("Cyclist", "cyclist", "cyclists")],
               "V": [("CarriesHazardousGoods", "carries hazardous goods", "carry hazardous goods"), ("PassesInspection", "passes the inspection", "pass the inspection"),
                     ("ExceedsWeightLimit", "exceeds the weight limit", "exceed the weight limit"), ("HoldsLicence", "holds a valid licence", "hold a valid licence"),
                     ("UsesTollRoad", "uses the toll road", "use the toll road"), ("ReceivesCompensation", "receives compensation", "receive compensation")],
               "R": [("Operates", "operates", "operate"), ("Transports", "transports", "transport")],
               "C": ["trans4", "ulla", "victor", "wanda", "bluesea"]},
 "energy": {"N": [("Producer", "producer", "producers"), ("Battery", "battery", "batteries"), ("WindFarm", "wind farm", "wind farms"),
                  ("Household", "household", "households"), ("Retailer", "retailer", "retailers"), ("Aggregator", "aggregator", "aggregators")],
            "V": [("SellsToGrid", "sells to the grid", "sell to the grid"), ("ReceivesSubsidy", "receives a subsidy", "receive a subsidy"),
                  ("ProvidesReserve", "provides reserve capacity", "provide reserve capacity"), ("HasSmartMeter", "has a smart meter", "have a smart meter"),
                  ("PaysCapacityFee", "pays the capacity fee", "pay the capacity fee"), ("IsCurtailed", "is curtailed", "are curtailed")],
            "R": [("Supplies", "supplies power to", "supply power to"), ("Owns", "owns", "own")],
            "C": ["gridco", "xavier", "yolanda", "zephyrwind", "voltix"]},
 "education": {"N": [("Student", "student", "students"), ("Teacher", "teacher", "teachers"), ("Graduate", "graduate", "graduates"),
                     ("ExchangeStudent", "exchange student", "exchange students"), ("Researcher", "researcher", "researchers"), ("Scholar", "scholarship holder", "scholarship holders")],
               "V": [("PaysTuition", "pays tuition", "pay tuition"), ("PassesExam", "passes the final exam", "pass the final exam"),
                     ("AttendsLectures", "attends the lectures", "attend the lectures"), ("PublishesPaper", "publishes a paper", "publish a paper"),
                     ("LivesOnCampus", "lives on campus", "live on campus"), ("ReceivesGrant", "receives a grant", "receive a grant")],
               "R": [("Supervises", "supervises", "supervise"), ("Teaches", "teaches", "teach")],
               "C": ["aiko", "bruno", "clara", "dmitri", "esther"]},
}

# General constructions: (formula, phrasings) over nouns n1, n2 and verb phrases v1, v2.
_GENERAL = [
 (lambda n1, n2, v1, v2: f"∀x ({n1[0]}(x) → {n2[0]}(x))", [lambda n1, n2, v1, v2: f"All {n1[2]} are {n2[2]}.", lambda n1, n2, v1, v2: f"Every {n1[1]} is {_article(n2[1])} {n2[1]}.", lambda n1, n2, v1, v2: f"Each {n1[1]} is also {_article(n2[1])} {n2[1]}."]),
 (lambda n1, n2, v1, v2: f"∀x ({n1[0]}(x) → {v1[0]}(x))", [lambda n1, n2, v1, v2: f"All {n1[2]} {v1[2]}.", lambda n1, n2, v1, v2: f"Every {n1[1]} {v1[1]}.", lambda n1, n2, v1, v2: f"Any {n1[1]} {v1[1]}."]),
 (lambda n1, n2, v1, v2: f"∀x ({n1[0]}(x) → ¬{v1[0]}(x))", [lambda n1, n2, v1, v2: f"No {n1[1]} {v1[1]}.", lambda n1, n2, v1, v2: f"{_capital(n1[2])} {_not_plural(v1)}.", lambda n1, n2, v1, v2: f"None of the {n1[2]} {v1[2]}."]),
 (lambda n1, n2, v1, v2: f"¬∃x ({n1[0]}(x) ∧ {n2[0]}(x))", [lambda n1, n2, v1, v2: f"No {n1[1]} is {_article(n2[1])} {n2[1]}.", lambda n1, n2, v1, v2: f"There is no {n1[1]} that is also {_article(n2[1])} {n2[1]}."]),
 (lambda n1, n2, v1, v2: f"∃x ({n1[0]}(x) ∧ {v1[0]}(x))", [lambda n1, n2, v1, v2: f"Some {n1[2]} {v1[2]}.", lambda n1, n2, v1, v2: f"At least one {n1[1]} {v1[1]}.", lambda n1, n2, v1, v2: f"There is {_article(n1[1])} {n1[1]} who {v1[1]}."]),
 (lambda n1, n2, v1, v2: f"∀x (({n1[0]}(x) ∧ {n2[0]}(x)) → {v1[0]}(x))", [lambda n1, n2, v1, v2: f"Every {n1[1]} who is also {_article(n2[1])} {n2[1]} {v1[1]}.", lambda n1, n2, v1, v2: f"All {n1[2]} that are {n2[2]} {v1[2]}."]),
 (lambda n1, n2, v1, v2: f"∀x ({v1[0]}(x) → {n1[0]}(x))", [lambda n1, n2, v1, v2: f"Only {n1[2]} {v1[2]}.", lambda n1, n2, v1, v2: f"Whoever {v1[1]} is {_article(n1[1])} {n1[1]}."]),
 (lambda n1, n2, v1, v2: f"∀x (({n1[0]}(x) ∧ ¬{v2[0]}(x)) → {v1[0]}(x))", [lambda n1, n2, v1, v2: f"All {n1[2]} {v1[2]} unless they {v2[2]}.", lambda n1, n2, v1, v2: f"Every {n1[1]} that {_not_singular(v2)} {v1[1]}."]),
 (lambda n1, n2, v1, v2: f"∀x ({n1[0]}(x) → ({v1[0]}(x) ⊕ {v2[0]}(x)))", [lambda n1, n2, v1, v2: f"Every {n1[1]} either {v1[1]} or {v2[1]}, but not both.", lambda n1, n2, v1, v2: f"Each {n1[1]} {v1[1]} or {v2[1]}, but never both."]),
 (lambda n1, n2, v1, v2: f"∀x ({n1[0]}(x) → ({v1[0]}(x) ∨ {v2[0]}(x)))", [lambda n1, n2, v1, v2: f"Every {n1[1]} {v1[1]} or {v2[1]}.", lambda n1, n2, v1, v2: f"All {n1[2]} {v1[2]} or {v2[2]}, or both."]),
 (lambda n1, n2, v1, v2: f"∀x ({v1[0]}(x) → {v2[0]}(x))", [lambda n1, n2, v1, v2: f"Anyone who {v1[1]} {v2[1]}.", lambda n1, n2, v1, v2: f"If someone {v1[1]}, then they {v2[2]}."]),
]

# Facts about an individual c.
_FACTS = [
 (lambda c, n1, v1, v2: f"{n1[0]}({c})", [lambda c, n1, v1, v2: f"{_display(c)} is {_article(n1[1])} {n1[1]}."]),
 (lambda c, n1, v1, v2: f"¬{n1[0]}({c})", [lambda c, n1, v1, v2: f"{_display(c)} is not {_article(n1[1])} {n1[1]}."]),
 (lambda c, n1, v1, v2: f"{v1[0]}({c})", [lambda c, n1, v1, v2: f"{_display(c)} {v1[1]}."]),
 (lambda c, n1, v1, v2: f"¬{v1[0]}({c})", [lambda c, n1, v1, v2: f"{_display(c)} {_not_singular(v1)}.", lambda c, n1, v1, v2: f"It is not the case that {_display(c)} {v1[1]}."]),
 (lambda c, n1, v1, v2: f"¬{v1[0]}({c}) ∧ ¬{v2[0]}({c})", [lambda c, n1, v1, v2: f"{_display(c)} neither {v1[1]} nor {v2[1]}."]),
 (lambda c, n1, v1, v2: f"({v1[0]}({c}) ∧ {v2[0]}({c})) ⊕ (¬{v1[0]}({c}) ∧ ¬{v2[0]}({c}))", [lambda c, n1, v1, v2: f"{_display(c)} either both {v1[1]} and {v2[1]}, or neither {v1[1]} nor {v2[1]}."]),
 (lambda c, n1, v1, v2: f"{v1[0]}({c}) ⊕ {v2[0]}({c})", [lambda c, n1, v1, v2: f"{_display(c)} either {v1[1]} or {v2[1]}, but not both."]),
 (lambda c, n1, v1, v2: f"{n1[0]}({c}) ∧ {v1[0]}({c})", [lambda c, n1, v1, v2: f"{_display(c)} is {_article(n1[1])} {n1[1]} who {v1[1]}."]),
]

def _conclusion_options(constants, nouns, verbs):
    options = []
    for c in constants:
        for n in nouns:
            options += [(f"{n[0]}({c})", f"{_display(c)} is {_article(n[1])} {n[1]}."),
                        (f"¬{n[0]}({c})", f"{_display(c)} is not {_article(n[1])} {n[1]}.")]
        for v in verbs:
            options += [(f"{v[0]}({c})", f"{_display(c)} {v[1]}."), (f"¬{v[0]}({c})", f"{_display(c)} {_not_singular(v)}.")]
    return options


def _domain_story(seed: int) -> dict | None:
    rng = random.Random(seed)
    domain = rng.choice(sorted(DOMAINS))
    d = DOMAINS[domain]
    nouns, verbs = rng.sample(d["N"], 3), rng.sample(d["V"], 4)
    constants = rng.sample(d["C"], rng.choice([1, 1, 2]))
    premises = []
    for _ in range(rng.randint(2, 4)):
        formula, phrasings = rng.choice(_GENERAL)
        n1, n2 = rng.sample(nouns, 2)
        v1, v2 = rng.sample(verbs, 2)
        premises.append((formula(n1, n2, v1, v2), rng.choice(phrasings)(n1, n2, v1, v2)))
    for c in constants:
        for _ in range(rng.randint(1, 2)):
            formula, phrasings = rng.choice(_FACTS)
            n1 = rng.choice(nouns)
            v1, v2 = rng.sample(verbs, 2)
            premises.append((formula(c, n1, v1, v2), rng.choice(phrasings)(c, n1, v1, v2)))
    if rng.random() < 0.3 and len(constants) == 2:
        relation = rng.choice(d["R"])
        premises.append((f"{relation[0]}({constants[0]}, {constants[1]})",
                         f"{_display(constants[0])} {relation[1]} {_display(constants[1])}."))
    rng.shuffle(premises)
    formulas = [p[0] for p in premises]
    if decide(formulas, formulas[0]) == "Error":
        return None
    options = _conclusion_options(constants, nouns, verbs)
    rng.shuffle(options)
    by_label = collections.defaultdict(list)
    for formula, text in options[:16]:
        label = decide(formulas, formula)
        if label in ("True", "False", "Uncertain") and formula not in formulas:
            by_label[label].append((formula, text))
    if not by_label:
        return None
    wanted = rng.choice(["True", "False", "Uncertain"])
    label = wanted if by_label.get(wanted) else rng.choice(sorted(by_label))
    formula, text = rng.choice(by_label[label])
    return {"domain": domain, "premises": [p[1] for p in premises], "premises-FOL": formulas, "conclusion": text,
            "conclusion-FOL": formula, "label": label,
            "distractors": [x[0] for x in rng.sample(d["N"] + d["V"], 3)]}


def domain_stories(n: int = 4000, seed: int = 0, processes: int | None = None) -> list[dict]:
    """About ``n`` stories over the ten domains, with the three labels balanced."""
    seeds = range(seed * 10**6, seed * 10**6 + int(n * 1.6))
    with Pool(processes) as pool:
        stories = [s for s in pool.map(_domain_story, seeds, chunksize=50) if s]
    by_label = collections.defaultdict(list)
    for story in stories:
        by_label[story["label"]].append(story)
    per_label = min(min(len(v) for v in by_label.values()), n // 3)
    selected = [s for label in sorted(by_label) for s in by_label[label][:per_label]]
    random.Random(seed).shuffle(selected)
    for story in selected:
        for formula in story["premises-FOL"] + [story["conclusion-FOL"]]:
            parse(formula)
    return selected


# ----------------------------------------------------------------------------- vocabulary groups (v4)

_USER_AGENT = {"User-Agent": "GroundLensResearch/0.1 (vocabulary for synthetic training data)"}
_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
_SKOS = "http://www.w3.org/2004/02/skos/core#"


def _camel(label: str) -> str | None:
    words = re.findall(r"[A-Za-z0-9]+", label.replace("'", ""))
    name = "".join(w[:1].upper() + w[1:] for w in words)
    return name if re.fullmatch(r"[A-Z][A-Za-z0-9]{2,40}", name or "") else None


def _clean(label: str) -> str | None:
    label = re.sub(r"\s+", " ", label).strip()
    return label if 2 < len(label) <= 60 and not re.search(r"[\(\)\[\]/;:]", label) else None


def _sparql(endpoint: str, query: str, timeout: int = 120) -> list[dict]:
    url = endpoint + "?" + urllib.parse.urlencode({"query": query, "format": "application/sparql-results+json"})
    request = urllib.request.Request(url, headers={**_USER_AGENT, "Accept": "application/sparql-results+json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["results"]["bindings"]


def _groups(source: str, concepts: dict, parent_of: dict, min_children: int, max_children: int) -> list[dict]:
    """A group is a parent concept with its children, each with an English name and labels."""
    children = collections.defaultdict(list)
    for child, parent in parent_of.items():
        if child in concepts and parent in concepts:
            children[parent].append(child)

    def node(concept):
        labels = {lang: _clean(text) for lang, text in concepts[concept].items() if _clean(text)}
        name = _camel(labels.get("en", "")) if labels.get("en") else None
        return {"name": name, "labels": labels} if name else None

    groups = []
    for parent, kids in children.items():
        parent_node = node(parent)
        nodes = [x for x in (node(c) for c in sorted(kids)) if x]
        names = set()
        nodes = [x for x in nodes if not (x["name"] in names or names.add(x["name"]))
                 and x["name"] != (parent_node or {}).get("name")]
        if parent_node and len(nodes) >= min_children:
            groups.append({"source": source, "parent": parent_node, "children": nodes[:max_children]})
    return groups


def _rdf_groups(source, repository, patterns, label_properties, parent_properties, work_dir, min_children, max_children):
    import rdflib

    directory = Path(work_dir) / source
    if not directory.is_dir():
        subprocess.run(["git", "clone", "-q", "--depth", "1", repository, str(directory)], check=True, timeout=600)
    graph = rdflib.Graph()
    for pattern in patterns:
        for path in glob.glob(str(directory / pattern), recursive=True):
            try:
                graph.parse(path)
            except Exception:  # some files of these repositories do not parse; they are skipped
                pass
    concepts, parent_of = {}, {}
    for prop in label_properties:
        for subject, label in graph.subject_objects(rdflib.URIRef(prop)):
            if isinstance(label, rdflib.Literal) and label.language in (None, "en") and isinstance(subject, rdflib.URIRef):
                concepts.setdefault(str(subject), {})["en"] = str(label)
    for prop in parent_properties:
        for subject, parent in graph.subject_objects(rdflib.URIRef(prop)):
            if isinstance(parent, rdflib.URIRef) and str(subject) in concepts:
                parent_of.setdefault(str(subject), str(parent))
    return concepts, _groups(source, concepts, parent_of, min_children, max_children)


def fetch_vocabulary_groups(work_dir: str | Path, min_children: int = 4, max_children: int = 12) -> tuple[list[dict], dict]:
    """Concepts grouped under their parent, from EuroVoc (or Wikidata if EuroVoc does not
    answer), FIBO and DPV. A source that fails is skipped and named in the report.
    Needs network access, ``git`` and ``rdflib``."""
    groups, report = [], {}
    selected = " ".join(f"?{lang}" for lang in _LANGUAGES)
    try:
        optional = " ".join(f'OPTIONAL {{ ?c skos:prefLabel ?{lang} FILTER(lang(?{lang})="{lang}") }}'
                            for lang in _LANGUAGES if lang != "en")
        concepts, parent_of = {}, {}
        for offset in range(0, 12000, 3000):
            query = f"""PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?c ?b {selected} WHERE {{ ?c skos:inScheme <http://eurovoc.europa.eu/100141> ; skos:prefLabel ?en ; skos:broader ?b .
FILTER(lang(?en)="en") {optional} }} LIMIT 3000 OFFSET {offset}"""
            rows = _sparql("https://publications.europa.eu/webapi/rdf/sparql", query)
            if not rows:
                break
            for row in rows:
                concept = row["c"]["value"]
                concepts[concept] = {lang: row[lang]["value"] for lang in _LANGUAGES if lang in row}
                parent_of[concept] = row["b"]["value"]
            time.sleep(1)
        found = _groups("eurovoc", concepts, parent_of, min_children, max_children)
        report["eurovoc_concepts"], report["eurovoc_groups"] = len(concepts), len(found)
        groups += found
    except Exception as error:
        report["eurovoc_error"] = str(error)[:200]

    if not report.get("eurovoc_groups"):
        try:
            labels = " ".join(f'?c rdfs:label ?{lang} FILTER(lang(?{lang})="{lang}")' for lang in _LANGUAGES)
            query = f"""SELECT ?c ?b {selected} WHERE {{ ?c wdt:P5437 ?ev ; wdt:P279 ?b . ?b wdt:P5437 ?ev2 . {labels} }} LIMIT 20000"""
            concepts, parent_of = {}, {}
            for row in _sparql("https://query.wikidata.org/sparql", query, timeout=180):
                concept = row["c"]["value"]
                concepts[concept] = {lang: row[lang]["value"] for lang in _LANGUAGES if lang in row}
                parent_of[concept] = row["b"]["value"]
            parents = f"""SELECT ?b {selected} WHERE {{ ?c wdt:P5437 ?ev ; wdt:P279 ?b . ?b wdt:P5437 ?ev2 . {labels.replace('?c rdfs', '?b rdfs')} }} LIMIT 20000"""
            for row in _sparql("https://query.wikidata.org/sparql", parents, timeout=180):
                concepts.setdefault(row["b"]["value"], {lang: row[lang]["value"] for lang in _LANGUAGES if lang in row})
            found = _groups("wikidata", concepts, parent_of, min_children, max_children)
            report["wikidata_concepts"], report["wikidata_groups"] = len(concepts), len(found)
            groups += found
        except Exception as error:
            report["wikidata_error"] = str(error)[:200]

    for source, repository, patterns, label_props, parent_props in (
            ("fibo", "https://github.com/edmcouncil/fibo.git", ["**/*.rdf"],
             [_RDFS + "label", _SKOS + "prefLabel"], [_RDFS + "subClassOf"]),
            ("dpv", "https://github.com/w3c/dpv.git", ["dpv/**/*.ttl", "*/modules/*.ttl", "**/dpv-owl.ttl"],
             [_SKOS + "prefLabel", _RDFS + "label"], [_SKOS + "broader", _RDFS + "subClassOf"])):
        try:
            concepts, found = _rdf_groups(source, repository, patterns, label_props, parent_props,
                                          work_dir, min_children, max_children)
            report[f"{source}_concepts"], report[f"{source}_groups"] = len(concepts), len(found)
            groups += found
        except Exception as error:
            report[f"{source}_error"] = str(error)[:200]
    report["groups_total"] = len(groups)
    return groups, report


# ----------------------------------------------------------------------------- vocabulary stories (v4)

_INDIVIDUALS = ["acme", "borealis", "cetra", "dunmore", "elvana", "fjordia", "galenco", "hestia", "ivora", "juniper"]

# General constructions over three classes A, B, C, and facts about an individual c.
_V_GENERAL = {
 "all": lambda A, B, C: f"∀x ({A}(x) → {B}(x))",
 "no": lambda A, B, C: f"∀x ({A}(x) → ¬{B}(x))",
 "some": lambda A, B, C: f"∃x ({A}(x) ∧ {B}(x))",
 "and_all": lambda A, B, C: f"∀x (({A}(x) ∧ {B}(x)) → {C}(x))",
 "only": lambda A, B, C: f"∀x ({B}(x) → {A}(x))",
 "unless": lambda A, B, C: f"∀x (({A}(x) ∧ ¬{C}(x)) → {B}(x))",
 "xor": lambda A, B, C: f"∀x ({A}(x) → ({B}(x) ⊕ {C}(x)))",
 "or": lambda A, B, C: f"∀x ({A}(x) → ({B}(x) ∨ {C}(x)))",
}
_V_FACTS = {
 "is": lambda c, A, B: f"{A}({c})",
 "not": lambda c, A, B: f"¬{A}({c})",
 "neither": lambda c, A, B: f"¬{A}({c}) ∧ ¬{B}({c})",
 "both_or_neither": lambda c, A, B: f"({A}({c}) ∧ {B}({c})) ⊕ (¬{A}({c}) ∧ ¬{B}({c}))",
 "xor": lambda c, A, B: f"{A}({c}) ⊕ {B}({c})",
 "both": lambda c, A, B: f"{A}({c}) ∧ {B}({c})",
}

# Sentence templates per language. General keys take labels (a, b, c); fact keys (f_*) take (c, a, b).
# Outside English the templates speak of membership in a category, which is grammatical for any noun phrase.
_V_TEMPLATES = {
 "en": {
  "all": [lambda a, b, c: f"Every {_bare(a)} is {_noun_phrase(b)}.", lambda a, b, c: f"Anything that is {_noun_phrase(a)} is also {_noun_phrase(b)}."],
  "no": [lambda a, b, c: f"No {_bare(a)} is {_noun_phrase(b)}.", lambda a, b, c: f"Nothing that is {_noun_phrase(a)} is {_noun_phrase(b)}."],
  "some": [lambda a, b, c: f"At least one {_bare(a)} is {_noun_phrase(b)}.", lambda a, b, c: f"There is {_noun_phrase(a)} that is also {_noun_phrase(b)}."],
  "and_all": [lambda a, b, c: f"Every {_bare(a)} that is also {_noun_phrase(b)} is {_noun_phrase(c)}."],
  "only": [lambda a, b, c: f"Only something that is {_noun_phrase(a)} can be {_noun_phrase(b)}."],
  "unless": [lambda a, b, c: f"Every {_bare(a)} is {_noun_phrase(b)}, unless it is {_noun_phrase(c)}."],
  "xor": [lambda a, b, c: f"Every {_bare(a)} is either {_noun_phrase(b)} or {_noun_phrase(c)}, but not both."],
  "or": [lambda a, b, c: f"Every {_bare(a)} is {_noun_phrase(b)} or {_noun_phrase(c)}."],
  "f_is": [lambda c, a, b: f"{c} is {_noun_phrase(a)}."], "f_not": [lambda c, a, b: f"{c} is not {_noun_phrase(a)}."],
  "f_neither": [lambda c, a, b: f"{c} is neither {_noun_phrase(a)} nor {_noun_phrase(b)}."],
  "f_both_or_neither": [lambda c, a, b: f"{c} is either both {_noun_phrase(a)} and {_noun_phrase(b)}, or neither."],
  "f_xor": [lambda c, a, b: f"{c} is either {_noun_phrase(a)} or {_noun_phrase(b)}, but not both."],
  "f_both": [lambda c, a, b: f"{c} is both {_noun_phrase(a)} and {_noun_phrase(b)}."]},
 "es": {
  "all": [lambda a, b, c: f"Todo lo que pertenece a la categoría {_quote('es',a)} pertenece también a la categoría {_quote('es',b)}.", lambda a, b, c: f"Cualquier elemento de la categoría {_quote('es',a)} es también de la categoría {_quote('es',b)}."],
  "no": [lambda a, b, c: f"Nada que pertenezca a la categoría {_quote('es',a)} pertenece a la categoría {_quote('es',b)}.", lambda a, b, c: f"Ningún elemento de la categoría {_quote('es',a)} es de la categoría {_quote('es',b)}."],
  "some": [lambda a, b, c: f"Al menos un elemento de la categoría {_quote('es',a)} pertenece también a la categoría {_quote('es',b)}."],
  "and_all": [lambda a, b, c: f"Todo lo que pertenece a las categorías {_quote('es',a)} y {_quote('es',b)} pertenece también a la categoría {_quote('es',c)}."],
  "only": [lambda a, b, c: f"Solo los elementos de la categoría {_quote('es',a)} pueden pertenecer a la categoría {_quote('es',b)}."],
  "unless": [lambda a, b, c: f"Todo lo que pertenece a la categoría {_quote('es',a)} pertenece también a la categoría {_quote('es',b)}, salvo que pertenezca a la categoría {_quote('es',c)}."],
  "xor": [lambda a, b, c: f"Todo elemento de la categoría {_quote('es',a)} pertenece o bien a la categoría {_quote('es',b)} o bien a la categoría {_quote('es',c)}, pero no a ambas."],
  "or": [lambda a, b, c: f"Todo elemento de la categoría {_quote('es',a)} pertenece a la categoría {_quote('es',b)} o a la categoría {_quote('es',c)}."],
  "f_is": [lambda c, a, b: f"{c} pertenece a la categoría {_quote('es',a)}."], "f_not": [lambda c, a, b: f"{c} no pertenece a la categoría {_quote('es',a)}."],
  "f_neither": [lambda c, a, b: f"{c} no pertenece ni a la categoría {_quote('es',a)} ni a la categoría {_quote('es',b)}."],
  "f_both_or_neither": [lambda c, a, b: f"{c} pertenece a las dos categorías {_quote('es',a)} y {_quote('es',b)}, o a ninguna de ellas."],
  "f_xor": [lambda c, a, b: f"{c} pertenece o bien a la categoría {_quote('es',a)} o bien a la categoría {_quote('es',b)}, pero no a ambas."],
  "f_both": [lambda c, a, b: f"{c} pertenece a las categorías {_quote('es',a)} y {_quote('es',b)}."]},
 "fr": {
  "all": [lambda a, b, c: f"Tout ce qui relève de la catégorie {_quote('fr',a)} relève aussi de la catégorie {_quote('fr',b)}."],
  "no": [lambda a, b, c: f"Rien de ce qui relève de la catégorie {_quote('fr',a)} ne relève de la catégorie {_quote('fr',b)}."],
  "some": [lambda a, b, c: f"Au moins un élément de la catégorie {_quote('fr',a)} relève aussi de la catégorie {_quote('fr',b)}."],
  "and_all": [lambda a, b, c: f"Tout ce qui relève à la fois des catégories {_quote('fr',a)} et {_quote('fr',b)} relève aussi de la catégorie {_quote('fr',c)}."],
  "only": [lambda a, b, c: f"Seuls les éléments de la catégorie {_quote('fr',a)} peuvent relever de la catégorie {_quote('fr',b)}."],
  "unless": [lambda a, b, c: f"Tout ce qui relève de la catégorie {_quote('fr',a)} relève aussi de la catégorie {_quote('fr',b)}, sauf s'il relève de la catégorie {_quote('fr',c)}."],
  "xor": [lambda a, b, c: f"Tout élément de la catégorie {_quote('fr',a)} relève soit de la catégorie {_quote('fr',b)}, soit de la catégorie {_quote('fr',c)}, mais pas des deux."],
  "or": [lambda a, b, c: f"Tout élément de la catégorie {_quote('fr',a)} relève de la catégorie {_quote('fr',b)} ou de la catégorie {_quote('fr',c)}."],
  "f_is": [lambda c, a, b: f"{c} relève de la catégorie {_quote('fr',a)}."], "f_not": [lambda c, a, b: f"{c} ne relève pas de la catégorie {_quote('fr',a)}."],
  "f_neither": [lambda c, a, b: f"{c} ne relève ni de la catégorie {_quote('fr',a)} ni de la catégorie {_quote('fr',b)}."],
  "f_both_or_neither": [lambda c, a, b: f"{c} relève soit des deux catégories {_quote('fr',a)} et {_quote('fr',b)}, soit d'aucune des deux."],
  "f_xor": [lambda c, a, b: f"{c} relève soit de la catégorie {_quote('fr',a)}, soit de la catégorie {_quote('fr',b)}, mais pas des deux."],
  "f_both": [lambda c, a, b: f"{c} relève à la fois des catégories {_quote('fr',a)} et {_quote('fr',b)}."]},
 "de": {
  "all": [lambda a, b, c: f"Alles, was zur Kategorie {_quote('de',a)} gehört, gehört auch zur Kategorie {_quote('de',b)}."],
  "no": [lambda a, b, c: f"Nichts, was zur Kategorie {_quote('de',a)} gehört, gehört zur Kategorie {_quote('de',b)}."],
  "some": [lambda a, b, c: f"Mindestens ein Element der Kategorie {_quote('de',a)} gehört auch zur Kategorie {_quote('de',b)}."],
  "and_all": [lambda a, b, c: f"Alles, was sowohl zur Kategorie {_quote('de',a)} als auch zur Kategorie {_quote('de',b)} gehört, gehört auch zur Kategorie {_quote('de',c)}."],
  "only": [lambda a, b, c: f"Nur Elemente der Kategorie {_quote('de',a)} können zur Kategorie {_quote('de',b)} gehören."],
  "unless": [lambda a, b, c: f"Alles, was zur Kategorie {_quote('de',a)} gehört, gehört auch zur Kategorie {_quote('de',b)}, es sei denn, es gehört zur Kategorie {_quote('de',c)}."],
  "xor": [lambda a, b, c: f"Jedes Element der Kategorie {_quote('de',a)} gehört entweder zur Kategorie {_quote('de',b)} oder zur Kategorie {_quote('de',c)}, aber nicht zu beiden."],
  "or": [lambda a, b, c: f"Jedes Element der Kategorie {_quote('de',a)} gehört zur Kategorie {_quote('de',b)} oder zur Kategorie {_quote('de',c)}."],
  "f_is": [lambda c, a, b: f"{c} gehört zur Kategorie {_quote('de',a)}."], "f_not": [lambda c, a, b: f"{c} gehört nicht zur Kategorie {_quote('de',a)}."],
  "f_neither": [lambda c, a, b: f"{c} gehört weder zur Kategorie {_quote('de',a)} noch zur Kategorie {_quote('de',b)}."],
  "f_both_or_neither": [lambda c, a, b: f"{c} gehört entweder zu beiden Kategorien {_quote('de',a)} und {_quote('de',b)} oder zu keiner von beiden."],
  "f_xor": [lambda c, a, b: f"{c} gehört entweder zur Kategorie {_quote('de',a)} oder zur Kategorie {_quote('de',b)}, aber nicht zu beiden."],
  "f_both": [lambda c, a, b: f"{c} gehört sowohl zur Kategorie {_quote('de',a)} als auch zur Kategorie {_quote('de',b)}."]},
 "nl": {
  "all": [lambda a, b, c: f"Alles wat tot de categorie {_quote('nl',a)} behoort, behoort ook tot de categorie {_quote('nl',b)}."],
  "no": [lambda a, b, c: f"Niets wat tot de categorie {_quote('nl',a)} behoort, behoort tot de categorie {_quote('nl',b)}."],
  "some": [lambda a, b, c: f"Minstens één element van de categorie {_quote('nl',a)} behoort ook tot de categorie {_quote('nl',b)}."],
  "and_all": [lambda a, b, c: f"Alles wat zowel tot de categorie {_quote('nl',a)} als tot de categorie {_quote('nl',b)} behoort, behoort ook tot de categorie {_quote('nl',c)}."],
  "only": [lambda a, b, c: f"Alleen elementen van de categorie {_quote('nl',a)} kunnen tot de categorie {_quote('nl',b)} behoren."],
  "unless": [lambda a, b, c: f"Alles wat tot de categorie {_quote('nl',a)} behoort, behoort ook tot de categorie {_quote('nl',b)}, tenzij het tot de categorie {_quote('nl',c)} behoort."],
  "xor": [lambda a, b, c: f"Elk element van de categorie {_quote('nl',a)} behoort ofwel tot de categorie {_quote('nl',b)} ofwel tot de categorie {_quote('nl',c)}, maar niet tot beide."],
  "or": [lambda a, b, c: f"Elk element van de categorie {_quote('nl',a)} behoort tot de categorie {_quote('nl',b)} of tot de categorie {_quote('nl',c)}."],
  "f_is": [lambda c, a, b: f"{c} behoort tot de categorie {_quote('nl',a)}."], "f_not": [lambda c, a, b: f"{c} behoort niet tot de categorie {_quote('nl',a)}."],
  "f_neither": [lambda c, a, b: f"{c} behoort noch tot de categorie {_quote('nl',a)} noch tot de categorie {_quote('nl',b)}."],
  "f_both_or_neither": [lambda c, a, b: f"{c} behoort ofwel tot beide categorieën {_quote('nl',a)} en {_quote('nl',b)}, ofwel tot geen van beide."],
  "f_xor": [lambda c, a, b: f"{c} behoort ofwel tot de categorie {_quote('nl',a)} ofwel tot de categorie {_quote('nl',b)}, maar niet tot beide."],
  "f_both": [lambda c, a, b: f"{c} behoort zowel tot de categorie {_quote('nl',a)} als tot de categorie {_quote('nl',b)}."]},
 "it": {
  "all": [lambda a, b, c: f"Tutto ciò che appartiene alla categoria {_quote('it',a)} appartiene anche alla categoria {_quote('it',b)}."],
  "no": [lambda a, b, c: f"Nulla di ciò che appartiene alla categoria {_quote('it',a)} appartiene alla categoria {_quote('it',b)}."],
  "some": [lambda a, b, c: f"Almeno un elemento della categoria {_quote('it',a)} appartiene anche alla categoria {_quote('it',b)}."],
  "and_all": [lambda a, b, c: f"Tutto ciò che appartiene sia alla categoria {_quote('it',a)} sia alla categoria {_quote('it',b)} appartiene anche alla categoria {_quote('it',c)}."],
  "only": [lambda a, b, c: f"Solo gli elementi della categoria {_quote('it',a)} possono appartenere alla categoria {_quote('it',b)}."],
  "unless": [lambda a, b, c: f"Tutto ciò che appartiene alla categoria {_quote('it',a)} appartiene anche alla categoria {_quote('it',b)}, a meno che non appartenga alla categoria {_quote('it',c)}."],
  "xor": [lambda a, b, c: f"Ogni elemento della categoria {_quote('it',a)} appartiene o alla categoria {_quote('it',b)} o alla categoria {_quote('it',c)}, ma non a entrambe."],
  "or": [lambda a, b, c: f"Ogni elemento della categoria {_quote('it',a)} appartiene alla categoria {_quote('it',b)} o alla categoria {_quote('it',c)}."],
  "f_is": [lambda c, a, b: f"{c} appartiene alla categoria {_quote('it',a)}."], "f_not": [lambda c, a, b: f"{c} non appartiene alla categoria {_quote('it',a)}."],
  "f_neither": [lambda c, a, b: f"{c} non appartiene né alla categoria {_quote('it',a)} né alla categoria {_quote('it',b)}."],
  "f_both_or_neither": [lambda c, a, b: f"{c} appartiene a entrambe le categorie {_quote('it',a)} e {_quote('it',b)} oppure a nessuna delle due."],
  "f_xor": [lambda c, a, b: f"{c} appartiene o alla categoria {_quote('it',a)} o alla categoria {_quote('it',b)}, ma non a entrambe."],
  "f_both": [lambda c, a, b: f"{c} appartiene sia alla categoria {_quote('it',a)} sia alla categoria {_quote('it',b)}."]},
}

_FACT_TEMPLATE = {"is": "f_is", "not": "f_not", "neither": "f_neither", "both_or_neither": "f_both_or_neither",
                  "xor": "f_xor", "both": "f_both"}


def _languages_of(group: dict) -> list[str]:
    """Languages in which the parent and at least three children have a label."""
    return [lang for lang in _LANGUAGES
            if lang in group["parent"]["labels"] and sum(lang in c["labels"] for c in group["children"]) >= 3]


def _vocabulary_story(groups: Sequence[dict], seed: int) -> dict | None:
    rng = random.Random(seed)
    group = rng.choice(groups)
    languages = _languages_of(group)
    if not languages:
        return None
    lang = rng.choice(languages)
    templates = _V_TEMPLATES[lang]
    parent = group["parent"]
    children = [c for c in group["children"] if lang in c["labels"]]
    chosen = rng.sample(children, min(len(children), rng.choice([3, 3, 4])))
    classes = chosen + [parent]
    label_of = {c["name"]: c["labels"][lang] for c in classes}
    individuals = rng.sample(_INDIVIDUALS, rng.choice([1, 1, 2]))
    shown = {c: c[:1].upper() + c[1:] for c in individuals}
    premises = []
    for child in rng.sample(chosen, rng.choice([1, 2])):
        premises.append((_V_GENERAL["all"](child["name"], parent["name"], None),
                         rng.choice(templates["all"])(label_of[child["name"]], label_of[parent["name"]], None)))
    for _ in range(rng.randint(1, 3)):
        key = rng.choice(list(_V_GENERAL))
        A, B, C = rng.sample([c["name"] for c in classes], 3)
        premises.append((_V_GENERAL[key](A, B, C), rng.choice(templates[key])(label_of[A], label_of[B], label_of[C])))
    for c in individuals:
        for _ in range(rng.randint(1, 2)):
            key = rng.choice(list(_V_FACTS))
            A, B = rng.sample([x["name"] for x in classes], 2)
            premises.append((_V_FACTS[key](c, A, B),
                             rng.choice(templates[_FACT_TEMPLATE[key]])(shown[c], label_of[A], label_of[B])))
    rng.shuffle(premises)
    formulas = [p[0] for p in premises]
    if decide(formulas, formulas[0]) == "Error":
        return None
    options = []
    for c in individuals:
        for x in classes:
            options.append((f"{x['name']}({c})", templates["f_is"][0](shown[c], label_of[x["name"]], None)))
            options.append((f"¬{x['name']}({c})", templates["f_not"][0](shown[c], label_of[x["name"]], None)))
    rng.shuffle(options)
    by_label = collections.defaultdict(list)
    for formula, text in options[:14]:
        label = decide(formulas, formula)
        if label in ("True", "False", "Uncertain") and formula not in formulas:
            by_label[label].append((formula, text))
    if not by_label:
        return None
    wanted = rng.choice(["True", "False", "Uncertain"])
    label = wanted if by_label.get(wanted) else rng.choice(sorted(by_label))
    formula, text = rng.choice(by_label[label])
    others = [c["name"] for c in group["children"] if c["name"] not in label_of]
    return {"source": group["source"], "lang": lang, "premises": [p[1] for p in premises], "premises-FOL": formulas,
            "conclusion": text, "conclusion-FOL": formula, "label": label,
            "distractors": rng.sample(others, min(3, len(others)))}


def vocabulary_stories(groups: Sequence[dict], n: int = 6000, seed: int = 1, processes: int | None = None) -> list[dict]:
    """About ``n`` stories over the vocabulary groups, shared among the languages, with
    the labels balanced within each language."""
    seeds = range(seed * 10**6, seed * 10**6 + int(n * 1.8))
    with Pool(processes) as pool:
        stories = [s for s in pool.map(partial(_vocabulary_story, list(groups)), seeds, chunksize=50) if s]
    by_cell = collections.defaultdict(list)
    for story in stories:
        by_cell[(story["lang"], story["label"])].append(story)
    languages = sorted({s["lang"] for s in stories})
    per_language = n // max(1, len(languages))
    selected = []
    for lang in languages:
        k = min(min(len(by_cell[(lang, label)]) for label in ("True", "False", "Uncertain")), per_language // 3)
        for label in ("True", "False", "Uncertain"):
            selected += by_cell[(lang, label)][:k]
    random.Random(seed).shuffle(selected)
    for story in selected:
        for formula in story["premises-FOL"] + [story["conclusion-FOL"]]:
            parse(formula)
    return selected
