# Copyright 2026 Javier Marín
# SPDX-License-Identifier: Apache-2.0
"""Everything that runs a model: QLoRA training of the translator, reading generation
with vLLM, pseudo-labels for the training mixtures, the controls outside FOLIO and the
hosted judges.

The heavy libraries (torch, transformers, peft, vllm, anthropic, openai) are imported
inside the functions, so the rest of the package works without them. Translation and
generation need a GPU; the paper used one NVIDIA L4 with 24 GB.
"""

from __future__ import annotations

import collections
import json
import math
import random
import re
import shutil
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .data import LANGUAGES, read_jsonl
from .logic import decide, equivalent, signature, vocabulary
from .prompts import (
    COMMON_SENSE,
    JUDGE_SYSTEM,
    SENTENCE_GRAMMAR,
    control_story_prompt,
    formula_lines,
    free_grammar,
    free_translator_prompt,
    judge_prompt,
    parse_judge,
    parse_translation,
    sentence_prompt,
    translator_grammar,
    translator_prompt,
)

__all__ = [
    "BASE_MODEL",
    "download_base_model",
    "train_adapter",
    "merge_adapter",
    "load_llm",
    "generate_readings",
    "formalize_dgt",
    "contract_pseudo_labels",
    "evaluate_controls",
    "ask_judge",
]

BASE_MODEL = "mistralai/Ministral-3-8B-Base-2512"
_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_NOT_TEXT = ("vision", "audio", "multi_modal", "mm_", "embed_vision", "embed_audio", "patch", "projector")


# ----------------------------------------------------------------------------- training

def download_base_model(local_dir: str | Path, model: str = BASE_MODEL) -> Path:
    """Download the base model once, in Hugging Face format, for training and for vLLM."""
    from huggingface_hub import snapshot_download

    local_dir = Path(local_dir)
    if not (local_dir / "config.json").exists():
        snapshot_download(model, local_dir=str(local_dir),
                          ignore_patterns=["consolidated*", "params.json", "*.md", ".gitattributes"])
    return local_dir


def train_adapter(base_dir: str | Path, train_path: str | Path, adapter_dir: str | Path, steps: int,
                  max_len: int = 1024, learning_rate: float = 2e-4, micro_batch: int = 4, accumulation: int = 4,
                  seed: int = 0) -> dict:
    """QLoRA on the language model only: the base stays frozen in 4-bit NF4 and rank 16
    adapters are trained on the seven linear projections of every block.

    Every example is a ``{"prompt", "target"}`` pair and the loss is computed on the
    target only. The schedule is 20 steps of linear warm-up and linear decay to zero.
    Translator v2 used 300 steps and ``max_len=768``; v3 400 and v4 500 steps with 1024.
    """
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, BitsAndBytesConfig

    torch.manual_seed(seed)
    random.seed(seed)
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(base_dir))
    except (ValueError, OSError):   # older transformers: Mistral's own tokenizer format
        tokenizer = transformers.MistralCommonBackend.from_pretrained(str(base_dir))
    eos = tokenizer.eos_token_id
    data = []
    for row in read_jsonl(train_path):
        prompt = list(tokenizer(row["prompt"], add_special_tokens=True)["input_ids"])
        target = list(tokenizer(row["target"], add_special_tokens=False)["input_ids"]) + [eos]
        data.append(((prompt + target)[:max_len], ([-100] * len(prompt) + target)[:max_len]))

    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = None
    for name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            model = getattr(transformers, name).from_pretrained(str(base_dir), dtype=torch.bfloat16, device_map={"": 0},
                                                                quantization_config=quantization)
            break
        except Exception:  # the base model registers under one of the two classes, depending on the version
            continue
    if model is None:
        raise RuntimeError(f"could not load {base_dir}")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    targets = sorted({name for name, module in model.named_modules()
                      if name.split(".")[-1] in _LORA_TARGETS and not any(s in name for s in _NOT_TEXT)
                      and (isinstance(module, torch.nn.Linear) or type(module).__name__.startswith("Linear4bit"))})
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=targets))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    model.train()
    position, running_loss = 0, 0.0
    for step in range(steps):
        for _ in range(accumulation):
            batch = [data[(position + j) % len(data)] for j in range(micro_batch)]
            position += micro_batch
            length = max(len(x) for x, _ in batch)
            inputs = torch.tensor([x + [pad] * (length - len(x)) for x, _ in batch]).cuda()
            labels = torch.tensor([y + [-100] * (length - len(y)) for _, y in batch]).cuda()
            mask = torch.tensor([[1] * len(x) + [0] * (length - len(x)) for x, _ in batch]).cuda()
            loss = model(input_ids=inputs, attention_mask=mask, labels=labels).loss
            (loss / accumulation).backward()
            running_loss = 0.98 * running_loss + 0.02 * loss.item() if step else loss.item()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad()
        if (step + 1) % 20 == 0:
            print(f"step {step + 1}/{steps}  loss {running_loss:.3f}", flush=True)

    adapter_dir = Path(adapter_dir)
    model.save_pretrained(str(adapter_dir))
    info = {"base": str(base_dir), "examples": len(data), "steps": steps, "micro_batch": micro_batch,
            "accumulation": accumulation, "learning_rate": learning_rate, "max_len": max_len,
            "trainable_parameters": trainable, "total_parameters": total, "final_loss": running_loss}
    (adapter_dir / "train_info.json").write_text(json.dumps(info, indent=1))
    return info


def merge_adapter(base_dir: str | Path, adapter_dir: str | Path, merged_dir: str | Path) -> Path:
    """Merge an adapter into a bfloat16 copy of the base model that vLLM can load. Runs on
    CPU, since the full model does not fit on a 24 GB GPU next to anything else."""
    import torch
    import transformers
    from peft import PeftModel

    merged_dir = Path(merged_dir)
    if (merged_dir / "config.json").exists():
        return merged_dir
    base = None
    for name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        try:
            base = getattr(transformers, name).from_pretrained(str(base_dir), dtype=torch.bfloat16, device_map="cpu",
                                                               low_cpu_mem_usage=True)
            break
        except Exception:
            continue
    if base is None:
        raise RuntimeError(f"could not load {base_dir}")
    PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload().save_pretrained(str(merged_dir),
                                                                                       safe_serialization=True)
    for path in Path(base_dir).iterdir():
        if (path.is_file() and not (merged_dir / path.name).exists() and not path.name.startswith("consolidated")
                and not path.name.endswith((".safetensors", ".safetensors.index.json")) and path.name != "params.json"):
            shutil.copy(path, merged_dir / path.name)
    return merged_dir


# ----------------------------------------------------------------------------- generation

def load_llm(model_dir: str | Path, gpu_memory: float = 0.88):
    """vLLM engine for a merged translator, text only, with engine seed 0."""
    from vllm import LLM

    error = None
    for limits in ({"image": 0, "audio": 0, "video": 0}, {"image": 0}, None):
        extra = {} if limits is None else {"limit_mm_per_prompt": limits}
        try:
            return LLM(model=str(model_dir), max_model_len=4096, gpu_memory_utilization=gpu_memory,
                       trust_remote_code=True, seed=0, **extra)
        except Exception as exc:  # vLLM versions differ in which modalities they accept for Mistral 3
            error = exc
    raise error


def _constrained(temperature: float, max_tokens: int, grammar: str, seed: int | None = None, **extra):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    return SamplingParams(temperature=temperature, seed=seed, max_tokens=max_tokens,
                          structured_outputs=StructuredOutputsParams(grammar=grammar), **extra)


def generate_readings(llm, stories: Sequence[dict], few_shot: bool, temperature: float = 0.5, k: int = 6,
                      only: set[str] | None = None) -> dict[str, list[list[str]]]:
    """Readings of every story: reading 0 by greedy decoding and readings 1 to ``k``
    sampled at ``temperature``, all constrained by the story's grammar.

    The seed of reading ``j`` of the story at position ``i`` is ``1000 i + j``, so a
    story can be regenerated alone (``only``) and gets the readings it would get in a
    full run. Returns ``{story id: [reading 0, ..., reading k]}``, each reading being one
    formula per premise followed by the claim formula.
    """
    prompts, params, keys = [], [], []
    for i, story in enumerate(stories):
        if only is not None and story["id"] not in only:
            continue
        prompt, grammar = translator_prompt(story, few_shot), translator_grammar(story)
        max_tokens = 80 * (len(story["premises"]) + 1)
        for j in range(k + 1):
            prompts.append(prompt)
            keys.append((i, j))
            params.append(_constrained(0.0 if j == 0 else temperature, max_tokens, grammar, seed=1000 * i + j))
    readings: dict[str, list] = collections.defaultdict(lambda: [None] * (k + 1))
    for (i, j), output in zip(keys, llm.generate(prompts, params)):
        story = stories[i]
        readings[story["id"]][j] = parse_translation(output.outputs[0].text, len(story["premises"]))
    return dict(readings)


def formalize_dgt(llm, dgt_rows: Sequence[dict]) -> list[str]:
    """The base model writes each English DGT sentence in logic, greedily and with four
    solved examples. These are the candidate answers of the v2 mixture."""
    params = _constrained(0.0, 200, SENTENCE_GRAMMAR)
    outputs = llm.generate([sentence_prompt(row["en"], True) for row in dgt_rows], params)
    return [o.outputs[0].text.strip() for o in outputs]


def contract_pseudo_labels(llm, cases: Sequence[dict], k: int = 4, temperature: float = 0.7) -> tuple[list[dict], dict]:
    """ContractNLI cases written in logic by translator v2 with its own names. A
    translation is kept when Z3 decides it with the human label, the claim is not a copy
    of a premise and the premises are consistent; it then becomes a training example in
    the vocabulary format, with the vocabulary of its own formulas."""
    prompts, params, keys = [], [], []
    for i, case in enumerate(cases):
        prompt, grammar = free_translator_prompt(case["premises"], case["conclusion"]), free_grammar(len(case["premises"]))
        max_tokens = 120 * (len(case["premises"]) + 1)
        for j in range(k):
            prompts.append(prompt)
            keys.append((i, j))
            params.append(_constrained(0.0 if j == 0 else temperature, max_tokens, grammar, seed=1000 * i + j))
    candidates = collections.defaultdict(list)
    for (i, _), output in zip(keys, llm.generate(prompts, params)):
        candidates[i].append(parse_translation(output.outputs[0].text, len(cases[i]["premises"])))
    rejected, examples = collections.Counter(), []
    for i, case in enumerate(cases):
        n = len(case["premises"])
        for reading in candidates[i]:
            premises, claim = reading[:n], reading[n]
            if not all(premises) or not claim:
                rejected["empty"] += 1
            elif claim in premises:
                rejected["copy"] += 1
            else:
                label = decide(premises, claim)
                if label == "Error":
                    rejected["inconsistent_or_unparsable"] += 1
                elif label != case["label"]:
                    rejected["wrong_label"] += 1
                else:
                    story = {"premises": case["premises"], "conclusion": case["conclusion"],
                             "vocab": vocabulary(reading, decimals=False)}
                    examples.append({"prompt": translator_prompt(story), "target": formula_lines(premises, claim),
                                     "src": "contractnli", "label": case["label"]})
                    break
    report = {"cases": len(cases), "kept": len(examples),
              "kept_by_label": dict(collections.Counter(e["label"] for e in examples)), "rejected_readings": dict(rejected)}
    return examples, report


# ----------------------------------------------------------------------------- controls outside FOLIO



def _same_numbers(sig: dict | None) -> dict | None:
    if sig is None:
        return None
    sig = dict(sig)
    sig["nums"] = tuple(sorted(re.sub(r"[.,\s]", "", n).lstrip("0") or "0" for n in sig["nums"]))
    return sig


def evaluate_controls(llm, folio_validation: Sequence[dict], dgt_test: Sequence[dict],
                      common_sense: Sequence[dict]) -> dict:
    """The controls of Appendix C.

    * FOLIO without a vocabulary: the translator chooses its own names and Z3 decides.
    * DGT: the same legal sentence in six languages should give the same formula. Reports
      the share of valid formulas and, against the English formula, the share with the
      same negations, the same numbers and the same shape.
    * Common sense: yes or no to "is X a kind of Y?" in six languages, read from the
      probabilities of the first answer token.
    """
    from vllm import SamplingParams

    results = {}
    # FOLIO without a vocabulary
    longest = max(len(r["premises"]) for r in folio_validation)
    prompts = [control_story_prompt(r["premises"], r["conclusion"]) for r in folio_validation]
    grammars = [free_grammar(len(r["premises"])) for r in folio_validation]
    outputs = llm.generate(prompts, [_constrained(0.0, 90 * (longest + 1), g) for g in grammars])
    counts = collections.Counter()
    for row, output in zip(folio_validation, outputs):
        reading = parse_translation(output.outputs[0].text, len(row["premises"]))
        decision = decide(reading[:-1], reading[-1]) if all(reading) else "Error"
        counts["n"] += 1
        counts["correct"] += decision == row["label"]
        counts["false_pass"] += decision == "True" and row["label"] != "True"
    results["folio_without_vocabulary"] = {"n": counts["n"], "accuracy": counts["correct"] / counts["n"],
                                           "false_pass": counts["false_pass"]}

    # DGT in six languages
    keys = [(i, lang) for i in range(len(dgt_test)) for lang in LANGUAGES]
    outputs = llm.generate([sentence_prompt(dgt_test[i][lang], True) for i, lang in keys],
                           _constrained(0.0, 160, SENTENCE_GRAMMAR))
    formulas = {key: o.outputs[0].text.strip().split("\n")[0].strip() for key, o in zip(keys, outputs)}
    parsed = total = 0
    against_english = collections.Counter()
    for i in range(len(dgt_test)):
        english = _same_numbers(signature(formulas[(i, "en")]))
        for lang in LANGUAGES:
            sig = signature(formulas[(i, lang)])
            total += 1
            parsed += sig is not None
            if lang == "en" or english is None:
                continue
            sig = _same_numbers(sig)
            against_english["n"] += 1
            if sig is None:
                continue
            against_english["same_shape"] += sig == english
            against_english["same_negations"] += sig["neg"] == english["neg"]
            against_english["same_numbers"] += sig["nums"] == english["nums"]
            against_english["equivalent"] += equivalent(formulas[(i, lang)], formulas[(i, "en")]) is True
    n = max(against_english["n"], 1)
    results["dgt"] = {"sentences": len(dgt_test), "valid_formulas": parsed / total,
                      **{k: against_english[k] / n for k in ("same_shape", "same_negations", "same_numbers", "equivalent")}}

    # Common sense
    prompts, keys = [], []
    for i, question in enumerate(common_sense):
        for lang in LANGUAGES:
            template, _, _, shots = COMMON_SENSE[lang]
            examples = "".join(template.format(x=a, y=b) + f" {answer}\n\n" for a, b, answer in shots)
            prompts.append(examples + template.format(x=question["x"][lang], y=question["y"][lang]))
            keys.append((i, lang))
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20))
    correct = 0
    for (i, lang), output in zip(keys, outputs):
        _, yes_words, no_words, _ = COMMON_SENSE[lang]
        p_yes = p_no = 0.0
        for info in (output.outputs[0].logprobs[0] if output.outputs[0].logprobs else {}).values():
            token = (info.decoded_token or "").strip().strip('"«»„“').lower()
            if not token:
                continue
            if any(token == w or (len(token) >= 2 and w.startswith(token)) for w in yes_words):
                p_yes += math.exp(info.logprob)
            elif any(token == w or (len(token) >= 2 and w.startswith(token)) for w in no_words):
                p_no += math.exp(info.logprob)
        answer = None if p_yes == p_no == 0 else p_yes > p_no
        correct += answer == common_sense[i]["answer"]
    results["common_sense"] = {"questions": len(common_sense), "accuracy": correct / len(keys)}
    return results


# ----------------------------------------------------------------------------- hosted judges

def _client(provider: str, model: str, reasoning_effort: str | None):
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic()

        def call(system: str, prompt: str) -> str:
            # No temperature is sent: the client library used in October 2026 rejected it for this
            # model, so the answers in data/judges/ were produced with the provider's default sampling.
            message = client.messages.create(model=model, max_tokens=1200, system=system,
                                             messages=[{"role": "user", "content": prompt}])
            return "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
    else:
        import openai

        client = openai.OpenAI()

        use_effort = {"value": bool(reasoning_effort)}

        def call(system: str, prompt: str) -> str:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            extra = {"reasoning_effort": reasoning_effort} if use_effort["value"] else {}
            try:
                response = client.chat.completions.create(model=model, messages=messages,
                                                          max_completion_tokens=6000, **extra)
            except openai.BadRequestError as error:
                if "reasoning_effort" not in str(error) or not use_effort["value"]:
                    raise
                use_effort["value"] = False   # a model without reasoning effort
                response = client.chat.completions.create(model=model, messages=messages, max_completion_tokens=6000)
            return response.choices[0].message.content or ""
    return call


def ask_judge(provider: str, model: str, sets: dict[str, Sequence[dict]], answers_path: str | Path,
              reasoning_effort: str | None = "low", workers: int = 8) -> dict[str, dict[str, str]]:
    """Ask a hosted model for one verdict per story, with the prompt of Appendix A.

    ``provider`` is ``"anthropic"`` or ``"openai"``. Every raw answer is appended to
    ``answers_path`` as it arrives, so an interrupted run resumes where it stopped.
    Returns ``{set: {story id: "True" | "False" | "Uncertain"}}``.
    """
    answers_path = Path(answers_path)
    done = {}
    if answers_path.exists():
        for line in answers_path.read_text(encoding="utf-8").splitlines():
            try:
                answer = json.loads(line)
            except json.JSONDecodeError:   # the last line of an interrupted run
                continue
            done[(answer["set"], answer["id"])] = answer["text"]
    call = _client(provider, model, reasoning_effort)
    lock = threading.Lock()

    def ask(job):
        set_name, story = job
        for attempt in range(8):
            try:
                text = call(JUDGE_SYSTEM, judge_prompt(story))
                break
            except Exception:  # rate limits and transient errors: back off and retry
                if attempt == 7:
                    raise
                time.sleep(min(60, 2 ** attempt + random.random()))
        with lock, open(answers_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"set": set_name, "id": story["id"], "text": text}, ensure_ascii=False) + "\n")
        done[(set_name, story["id"])] = text

    jobs = [(name, s) for name, stories in sets.items() for s in stories if (name, s["id"]) not in done]
    with ThreadPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(ask, job) for job in jobs]):
            future.result()
    return {name: {s["id"]: parse_judge(done.get((name, s["id"]))) for s in stories} for name, stories in sets.items()}
