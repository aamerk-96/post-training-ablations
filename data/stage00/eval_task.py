"""
Stage 00 / step 4 — the eval task. ONE task, reused unchanged for every
checkpoint. A second task per stage would make the numbers incomparable.

Four scorers over the same generation:

  field_recall          gold values found anywhere in the output, prose
                        tolerated. Nonzero on an untrained base model — covers
                        the bottom of the range. NEVER use as a GRPO reward:
                        recall-only, so echoing the document scores 1.0.
  json_parse_rate       binary: did a parseable object come out.
  schema_complete_rate  binary: does it carry all four keys. Separate from
                        parse rate because "emits JSON" and "emits the right
                        record" are learned at different points.
  field_f1_strict       HEADLINE. token-F1 per field, mean over four fields.
                        What DPO/GRPO move; safe as a GRPO reward.

Reference rows come from run_baselines.py (no model needed): constant-answer
baseline 0.26, empty baseline 0.16, human ceiling 0.85 at n=15.

    # base model — completions endpoint, no chat template
    OPENAI_API_KEY=none inspect eval eval_task.py \
      --model vllm-completions/HuggingFaceTB/SmolLM2-360M \
      --model-base-url http://localhost:8000/v1

    # pass@k arm, separate run
    ... -T temperature=0.7 -T epochs=4
"""
import json
import os
import re

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Target, max_score, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate

import ctg_common as C

GOLD_FILE = os.environ.get("STAGE00_GOLD", "eval_gold.jsonl")


# --------------------------------------------------------------------- dataset
def load_dataset(path=GOLD_FILE):
    samples = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("dropped") or "gold" not in r:
                continue
            samples.append(
                Sample(
                    id=r["nct_id"],
                    input=C.build_prompt(r["document"]),
                    target=json.dumps(r["gold"]),
                    metadata={"document": r["document"], "gold": r["gold"]},
                )
            )
    if not samples:
        raise RuntimeError(f"no usable records in {path}")
    return MemoryDataset(samples)


# ------------------------------------------------------------------ extraction
def extract_json(text):
    """First balanced {...}. Lenient: models wrap JSON in prose and code fences,
    and that is not what we are measuring."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def gold_of(state: TaskState, target: Target):
    g = (state.metadata or {}).get("gold")
    return g if g else json.loads(target.text)


# -------------------------------------------------------------------- scorers
@scorer(metrics=[mean(), stderr()])
def field_recall():
    """Are the gold values present anywhere in the output, in any format?

    This is the only metric that works on an untrained base model, which emits
    prose rather than JSON. Without it the first three rows of the table read
    0.00 and the CPT stage is unmeasurable.
    """

    async def score(state: TaskState, target: Target):
        gold = gold_of(state, target)
        out = C.norm(state.output.completion)
        wanted = []
        for f in ("conditions", "interventions"):
            wanted += [str(v) for v in (gold.get(f) or [])]
        for f in ("intervention_types",):
            wanted += [str(v) for v in (gold.get(f) or [])]
        if gold.get("allocation"):
            wanted.append(str(gold["allocation"]))
        if not wanted:
            return Score(value=1.0, explanation="nothing to find")
        hits = sum(1 for v in wanted if C.span_present(v, out))
        return Score(
            value=hits / len(wanted),
            answer=state.output.completion[:200],
            explanation=f"{hits}/{len(wanted)} gold values present in the output",
        )

    return score


@scorer(metrics=[mean(), stderr()])
def json_parse_rate():
    """Did a parseable JSON object come out at all? Strictly binary.

    Previously this returned 0.5 for "parsed but missing keys", which made the
    mean unreadable: a 0.49 could mean half the samples were perfect or that
    nearly all of them were partial. Those are completely different findings,
    so schema completeness is now its own scorer.
    """

    async def score(state: TaskState, target: Target):
        obj = extract_json(state.output.completion)
        if obj is None:
            return Score(value=0.0, explanation="no parseable JSON object")
        return Score(value=1.0, answer=json.dumps(obj)[:200],
                     explanation="parseable object")

    return score


@scorer(metrics=[mean(), stderr()])
def schema_complete_rate():
    """Of the output, does it carry all four schema keys? Also binary.

    Splitting this from parse rate separates "learned to emit JSON" from
    "learned the right record", which are taught at different points.
    """

    async def score(state: TaskState, target: Target):
        obj = extract_json(state.output.completion)
        if obj is None:
            return Score(value=0.0, explanation="no parseable JSON object")
        missing = [k for k in C.SCHEMA_KEYS if k not in obj]
        if missing:
            return Score(value=0.0, explanation=f"missing keys: {missing}")
        extra = [k for k in obj if k not in C.SCHEMA_KEYS]
        return Score(value=1.0,
                     explanation="all schema keys present"
                                 + (f" (plus extra: {extra})" if extra else ""))

    return score


@scorer(metrics=[mean(), stderr()])
def field_f1_strict():
    """THE HEADLINE. Mean token-F1 over the four fields.

    Token-F1 rather than exact match, and that choice is empirically justified:
    two human raters following the same written rules reach only 40% exact
    agreement on `interventions` but 0.81 token-F1. Exact match would put the
    human ceiling at 0.40 and make the eval look broken.

    Per-field scores go in metadata so the failure taxonomy can be read off the
    logs without re-running anything.
    """

    async def score(state: TaskState, target: Target):
        gold = gold_of(state, target)
        obj = extract_json(state.output.completion)
        if obj is None:
            return Score(
                value=0.0,
                explanation="no parseable JSON — scored 0, see json_parse_rate",
                metadata={f: 0.0 for f in C.SCHEMA_KEYS},
            )
        per_field = C.record_score(obj, gold)
        return Score(
            value=C.headline(per_field),
            answer=json.dumps(obj)[:300],
            explanation=", ".join(f"{k}={v:.2f}" for k, v in per_field.items()),
            metadata=per_field,
        )

    return score


# ----------------------------------------------------------------------- task
@task
def extraction(gold: str = GOLD_FILE, temperature: float = 0.0,
               epochs: int = 1, stop: bool = False):
    """
    Decoding is pinned HERE, not on the CLI. Half of every "my fine-tune
    improved things" is temperature drift between runs; putting it in the task
    config means every checkpoint is scored under identical conditions and the
    config travels with the repo.

    TWO INTENDED CONFIGURATIONS, and mixing them is a mistake:

      headline   temperature=0.0, epochs=1   <- the default, the table's number
      pass@k     temperature=0.7, epochs=4   <- a SEPARATE column

    epochs>1 at temperature 0 is greedy, so every epoch is an identical
    generation: you pay k times for one result AND the reported stderr treats
    k*n correlated samples as independent, understating it by roughly sqrt(k).

    With epochs>1 the reducer becomes max, so the reported number is "best of
    k" — actual pass@k. inspect's default reducer is the MEAN across epochs,
    which at temperature 0.7 measures the average quality of a sampled
    generation and reads LOWER than greedy. That is a different quantity and
    reporting it as pass@k would be wrong.
    """
    return Task(
        dataset=load_dataset(gold),
        solver=generate(),
        scorer=[field_recall(), json_parse_rate(), schema_complete_rate(),
                field_f1_strict()],
        # mean across epochs answers "how good is a typical sample";
        # max answers "could it do it in k tries". pass@k wants the latter.
        epochs=Epochs(epochs, max_score()) if epochs > 1 else epochs,
        config=GenerateConfig(
            temperature=temperature,
            top_p=1.0,
            max_tokens=512,
            seed=20260906,
            # DIAGNOSTIC ONLY — not part of the headline configuration.
            # A base model has no stop token, so it continues past its answer
            # and invents the next document, which makes the JSON unparseable.
            # `-T stop=true` measures how much of the parse failure is
            # termination rather than comprehension. Report it as its own
            # labelled row; never swap it into the headline, or every
            # checkpoint after it becomes incomparable.
            stop_seqs=["\n\n", "\nDescription:", "\nKeywords:"] if stop else None,
        ),
    )
