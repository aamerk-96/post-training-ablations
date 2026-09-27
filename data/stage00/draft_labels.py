#!/usr/bin/env python3
"""
Stage 00 / step 3a — draft gold labels with Claude.

Reads a jsonl of {nct_id, document} and writes {nct_id, gold} drafts. These are
DRAFTS. They become gold only after adjudication (adjudicate.py), and the
dataset card must say the labels were model-drafted and human-adjudicated,
with the kappa from the blind pass.

    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python draft_labels.py --in eval_candidates.jsonl --out eval_drafts.jsonl

Resumable — re-run after an interruption and it only does what's missing.
"""
import argparse, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor

import ctg_common as C

ALLOWED_ALLOC = {"RANDOMIZED", "NON_RANDOMIZED"}
ALLOWED_TYPES = set(C.INTERVENTION_TYPES)

_lock = threading.Lock()

def pick_model(client, requested):
    if requested:
        return requested
    # Don't hardcode a model id that may have been retired — ask the API.
    models = [m.id for m in client.models.list(limit=50).data]
    for want in ("sonnet", "opus", "haiku"):
        for m in models:
            if want in m.lower():
                return m
    if not models:
        sys.exit("No models available to this API key.")
    return models[0]

def extract_json(text):
    """First balanced {...} in the response. Lenient extraction, strict validation."""
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
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None

def clean(raw):
    """Coerce a model response into the schema. Invalid enum values become null."""
    if not isinstance(raw, dict):
        return None

    def as_list(v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return [str(x).strip() for x in v if str(x).strip()]

    def as_enum(v, allowed):
        if v is None:
            return None
        s = str(v).strip().upper().replace(" ", "_").replace("-", "_")
        s = {"PHASE_1": "PHASE1", "PHASE_2": "PHASE2", "PHASE_3": "PHASE3",
             "PHASE_4": "PHASE4", "NULL": None, "NONE": None}.get(s, s)
        return s if s in allowed else None

    types = [t for t in (as_enum(x, ALLOWED_TYPES) for x in as_list(
        raw.get("intervention_types"))) if t]
    return {
        "conditions":         as_list(raw.get("conditions")),
        "interventions":      as_list(raw.get("interventions")),
        "intervention_types": sorted(set(types)),
        "allocation":         as_enum(raw.get("allocation"), ALLOWED_ALLOC),
    }

# The anthropic SDK (>=1.x) dropped `temperature` from messages.create and added
# structured outputs via output_config.format. That is strictly better here: the
# API enforces the schema, so we never see a malformed record from the labeller.
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "conditions":         {"type": "array", "items": {"type": "string"}},
        "interventions":      {"type": "array", "items": {"type": "string"}},
        "intervention_types": {"type": "array",
                               "items": {"type": "string",
                                         "enum": C.INTERVENTION_TYPES}},
        # A nullable enum must be expressed as anyOf. Putting null inside `enum`
        # alongside "type": ["string","null"] is rejected by the validator.
        "allocation":         {"anyOf": [
                                  {"type": "string",
                                   "enum": ["RANDOMIZED", "NON_RANDOMIZED"]},
                                  {"type": "null"}]},
    },
    "required": ["conditions", "interventions", "intervention_types",
                 "allocation"],
    "additionalProperties": False,
}

def _create(client, model, prompt, structured=True):
    kwargs = {"model": model, "max_tokens": 1000,
              "messages": [{"role": "user", "content": prompt}]}
    if structured:
        kwargs["output_config"] = {"format": {"type": "json_schema",
                                              "schema": JSON_SCHEMA}}
    return client.messages.create(**kwargs)

def label_one(client, model, rec, retries=3, structured=True):
    for attempt in range(retries):
        try:
            msg = _create(client, model, C.build_prompt(rec["document"]), structured)
            text = "".join(b.text for b in msg.content if b.type == "text")
            gold = clean(extract_json(text))
            if gold is not None:
                return {"nct_id": rec["nct_id"], "gold": gold, "model": model}
        except TypeError as e:
            # SDK doesn't know output_config — fall back to plain prompting,
            # which is why extract_json() above stays lenient.
            if structured:
                structured = False
                continue
            return {"nct_id": rec["nct_id"], "error": str(e)[:200]}
        except Exception as e:                     # rate limit, transient 5xx
            if attempt == retries - 1:
                return {"nct_id": rec["nct_id"], "error": str(e)[:200]}
            import time
            time.sleep(2 ** attempt)
    return {"nct_id": rec["nct_id"], "error": "unparseable response"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="eval_candidates.jsonl")
    ap.add_argument("--out", dest="out", default="eval_drafts.jsonl")
    ap.add_argument("--model", default=os.environ.get("ANTHROPIC_MODEL"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0,
                    help="draft only the first N records (cost control). The "
                         "input order is already shuffled by build_dataset.py's "
                         "seed, so a prefix is a random sample.")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set.")
    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")

    client = anthropic.Anthropic()
    model = pick_model(client, args.model)
    print(f"model: {model}", file=sys.stderr)

    recs = [json.loads(l) for l in open(args.inp) if l.strip()]
    if args.limit:
        recs = recs[: args.limit]
    done = {}
    if os.path.exists(args.out):
        done = {json.loads(l)["nct_id"]: json.loads(l)
                for l in open(args.out) if l.strip()}
        done = {k: v for k, v in done.items() if "error" not in v}
    todo = [r for r in recs if r["nct_id"] not in done]
    print(f"{len(done)} already drafted, {len(todo)} to go", file=sys.stderr)

    n_done = [0]
    def work(r):
        out = label_one(client, model, r)
        with _lock:
            done[r["nct_id"]] = out
            n_done[0] += 1
            if n_done[0] % 10 == 0 or n_done[0] == len(todo):
                print(f"  {n_done[0]}/{len(todo)}", file=sys.stderr)
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))

    with open(args.out, "w") as f:
        for r in recs:                              # preserve input order
            if r["nct_id"] in done:
                f.write(json.dumps(done[r["nct_id"]]) + "\n")

    errs = [v for v in done.values() if "error" in v]
    print(f"\nwrote {args.out}: {len(done) - len(errs)} drafted, {len(errs)} failed")
    for e in errs[:5]:
        print(f"  {e['nct_id']}: {e['error']}")
    if errs:
        print("re-run to retry the failures")

if __name__ == "__main__":
    main()
