#!/usr/bin/env python3
"""
Stage 02 prep — turn drafted labels into SFT training pairs, with a quality gate.

Why this exists: `sft_pool.jsonl` ships with gold derived from the registry's
structured fields, and Stage 00 established that those are frequently wrong about
the study's own description (a day-night-rhythm study in healthy volunteers is
labelled `Diabetes Mellitus, Type 2`). Training on them would teach the model to
reproduce registry noise, and would make the SFT row uninterpretable — a low
score would not distinguish "training failed" from "trained correctly on wrong
targets".

So the labels are re-drafted from the document by a model, then filtered here.

    python draft_labels.py --in sft_pool.jsonl --out sft_drafts.jsonl
    python build_sft.py

THE QUALITY GATE
----------------
Drafted labels are not reviewed by hand — there are too many. Instead every
value is checked back against the document it came from, using the same
`span_present` machinery the eval uses. A drafted condition or intervention that
does not appear in its own source text is a hallucination, and the record is
dropped rather than trained on. This is free, deterministic, and catches the one
failure mode that matters for a label the human never sees.

Records are also dropped when a draft failed outright, or when the label is
empty on both span fields (nothing to learn).

OUTPUT
------
`sft_train.jsonl`, one object per line:

    {"nct_id", "prompt", "completion", "document", "gold"}

`prompt` and `completion` are what TRL's SFTTrainer consumes; the extra keys are
kept for provenance and for regenerating the pairs if the prompt changes.
Completion-only loss means the model is trained on `completion` alone — the
prompt is masked. Do not concatenate them yourself.
"""
import argparse, json, os, sys
from collections import Counter

import ctg_common as C

MIN_SPAN_PRESENCE = 0.5      # fraction of a field's values that must be in the text


def load(path, need_gold=True):
    if not os.path.exists(path):
        sys.exit(f"{path} not found")
    out = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if need_gold and ("gold" not in r or r.get("error")):
            continue
        out[r["nct_id"]] = r
    return out


def grounded(gold, doc_norm):
    """Is every span-valued label actually present in the document?

    Returns (ok, reason). Enum fields are not checked — a modality is inferred
    from the intervention name, not quoted from the text.
    """
    for field in ("conditions", "interventions"):
        vals = gold.get(field) or []
        if not vals:
            continue
        present = sum(1 for v in vals if C.span_present(v, doc_norm))
        if present / len(vals) < MIN_SPAN_PRESENCE:
            return False, f"{field}: only {present}/{len(vals)} values in the text"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="sft_pool.jsonl",
                    help="documents (its `gold` is the registry derivation and is ignored)")
    ap.add_argument("--drafts", default="sft_drafts.jsonl")
    ap.add_argument("--out", default="sft_train.jsonl")
    ap.add_argument("--eval-gold", default="eval_gold.jsonl",
                    help="checked for id overlap — must be zero")
    ap.add_argument("--no-filter", action="store_true",
                    help="skip the grounding gate (for measuring how much it removes)")
    args = ap.parse_args()

    pool = load(args.pool, need_gold=False)
    drafts = load(args.drafts)
    print(f"pool {len(pool)} documents, {len(drafts)} successful drafts")

    # contamination check — the eval set must never appear in training
    ev = set(load(args.eval_gold, need_gold=False)) if os.path.exists(args.eval_gold) else set()
    overlap = ev & set(pool)
    if overlap:
        sys.exit(f"ABORT: {len(overlap)} eval ids present in {args.pool}: "
                 f"{sorted(overlap)[:5]}")

    rows, dropped = [], Counter()
    for nct, d in drafts.items():
        rec = pool.get(nct)
        if not rec:
            dropped["no matching document"] += 1
            continue
        gold = d["gold"]
        if not gold.get("conditions") and not gold.get("interventions"):
            dropped["empty on both span fields"] += 1
            continue
        if not args.no_filter:
            ok, why = grounded(gold, C.norm(rec["document"]))
            if not ok:
                dropped[f"ungrounded ({why.split(':')[0]})"] += 1
                continue
        rows.append({
            "nct_id": nct,
            "prompt": C.build_prompt(rec["document"]),
            "completion": json.dumps(gold, ensure_ascii=False),
            "document": rec["document"],
            "gold": gold,
        })

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    kept = len(rows)
    total = len(drafts)
    print(f"\nwrote {args.out}: {kept} training pairs "
          f"({kept/total:.0%} of drafts kept)")
    for k, v in dropped.most_common():
        print(f"  dropped, {k}: {v}")

    if rows:
        pl = sum(len(r["prompt"]) for r in rows) / kept / 4
        cl = sum(len(r["completion"]) for r in rows) / kept / 4
        print(f"\navg prompt ~{pl:.0f} tokens, avg completion ~{cl:.0f} tokens")
        print(f"~{kept * (pl + cl) / 1e6:.2f}M tokens per epoch")

        # ---- distribution check against the eval set ----------------------
        # The training labels and the eval gold come from the same model but
        # different runs. If their label distributions diverge, a later SFT
        # result is partly explained by that shift rather than by learning, and
        # it is very hard to diagnose after the fact. Check it now.
        def profile(golds):
            n = len(golds) or 1
            return {
                "avg conditions":     sum(len(g.get("conditions") or []) for g in golds) / n,
                "empty conditions":   sum(1 for g in golds if not g.get("conditions")) / n,
                "avg interventions":  sum(len(g.get("interventions") or []) for g in golds) / n,
                "empty interventions":sum(1 for g in golds if not g.get("interventions")) / n,
                "avg types":          sum(len(g.get("intervention_types") or []) for g in golds) / n,
                "allocation null":    sum(1 for g in golds if g.get("allocation") is None) / n,
            }

        train_p = profile([r["gold"] for r in rows])
        ev_rows = load(args.eval_gold) if os.path.exists(args.eval_gold) else {}
        ev_p = profile([r["gold"] for r in ev_rows.values()]) if ev_rows else None

        # Rates live in [0,1] so an absolute threshold is meaningful; counts do
        # not, and judging a mean of 2.3 by the same 0.15 yardstick flags a 10%
        # difference as a problem. Rates get an absolute test, counts a relative
        # one.
        RATE_ROWS = {"empty conditions", "empty interventions", "allocation null"}
        print(f"\n{'':22}{'train':>9}{'eval':>9}{'delta':>9}{'rel':>8}")
        print("-" * 57)
        flagged = []
        for k in train_p:
            if not ev_p:
                print(f"{k:<22}{train_p[k]:>9.2f}{'—':>9}{'—':>9}{'—':>8}")
                continue
            d = train_p[k] - ev_p[k]
            rel = d / ev_p[k] if ev_p[k] else 0.0
            bad = abs(d) > 0.15 if k in RATE_ROWS else abs(rel) > 0.25
            if bad:
                flagged.append(k)
            print(f"{k:<22}{train_p[k]:>9.2f}{ev_p[k]:>9.2f}{d:>+9.2f}"
                  f"{rel:>+7.0%}{'   <-- check' if bad else ''}")
        if ev_p:
            if flagged:
                print(f"\nFlagged: {', '.join(flagged)}. The training and test label\n"
                      "distributions differ enough to confound the SFT result — a\n"
                      "later change in score would be partly explained by the shift.\n"
                      "Fix the drafting prompt and re-draft before spending more.")
            else:
                print("\nDistributions match. Note that eval gold was adjudicated and "
                      "these\ndrafts were not, so a small gap on the count rows is "
                      "expected: hand\nadjudication tends to add missed conditions and "
                      "prune over-listed\ninterventions.")

        print(f"\nexample completion:\n  {rows[0]['completion'][:160]}")

    print("\nThe grounding gate is a proxy for review, not a substitute. It "
          "catches values\nthat are not in the source document; it cannot catch "
          "a plausible value that is\nin the document but wrong. Report the keep "
          "rate in the dataset card.")


if __name__ == "__main__":
    main()
