#!/usr/bin/env python3
"""
Stage 00 / step 5 — the two reference rows.

Neither needs a model, a GPU, or the network. They bracket the table:

  constant baseline   a "model" that emits the single most common answer for
                      every field. Tells a reader how much of any score is real
                      signal versus the field's base rate. This is the control
                      that would have caught `phase` (0.91 for free) before it
                      polluted the headline.

  human ceiling       your blind labels scored against the adjudicated gold,
                      through the identical scorer. Nothing should be expected
                      to beat it: two careful raters following the same written
                      rules only got this far.

    python run_baselines.py
"""
import argparse, json, os
from collections import Counter

import ctg_common as C


def load(path):
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("dropped") or "gold" not in r:
            continue
        out[r["nct_id"]] = r["gold"]
    return out


def most_common_answer(golds):
    """The best single constant a lazy model could emit for each field."""
    const = {}
    for f in C.SCHEMA_KEYS:
        vals = [g.get(f) for g in golds]
        if f == "allocation":
            const[f] = Counter(str(v) for v in vals).most_common(1)[0][0]
            if const[f] == "None":
                const[f] = None
        else:
            # most common exact list/set, as a tuple so it's hashable
            c = Counter(tuple(sorted(v or [])) for v in vals)
            const[f] = list(c.most_common(1)[0][0])
    return const


def score_all(preds, golds):
    """preds and golds are aligned lists of records."""
    per_field = {f: [] for f in C.SCHEMA_KEYS}
    headline = []
    for p, g in zip(preds, golds):
        s = C.record_score(p, g)
        for f in C.SCHEMA_KEYS:
            per_field[f].append(s[f])
        headline.append(C.headline(s))
    return ({f: sum(v) / len(v) for f, v in per_field.items()},
            sum(headline) / len(headline))


def row(label, per_field, head, n):
    cells = "".join(f"{per_field[f]:>21.2f}" for f in C.SCHEMA_KEYS)
    print(f"{label:<22}{head:>10.2f}{cells}{n:>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="eval_gold.jsonl")
    ap.add_argument("--blind", default="eval_blind.jsonl")
    args = ap.parse_args()

    gold = load(args.gold)
    if not gold:
        raise SystemExit(f"no records in {args.gold}")
    golds = list(gold.values())

    hdr = "".join(f"{f:>21}" for f in C.SCHEMA_KEYS)
    print(f"\n{'row':<22}{'headline':>10}{hdr}{'n':>6}")
    print("-" * (38 + 21 * len(C.SCHEMA_KEYS)))

    # ---- constant baseline ------------------------------------------------
    const = most_common_answer(golds)
    pf, head = score_all([const] * len(golds), golds)
    row("constant baseline", pf, head, len(golds))
    print(f"{'':22}emits {json.dumps(const)}")

    # ---- empty baseline ---------------------------------------------------
    empty = {f: (None if f == "allocation" else []) for f in C.SCHEMA_KEYS}
    pf, head = score_all([empty] * len(golds), golds)
    row("empty baseline", pf, head, len(golds))

    # ---- human ceiling ----------------------------------------------------
    blind = load(args.blind)
    shared = [i for i in blind if i in gold]
    if shared:
        pf, head = score_all([blind[i] for i in shared], [gold[i] for i in shared])
        row("human ceiling", pf, head, len(shared))
        print(f"\nThe human ceiling is the practical maximum for this eval — two\n"
              f"raters following the same written rules reached {head:.2f}. Quote it\n"
              f"in the table above the model rows. n={len(shared)}, so treat it as\n"
              f"indicative rather than tight.")
    else:
        print(f"\n(no overlap between {args.blind} and {args.gold} — "
              f"run `adjudicate.py --blind 15` for the ceiling row)")

    print("\nBoth rows are model-free: they come from the gold file alone, so they\n"
          "cost nothing to recompute whenever the eval set changes.")


if __name__ == "__main__":
    main()
