#!/usr/bin/env python3
"""
Stage 00 / step 3b — adjudicate, and measure your own agreement.

Three modes.

  python adjudicate.py
      Compares the deterministic gold against Claude's draft and shows you ONLY
      the records where they disagree. Agreements pass through untouched. Writes
      eval_gold.jsonl. This is the time saver — expect to see 25-35 of 100.

  python adjudicate.py --blind 15
      Labels 15 random records from scratch with BOTH machine answers hidden.
      This is the control. Do it before or after adjudication, never during, and
      do not peek. Writes eval_blind.jsonl.

  python adjudicate.py --kappa
      Cohen's kappa between your blind labels and Claude's drafts, per field.
      This is the number that makes the dataset defensible.

Keys when adjudicating: <enter>=take run A  a=take run B  e=edit  d=drop  b=back  q=quit
Keys in the blind pass:  1-4=set a field  <enter>=next record  q=quit
"""
import argparse, json, os, random, re, sys, textwrap
from collections import Counter

import ctg_common as C

# ---------------------------------------------------------------- display
# Inlined (previously imported from review.py, which is gone) so this script
# has no dependency beyond ctg_common.
BOLD, DIM, GREEN, RED, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m")
if not (sys.stdout.isatty() and os.environ.get("TERM") not in (None, "dumb")):
    BOLD = DIM = GREEN = RED = YELLOW = RESET = ""

DIM_, RESET_ = DIM, RESET

def show_doc(doc, values):
    """Print the document, highlighting the content words of every gold value."""
    toks = set()
    for v in values:
        toks.update(C.content_tokens(str(v)))
    out = doc
    if toks:
        pattern = "|".join(sorted((re.escape(t) for t in toks), key=len, reverse=True))
        out = re.sub(rf"(?i)(?<!\w)({pattern})(?!\w)", GREEN + BOLD + r"\1" + RESET, out)
    print(DIM + "-" * 78 + RESET)
    for para in out.split("\n\n"):
        print(textwrap.fill(para, 78, replace_whitespace=False))
        print()

def fmt(field, value, doc_norm):
    if field in LIST_FIELDS:
        if not value:
            return RED + "[]" + RESET
        if field == "intervention_types":          # enum labels, not text spans
            return "[" + ", ".join(value) + "]"
        marks = []
        for v in value:
            ok = C.span_present(v, doc_norm)
            marks.append(f"{GREEN if ok else RED}{v}{RESET}")
        return "[" + ", ".join(marks) + "]"
    if value is None:
        return YELLOW + "null" + RESET
    ok = C.enum_present(value, doc_norm)
    colour = GREEN if ok else (YELLOW if ok is None else RED)
    return f"{colour}{value}{RESET}"

def edit_field(gold, field):
    cur = gold[field]
    if field in LIST_FIELDS:
        print(f"{DIM}current: {cur}")
        print(f"comma-separated values, or empty line to clear the list{RESET}")
        raw = input(f"{field}> ").strip()
        vals = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
        if field == "intervention_types":
            vals = sorted({v.upper() for v in vals})
        gold[field] = vals
    else:
        print(f"{DIM}current: {cur}   (type 'null' if the text does not state it){RESET}")
        raw = input(f"{field}> ").strip()
        gold[field] = None if raw.lower() in ("null", "none", "") else raw.upper()
    return gold

FIELDS = C.SCHEMA_KEYS
LIST_FIELDS = {"conditions", "interventions", "intervention_types"}

def load(path, key="nct_id"):
    if not os.path.exists(path):
        return {}
    return {json.loads(l)[key]: json.loads(l) for l in open(path) if l.strip()}

def load_rows(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]

def save(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

# Agreement is judged with the same tolerance the scorer uses. Demanding exact
# token equality flagged 97/100 records as contested when most differed only in
# wording ("Hearing Loss" vs "Hearing Impairment"), which token-F1 already
# treats as near-identical.
AGREE_AT = 0.7

def field_agrees(field, a, b, thresh=None):
    t = AGREE_AT if thresh is None else thresh
    if field == "intervention_types":
        return C.set_f1(a, b) >= t
    if field in LIST_FIELDS:
        return C.list_f1(a, b) >= t
    return C.enum_score(a, b) == 1.0

# ------------------------------------------------------------------ adjudicate
def adjudicate(args):
    cands = load_rows(args.inp)
    drafts = load(args.drafts)
    if not drafts:
        sys.exit(f"{args.drafts} not found — run draft_labels.py first.")
    b = load(args.drafts_b) if args.drafts_b else {}
    out = load(args.out)

    # What sits in the left-hand column, and what "contested" is measured against.
    def reference(rec):
        if b:
            r = b.get(rec["nct_id"])
            return r["gold"] if r and "gold" in r else None
        return rec["gold"]

    contested, auto_ok = [], 0
    for rec in cands:
        d = drafts.get(rec["nct_id"])
        ref = reference(rec)
        if not d or "gold" not in d or ref is None:
            contested.append((rec, None))
            continue
        diffs = [f for f in FIELDS
                 if not field_agrees(f, ref[f], d["gold"][f])]
        if diffs:
            contested.append((rec, d["gold"]))
        else:
            auto_ok += 1
            if rec["nct_id"] not in out:
                # both runs agree -> take the drafted record, not the derivation
                out[rec["nct_id"]] = {"nct_id": rec["nct_id"],
                                      "document": rec["document"],
                                      "gold": d["gold"] if b else rec["gold"],
                                      "source": "agreed"}

    todo = [(r, g) for r, g in contested if r["nct_id"] not in out]
    mode = "two drafting runs agree" if b else "derivation and draft agree"
    print(f"{auto_ok}/{len(cands)} settled ({mode}).")
    print(f"{len(contested)} contested, {len(todo)} left to adjudicate.\n")
    if not todo:
        print("Nothing left. Run --kappa when your blind pass is done.")
        save(args.out, list(out.values()))
        return

    i = 0
    while 0 <= i < len(todo):
        rec, claude = todo[i]
        ref = reference(rec) or rec["gold"]
        gold = json.loads(json.dumps(ref))
        dn = C.norm(rec["document"])
        while True:
            print(f"\n{BOLD}=== {i+1}/{len(todo)}  {rec['nct_id']} ==={RESET}")
            show_doc(rec["document"],
                       list(gold["conditions"]) + list(gold["interventions"]))
            left_label = "RUN B (a)" if b else "DERIVED (a)"
            print(f"  {'':18}{left_label:<36}RUN A / CLAUDE (enter)")
            for k, f in enumerate(FIELDS, 1):
                same = claude and field_agrees(f, gold[f], claude[f])
                mark = " " if same else YELLOW + "≠" + RESET
                left = fmt(f, gold[f], dn)
                right = fmt(f, claude[f], dn) if claude else DIM + "(none)" + RESET
                pad = 36 + len(left) - len(re.sub(r"\x1b\[[0-9;]*m", "", left))
                print(f" {mark}{k} {f:<20}{left:<{pad}}{right}")
            cmd = input("\n<enter>=Claude  a=derived  e=edit  d=drop  b=back  q=quit > ").strip().lower()
            if cmd in ("", "c") and claude:
                gold, src = json.loads(json.dumps(claude)), "claude"
            elif cmd == "a":
                src = "derived"
            elif cmd == "e":
                which = input(f"field 1-{len(FIELDS)}> ").strip()
                if which in {str(x) for x in range(1, len(FIELDS) + 1)}:
                    gold = edit_field(gold, FIELDS[int(which) - 1])
                continue
            elif cmd == "d":
                out[rec["nct_id"]] = {"nct_id": rec["nct_id"], "dropped": True}
                i += 1
                break
            elif cmd == "b":
                i = max(0, i - 1)
                break
            elif cmd == "q":
                i = -1
                break
            else:
                print("  ?")
                continue
            out[rec["nct_id"]] = {"nct_id": rec["nct_id"], "document": rec["document"],
                                  "gold": gold, "source": src, "adjudicated": True}
            i += 1
            break
        save(args.out, list(out.values()))

    kept = [r for r in out.values() if not r.get("dropped")]
    print(f"\nsaved {args.out}: {len(kept)} kept, {len(out)-len(kept)} dropped")

# ----------------------------------------------------------------------- blind
def blind(args):
    cands = load_rows(args.inp)
    done = load(args.blind_out)
    rng = random.Random(args.seed)
    sample = rng.sample(cands, min(args.blind, len(cands)))
    todo = [r for r in sample if r["nct_id"] not in done]
    print(f"BLIND PASS — {len(todo)} of {len(sample)} left.")
    print("Both machine answers are hidden. Label from the document only.\n")

    for j, rec in enumerate(todo):
        gold = {"conditions": [], "interventions": [], "intervention_types": [],
                "allocation": None}
        while True:
            print(f"\n{BOLD}=== blind {j+1}/{len(todo)}  {rec['nct_id']} ==={RESET}")
            show_doc(rec["document"], [])          # no highlights — no hints
            for k, f in enumerate(FIELDS, 1):
                print(f"  {k} {f:<18}{gold[f]}")
            print(f"\n{DIM}types: {' '.join(C.INTERVENTION_TYPES)}{RESET}")
            cmd = input(f"1-{len(FIELDS)}=set  <enter>=done with this record  "
                        f"q=quit > ").strip().lower()
            if cmd == "":
                done[rec["nct_id"]] = {"nct_id": rec["nct_id"], "gold": gold}
                break
            if cmd == "q":
                save(args.blind_out, list(done.values()))
                return
            if cmd in {str(x) for x in range(1, len(FIELDS) + 1)}:
                f = FIELDS[int(cmd) - 1]
                gold = edit_field(gold, f)
        save(args.blind_out, list(done.values()))
    print(f"\nsaved {args.blind_out}: {len(done)} records")

# ----------------------------------------------------------------------- kappa
def cohens_kappa(pairs):
    """pairs = [(rater_a_label, rater_b_label)]. Labels must be hashable."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca[k] / n) * (cb.get(k, 0) / n) for k in ca)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)

def kappa(args):
    mine = load(args.blind_out)
    drafts = load(args.drafts)
    if not mine:
        sys.exit(f"{args.blind_out} not found — run --blind N first.")
    ids = [i for i in mine if i in drafts and "gold" in drafts[i]]
    if not ids:
        sys.exit("No overlap between your blind labels and the drafts.")

    print(f"n = {len(ids)} records labelled both by you and by Claude\n")
    print(f"{'field':<20}{'metric':<12}{'agree':>7}{'value':>8}   interpretation")
    print("-" * 76)

    skewed = []
    for f in ["allocation"]:
        pairs = [(mine[i]["gold"][f] or "NULL", drafts[i]["gold"][f] or "NULL")
                 for i in ids]
        k = cohens_kappa(pairs)
        po = sum(1 for a, b in pairs if a == b) / len(pairs)
        # Kappa paradox: when one label dominates, chance agreement is already
        # high and kappa collapses even though raters agree. Report both, always.
        if po >= 0.80 and k < 0.40:
            note = "κ suppressed by skewed marginals — quote agreement too"
            skewed.append(f)
        else:
            note = ("substantial" if k >= 0.61 else "moderate" if k >= 0.41 else
                    "fair" if k >= 0.21 else "poor — the task is ambiguous")
        print(f"{f:<20}{'Cohen κ':<12}{po:>6.0%}{k:>8.2f}   {note}")
        dist = Counter(a for a, _ in pairs)
        print(f"{'':20}{DIM_}your label distribution: "
              f"{dict(dist)}{RESET_}")

    # Span/set fields: kappa is not meaningful over open vocabularies, so report
    # mean pairwise F1 instead. Say so in the card rather than fudging a kappa.
    for f in ["conditions", "interventions", "intervention_types"]:
        fn = C.set_f1 if f == "intervention_types" else C.list_f1
        vals = [fn(mine[i]["gold"][f], drafts[i]["gold"][f]) for i in ids]
        m = sum(vals) / len(vals)
        exact = sum(1 for v in vals if v >= 0.999) / len(vals)
        print(f"{f:<20}{'mean F1':<16}{m:>8.2f}   exact agreement {exact:.0%}")

    print("\nκ ≥ 0.61 on the enums means your labels are defensible. Report these\n"
          "numbers in the dataset card, with n and the fact that labels were\n"
          "model-drafted and human-adjudicated.")
    if skewed:
        print(f"\nNote on {', '.join(skewed)}: nearly every record carries the same\n"
              "label, so chance agreement is already high and κ understates how much\n"
              "you and the model actually agree. Quote raw agreement alongside κ and\n"
              "say why — this is a known property of κ, not a defect in your labels.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="eval_candidates.jsonl")
    ap.add_argument("--drafts", default="eval_drafts.jsonl")
    ap.add_argument("--drafts-b", default=None,
                    help="a second independent drafting run. When given, records "
                         "are triaged by whether the two runs agree with each "
                         "other rather than against the registry derivation, "
                         "which is unreliable (its conditions field is often the "
                         "study title or an unrelated term).")
    ap.add_argument("--out", default="eval_gold.jsonl")
    ap.add_argument("--blind", type=int, default=0)
    ap.add_argument("--blind-out", default="eval_blind.jsonl")
    ap.add_argument("--kappa", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--agree-at", type=float, default=None,
                    help="similarity above which two labels count as agreeing")
    args = ap.parse_args()

    if args.agree_at is not None:
        globals()["AGREE_AT"] = args.agree_at
    if args.kappa:
        kappa(args)
    elif args.blind:
        blind(args)
    else:
        adjudicate(args)

if __name__ == "__main__":
    main()
