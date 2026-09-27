#!/usr/bin/env python3
"""
Stage 00 / step 2 — build the dataset.

Produces three disjoint splits from ClinicalTrials.gov, by NCT id, with a fixed
seed. Disjoint-by-id is the contamination control: no study whose text is in the
CPT corpus may appear in the eval set, or Stage 01 would be training on the test.

  eval_candidates.jsonl   100  doc + gold, for you to hand-verify -> the held-out set
  sft_pool.jsonl          400  doc + gold, source for Stage 02 training pairs
  cpt_corpus.jsonl       rest  raw text only, no labels, for Stage 01
  manifest.json                counts, seed, id lists, field statistics

    python build_dataset.py --n 1500
"""
import argparse, json, os, random, sys, urllib.error, urllib.parse, urllib.request
from collections import Counter

import ctg_common as C

API = "https://clinicaltrials.gov/api/v2/studies"
SEED = 20260906

def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "stage00-dataset"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        print(f"\nHTTP {e.code} for {url}\n{e.read().decode('utf-8','replace')[:800]}",
              file=sys.stderr)
        raise

def fetch(n, max_pages=None):
    # Roughly 55% of completed trials survive the interventional +
    # has-detailed-description filter, so reaching n usable records needs about
    # 2x that many pages. A fixed cap of 80 silently truncated a 15k request to
    # 4.2k. Scale it with n and leave headroom.
    if max_pages is None:
        max_pages = max(80, (n // 100) * 3 + 40)
    out, seen, token, pages = [], set(), None, 0
    while len(out) < n and pages < max_pages:
        q = {"pageSize": 100, "filter.overallStatus": "COMPLETED"}
        if token:
            q["pageToken"] = token
        d = _get(f"{API}?{urllib.parse.urlencode(q)}")
        pages += 1
        for s in d.get("studies", []):
            p = s.get("protocolSection", {})
            nct = C.g(p, "identificationModule", "nctId")
            if not nct or nct in seen:
                continue
            if C.g(p, "designModule", "studyType") != "INTERVENTIONAL":
                continue
            if not C.g(p, "descriptionModule", "detailedDescription"):
                continue
            seen.add(nct)
            out.append(s)
        token = d.get("nextPageToken")
        if pages % 10 == 0 or not token:
            print(f"  page {pages}/{max_pages}: {len(out)} usable studies",
                  file=sys.stderr)
        if not token:
            break
    return out[:n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--n-sft", type=int, default=2000)
    ap.add_argument("--freeze-eval", default="manifest.json",
                    help="reuse the eval NCT ids from this manifest, so growing "
                         "the corpus never reshuffles the set you hand-labelled")
    args = ap.parse_args()

    # An eval set is frozen the moment you start labelling it. If a manifest
    # already names 100 studies, those stay the eval split no matter how much
    # bigger the pull gets; everything else is reshuffled around them.
    frozen = []
    if args.freeze_eval and os.path.exists(args.freeze_eval):
        frozen = json.load(open(args.freeze_eval)).get("eval_nct_ids", [])
        if frozen:
            print(f"freezing eval split: {len(frozen)} ids from {args.freeze_eval}",
                  file=sys.stderr)

    studies = fetch(args.n)
    print(f"\nfetched {len(studies)} studies", file=sys.stderr)
    if len(studies) < args.n * 0.9:
        print(f"  note: asked for {args.n}, the registry ran out of matching "
              f"studies at {len(studies)}", file=sys.stderr)

    records, dropped = [], Counter()
    for s in studies:
        nct = C.g(s, "protocolSection", "identificationModule", "nctId")
        doc = C.build_document(s)
        if len(doc) < 500:
            dropped["doc too short"] += 1
            continue
        gold = C.build_gold(s, doc)
        # a usable example must have at least one condition AND one intervention
        # actually stated in the text; otherwise there is nothing to extract
        if not gold["conditions"] or not gold["interventions"]:
            dropped["no extractable condition/intervention"] += 1
            continue
        # CPT gets the untruncated text: there is no gold to align to and no
        # prompt to leave room for, so capping it throws away training tokens.
        records.append({"nct_id": nct, "document": doc, "gold": gold,
                        "full_text": C.build_document(s, cap=10**9)})

    print(f"usable: {len(records)}", file=sys.stderr)
    for k, v in dropped.items():
        print(f"  dropped, {k}: {v}", file=sys.stderr)

    rng = random.Random(SEED)
    rng.shuffle(records)

    if frozen:
        keep = set(frozen)
        ev = [r for r in records if r["nct_id"] in keep]
        rest = [r for r in records if r["nct_id"] not in keep]
        missing = len(keep) - len(ev)
        if missing:
            print(f"  warning: {missing} frozen eval ids were not in this pull; "
                  f"topping up from the pool", file=sys.stderr)
            ev += rest[:missing]
            rest = rest[missing:]
    else:
        ev, rest = records[: args.n_eval], records[args.n_eval:]

    sft = rest[: args.n_sft]
    cpt = rest[args.n_sft:]

    def dump(path, rows, keys):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in keys}) + "\n")
        print(f"wrote {path:<26}{len(rows):>5} records")

    dump("eval_candidates.jsonl", ev,  ["nct_id", "document", "gold"])
    dump("sft_pool.jsonl",        sft, ["nct_id", "document", "gold"])
    with open("cpt_corpus.jsonl", "w") as f:                     # no labels
        for r in cpt:
            f.write(json.dumps({"nct_id": r["nct_id"],
                                "text": r["full_text"]}) + "\n")
    toks = sum(len(r["full_text"]) for r in cpt) / 4              # ~4 chars/token
    print(f"wrote {'cpt_corpus.jsonl':<26}{len(cpt):>5} records"
          f"  (~{toks/1e6:.1f}M tokens, untruncated)")

    # ---- statistics you need before hand-verifying -------------------------
    def stats(rows):
        n = len(rows) or 1
        nulls = {k: sum(1 for r in rows if r["gold"].get(k) is None)
                 for k in C.SCHEMA_KEYS
                 if not isinstance(rows[0]["gold"].get(k), list)} if rows else {}
        return {
            "n": len(rows),
            "avg_doc_chars": round(sum(len(r["document"]) for r in rows) / n),
            "avg_conditions": round(sum(len(r["gold"]["conditions"]) for r in rows) / n, 2),
            "avg_interventions": round(sum(len(r["gold"]["interventions"]) for r in rows) / n, 2),
            "avg_intervention_types": round(sum(len(r["gold"]["intervention_types"]) for r in rows) / n, 2),
            "null_rate": {k: round(v / n, 2) for k, v in nulls.items()},
        }

    manifest = {
        "seed": SEED,
        "source": "ClinicalTrials.gov API v2, overallStatus=COMPLETED, INTERVENTIONAL",
        "doc_char_cap": C.DOC_CHAR_CAP,
        "overlap_threshold": C.OVERLAP_THRESHOLD,
        "schema_keys": C.SCHEMA_KEYS,
        "splits": {"eval": stats(ev), "sft_pool": stats(sft), "cpt_corpus": stats(cpt)},
        "eval_nct_ids": [r["nct_id"] for r in ev],
        "disjoint": {
            "eval_x_cpt": len(set(r["nct_id"] for r in ev) &
                              set(r["nct_id"] for r in cpt)),
            "eval_x_sft": len(set(r["nct_id"] for r in ev) &
                              set(r["nct_id"] for r in sft)),
        },
    }
    json.dump(manifest, open("manifest.json", "w"), indent=2)

    print("\n--- eval split ---")
    for k, v in manifest["splits"]["eval"].items():
        print(f"  {k}: {v}")
    print(f"\ncontamination check (must both be 0): {manifest['disjoint']}")
    print("\nwrote manifest.json")
    print("\nNext: python draft_labels.py --out eval_drafts.jsonl")

if __name__ == "__main__":
    main()
