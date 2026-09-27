import json, random, sys, types
import ctg_common as C
import build_dataset as B

def mk(nct, title, summary, detail, conds, ivs, alloc, phase):
    return {"protocolSection": {
        "identificationModule": {"nctId": nct, "briefTitle": title},
        "descriptionModule": {"briefSummary": summary, "detailedDescription": detail},
        "conditionsModule": {"conditions": conds},
        "armsInterventionsModule": {"interventions": ivs},
        "designModule": {"studyType": "INTERVENTIONAL", "phases": [phase] if phase else [],
                         "designInfo": {"allocation": alloc}},
    }}

base_detail = ("This randomized, double-blind, placebo-controlled study evaluated the "
  "efficacy of remdesivir in adults hospitalized with non-small cell lung cancer. "
  "Participants received intravenous therapy daily for ten days. " * 6)

studies = []
for i in range(700):
    studies.append(mk(f"NCT{i:08d}", "A Study of Remdesivir in NSCLC",
        "Evaluating remdesivir for non-small cell lung cancer.", base_detail,
        ["Carcinoma, Non-Small-Cell Lung"],
        [{"name": "Remdesivir 100mg IV", "type": "DRUG"}],
        "RANDOMIZED", "PHASE3" if i % 3 else None))
# one where phase IS stated
studies.append(mk("NCT99999999", "Phase 3 Aspirin Trial",
    "A phase 3 randomized trial of aspirin in hypertension.", base_detail.replace("remdesivir","aspirin"),
    ["Hypertension"], [{"name":"Aspirin","type":"DRUG"}], "RANDOMIZED", "PHASE3"))
# one that should be DROPPED (nothing extractable)
studies.append(mk("NCT88888888", "Behavioural Programme",
    "A counselling programme.", "Weekly sessions were delivered. " * 40,
    ["Tobacco Use Disorder"], [{"name":"Motivational Interviewing","type":"BEHAVIORAL"}],
    None, None))

B.fetch = lambda n, max_pages=80: studies[:n]
sys.argv = ["build_dataset.py", "--n", "702", "--n-eval", "20", "--n-sft", "20"]
B.main()

print("\n=== gold sanity ===")
row = json.loads(open("eval_candidates.jsonl").readline())
print(json.dumps(row["gold"], indent=1))
print("doc chars:", len(row["document"]))

print("\n=== scorer sanity ===")
gold = {"conditions":["Carcinoma, Non-Small-Cell Lung"],"interventions":["Remdesivir 100mg IV"],
        "intervention_types":["DRUG"],"allocation":"RANDOMIZED","phase":None}
cases = {
 "perfect":       {"conditions":["Carcinoma, Non-Small-Cell Lung"],"interventions":["Remdesivir 100mg IV"],"intervention_types":["DRUG"],"allocation":"RANDOMIZED","phase":None},
 "prose variant": {"conditions":["non small cell lung cancer"],"interventions":["remdesivir"],"intervention_types":["DRUG"],"allocation":"RANDOMIZED","phase":None},
 "hallucinated phase": {"conditions":["Carcinoma, Non-Small-Cell Lung"],"interventions":["Remdesivir 100mg IV"],"intervention_types":["DRUG"],"allocation":"RANDOMIZED","phase":"PHASE3"},
 "empty":         {"conditions":[],"interventions":[],"intervention_types":[],"allocation":None,"phase":None},
}
for name, pred in cases.items():
    s = C.record_score(pred, gold)
    print(f"{name:<20} mean={sum(s.values())/5:.2f}  {[f'{k}={v:.2f}' for k,v in s.items()]}")
