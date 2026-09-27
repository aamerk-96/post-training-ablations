"""
Shared normalization, matching and scoring for the ClinicalTrials.gov extraction
task. Imported by build_dataset.py and by the inspect-ai scorers, so the eval
and the data are guaranteed to use identical rules.

LOCKED DECISIONS (Stage 00, Sept 2026)
--------------------------------------
Document   = briefTitle + briefSummary + detailedDescription, truncated to
             DOC_CHAR_CAP. Eligibility criteria excluded: long boilerplate that
             inflated match rates without carrying the target fields.
Gold        is derived from the TRUNCATED document, never from the full record.
            If a value isn't in the text the model sees, the correct answer is
            null / absent — not the registry's value.
Matching    is token-overlap, not string equality. Measured headroom of +20 to
            +36 points over exact match on 25 records: the registry's canonical
            forms ("Remdesivir 100mg IV") differ from prose ("remdesivir").
Schema      4 keys, always all 4 present in the output.
"""
import re

DOC_CHAR_CAP = 2500          # keeps doc + prompt + answer inside a 2k-token budget
OVERLAP_THRESHOLD = 0.5      # fraction of a value's content words that must appear

SCHEMA_KEYS = ["conditions", "interventions", "intervention_types", "allocation"]

# `phase` was cut from the schema entirely. On the real eval split it is null
# 91% of the time, so a model that always answers null banked 0.91 on it for
# free; that near-constant inflated every checkpoint equally, compressed the
# range the headline could move in, and cost a labelling slot per record for
# almost no information. `allocation` stays: 42% of records carry a real value,
# and it is the only field that tests abstention — whether the model invents a
# design detail the document never states.
HEADLINE_FIELDS = list(SCHEMA_KEYS)
DIAGNOSTIC_FIELDS = []

# A trial can mix modalities — surgery + device + behavioural in one study — so a
# single intervention_type has no correct answer and would inject unlearnable
# noise. It is a SET, scored as set-F1.
INTERVENTION_TYPES = ["DRUG", "DEVICE", "BIOLOGICAL", "PROCEDURE", "RADIATION",
                      "BEHAVIORAL", "GENETIC", "DIETARY_SUPPLEMENT",
                      "COMBINATION_PRODUCT", "DIAGNOSTIC_TEST", "OTHER"]

# ---------------------------------------------------------------- normalization
_WS = re.compile(r"\s+")
# Deliberately minimal. Every word removed here is a word the scorer can no
# longer tell apart, so "Type 2 Diabetes" vs "Type 1 Diabetes" must survive.
STOP = {"the", "a", "an", "of", "and", "or", "in", "for", "with", "to", "on",
        "study", "trial", "patients", "subjects"}

def norm(s):
    s = (s or "").lower().replace("–", "-").replace("—", "-").replace("'", "'")
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)      # 1,062 -> 1062
    s = re.sub(r"(?<=\d)\.(?=\d)", "\x01", s)    # protect decimals: 0.5
    # Hyphens and underscores SPLIT. The registry writes "Non-Small-Cell" and
    # "NON_RANDOMIZED"; prose writes "non-small cell" and "non randomized".
    # Keeping them joined makes those one unmatchable token — this was a real
    # bug that dropped 700/702 records in testing.
    s = re.sub(r"[^\w\s]", " ", s).replace("_", " ")
    s = s.replace("\x01", ".")
    return _WS.sub(" ", s).strip()

def content_tokens(s):
    return [t for t in norm(s).split()
            if t not in STOP and (len(t) > 2 or t.isdigit())]

# ------------------------------------------------------------- enum surface forms
# Registry enums are SCREAMING_SNAKE codes no human writes. To ask "is this stated
# in the document?" we match the forms a person would actually type.
ALIASES = {
    "PHASE1": ["phase 1", "phase i", "phase-1"],
    "PHASE2": ["phase 2", "phase ii", "phase-2"],
    "PHASE3": ["phase 3", "phase iii", "phase-3"],
    "PHASE4": ["phase 4", "phase iv", "phase-4"],
    "EARLY_PHASE1": ["early phase 1", "phase 0"],
    "RANDOMIZED": ["randomized", "randomised", "randomly assigned",
                   "randomization", "randomisation"],
    "NON_RANDOMIZED": ["non-randomized", "nonrandomized", "non-randomised",
                       "non randomized"],
}
DEGENERATE = {"ALL", "NA", "N/A", "", "OTHER", "NONE"}

def enum_present(code, doc_norm):
    """Is this enum's concept stated in the document? None = not applicable."""
    if code is None or str(code).upper() in DEGENERATE:
        return None
    for alias in ALIASES.get(str(code).upper(), [str(code)]):
        a = norm(alias)
        if a and re.search(rf"(?<!\w){re.escape(a)}(?!\w)", doc_norm):
            return True
    return False

def span_present(value, doc_norm, threshold=OVERLAP_THRESHOLD):
    """Token-overlap presence for a free-text value."""
    toks = content_tokens(str(value))
    if not toks:
        return False
    hit = sum(1 for t in toks if re.search(rf"(?<!\w){re.escape(t)}", doc_norm))
    return (hit / len(toks)) >= threshold

# ------------------------------------------------------------------- scoring
def token_f1(pred, gold):
    """Token-level F1 between two strings. The per-span metric."""
    p, g = set(content_tokens(str(pred))), set(content_tokens(str(gold)))
    if not p or not g:
        return float(p == g)
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)

def list_f1(preds, golds):
    """
    Micro-F1 over a list field. Greedy best-match pairing: each gold element is
    matched to its best unused prediction, credit is that pair's token_f1.
    Unmatched predictions cost precision; unmatched golds cost recall.
    """
    preds = [p for p in (preds or []) if str(p).strip()]
    golds = [g for g in (golds or []) if str(g).strip()]
    if not preds and not golds:
        return 1.0                      # correctly said "nothing here"
    if not preds or not golds:
        return 0.0
    used, credit = set(), 0.0
    for g in golds:
        best_i, best_s = None, 0.0
        for i, p in enumerate(preds):
            if i in used:
                continue
            s = token_f1(p, g)
            if s > best_s:
                best_i, best_s = i, s
        if best_i is not None and best_s > 0:
            used.add(best_i)
            credit += best_s
    prec, rec = credit / len(preds), credit / len(golds)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def enum_score(pred, gold):
    """Exact match on enums; null is a real answer and must be matched exactly."""
    if gold is None:
        return float(pred is None)
    if pred is None:
        return 0.0
    return float(str(pred).strip().upper() == str(gold).strip().upper())

def set_f1(preds, golds):
    """F1 over two sets of exact labels (the enum-set fields)."""
    p = {str(x).strip().upper() for x in (preds or []) if str(x).strip()}
    g = {str(x).strip().upper() for x in (golds or []) if str(x).strip()}
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)

def record_score(pred, gold):
    """Per-field scores for one record. Mean of these four is field_f1_strict.
    A key the model's output omits entirely scores 0 for that field — distinct
    from the key being present with an empty/null value."""
    pred = pred or {}
    return {
        "conditions":         list_f1(pred["conditions"], gold["conditions"])
                              if "conditions" in pred else 0.0,
        "interventions":      list_f1(pred["interventions"], gold["interventions"])
                              if "interventions" in pred else 0.0,
        "intervention_types": set_f1(pred["intervention_types"], gold["intervention_types"])
                              if "intervention_types" in pred else 0.0,
        "allocation":         enum_score(pred["allocation"], gold["allocation"])
                              if "allocation" in pred else 0.0,
    }

def headline(scores):
    """field_f1_strict: mean over the four discriminating fields."""
    return sum(scores[f] for f in HEADLINE_FIELDS) / len(HEADLINE_FIELDS)

# --------------------------------------------------------- document + gold build
def g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d

def build_document(study, cap=DOC_CHAR_CAP):
    p = study.get("protocolSection", {})
    parts = [g(p, "identificationModule", "briefTitle", default=""),
             g(p, "descriptionModule", "briefSummary", default=""),
             g(p, "descriptionModule", "detailedDescription", default="")]
    doc = "\n\n".join(x.strip() for x in parts if x and x.strip())
    doc = _WS.sub(" ", doc.replace("\n\n", "\x00")).replace("\x00", "\n\n").strip()
    if len(doc) > cap:                       # cut on a sentence boundary if we can
        cut = doc.rfind(". ", 0, cap)
        doc = doc[: cut + 1] if cut > cap * 0.6 else doc[:cap]
    return doc

def build_gold(study, doc):
    """
    Gold record, derived from the document the model will actually see.
    A value the text doesn't state becomes null (enums) or is omitted (lists).
    """
    p = study.get("protocolSection", {})
    dn = norm(doc)

    conds = [c for c in (g(p, "conditionsModule", "conditions", default=[]) or [])
             if span_present(c, dn)]
    ivs_all = g(p, "armsInterventionsModule", "interventions", default=[]) or []
    ivs = [iv for iv in ivs_all if iv.get("name") and span_present(iv["name"], dn)]

    di = g(p, "designModule", "designInfo", default={}) or {}
    alloc = di.get("allocation")
    phases = g(p, "designModule", "phases", default=[]) or []
    phase = phases[0] if phases else None

    # types of the interventions that survived — a set, deduped, order-independent.
    # Not presence-gated: the modality is inferable from the intervention's name.
    types = sorted({iv["type"] for iv in ivs if iv.get("type")})

    return {
        "conditions":         conds,
        "interventions":      [iv["name"] for iv in ivs],
        "intervention_types": types,
        "allocation":         alloc if enum_present(alloc, dn) else None,
        "phase":              phase if enum_present(phase, dn) else None,
    }

PROMPT = """Extract a structured record from the clinical trial description below.

Return ONLY a JSON object with exactly these four keys:
  "conditions"         list of strings — diseases/conditions studied, as the text names them
  "interventions"      list of strings — treatments/procedures tested, as the text names them
  "intervention_types" list of strings — one per distinct modality, from: {types}
  "allocation"         "RANDOMIZED" | "NON_RANDOMIZED" | null

LABELLING RULES (these resolve the ambiguities that cause rater disagreement):

1. Extract only what THIS text states. If the text does not state a value, use
   null (allocation) or leave the list empty.
2. Use the document's own wording, not canonical clinical vocabulary.
3. Use the MOST SPECIFIC phrase the document uses. "sub-acute stroke", not
   "stroke", if the document says sub-acute.
4. conditions = the disease or health state under study. If the study is on
   healthy volunteers and names no disease, return an empty list. Never put the
   study's title or aim in this field.
5. interventions = only what is administered to or performed on participants AS
   THE THING BEING STUDIED — the study arms. NOT measurements, assessments or
   data-collection procedures. Muscle biopsies, blood draws, calorimetry,
   questionnaires and imaging are excluded unless the study exists to evaluate
   that procedure itself.
6. Do not list brand or model variants separately unless the document presents
   them as separate arms being compared.
7. Strip leading verbs and articles: "using continuous positive airway pressure"
   becomes "continuous positive airway pressure".
8. Do not infer allocation from design conventions. "Open-label" is masking,
   not allocation. The word must be in the text.

Description:
{doc}

JSON:"""

def build_prompt(doc):
    return PROMPT.format(types=" ".join(INTERVENTION_TYPES), doc=doc)
