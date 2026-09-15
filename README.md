# Post-Training Ablations on a Specialist Corpus

Four post-training stages (CPT, SFT, DPO, RLVR) run end to end on one small model
and one corpus, measured through a single eval harness.

**Task:** read a clinical trial description, emit a validated structured record
(conditions, interventions, intervention types, allocation).
**Corpus:** ClinicalTrials.gov, 9,869 completed interventional studies, 6.7M tokens.
**Base model:** SmolLM2-360M. 
**Hardware:** single RTX 5070 Ti, 16GB.

---

## Results so far

### Extraction quality (Stage 00 eval harness, n=100 held out)

| Row | field_f1 | recall | json parse | notes |
|---|---|---|---|---|
| Human ceiling | 0.85 | | | two raters, same written rules, n=15 blind |
| **Base model (Run 0)** | **0.157 ± 0.019** | 0.332 | 0.510 | untouched SmolLM2-360M |
| Constant baseline | 0.26 | | | most common answer per field, no model |
| Empty baseline | 0.16 | | | empty record every time |

The base model scores below a constant baseline. That is not a bug. It produces
malformed output that costs precision, where a constant at least gets the base
rates right.

### Language modelling (Stage 01, CPT)

All fp32, 1024-token packed blocks, identical harness before and after.

| Model | Domain PPL | Domain bits/char | General PPL (WikiText-2) |
|---|---|---|---|
| SmolLM2-360M base | 10.51 | 1.639 | 13.12 |
| **SmolLM2-360M after CPT** | **8.83** | **1.580** | 14.35 |
| Qwen3-4B-Instruct (reference) | 8.89 | 1.573 | not measured |
| | **-16.0%** | | **+9.4%** |

Note: 

1. One epoch of CPT on 6.7M tokens moved domain perplexity down 16% and cost 9.4%
on general text. That trade-off is what CPT is. 

2. After CPT the 360M model matches Qwen3-4B-Instruct on domain perplexity, 1.580
against 1.573 bits per character. A model roughly 10x smaller, on this corpus,
because the corpus is narrow and the bigger model has no particular advantage in
it.

