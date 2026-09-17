# Post-Training Ablations on a Specialist Corpus

## Goal

Learn the post-training toolchain by running each stage — continued pretraining,
supervised fine-tuning, preference optimisation, and RL with verifiable rewards —
on one model and one corpus, and scoring every stage through the same eval
harness.

The target task is structured extraction: read a clinical trial description and
emit a validated record of conditions, interventions, intervention types and
allocation.

```text
INPUT — trial description
┌──────────────────────────────────────────────────────────────────────┐
│ A randomized, double-blind study evaluating dapagliflozin 10 mg once │
│ daily versus matching placebo in adults with type 2 diabetes         │
│ mellitus and chronic kidney disease. Participants are assigned 1:1   │
│ for 24 weeks.                                                        │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                          SmolLM2-360M
                                  │
                                  ▼
OUTPUT — validated record
┌──────────────────────────────────────────────────────────────────────┐
│ {                                                                    │
│   "conditions":         ["Type 2 Diabetes Mellitus",                 │
│                          "Chronic Kidney Disease"],                  │
│   "interventions":      ["Dapagliflozin", "Placebo"],                │
│   "intervention_types": ["Drug", "Drug"],                            │
│   "allocation":         "Randomized"                                 │
│ }                                                                    │
└──────────────────────────────────────────────────────────────────────┘
```

The harness scores each field against the study's registered record, and
separately records whether the output parsed as valid JSON at all.

**Corpus.** ClinicalTrials.gov — 9,869 completed interventional studies,
6.7M tokens.

**Base model.** SmolLM2-360M.

**Hardware.** A single RTX 5070 Ti, 16GB.

---

## Work done and results

### Evaluation harness

Built before any training, along with a human ceiling and two reference
baselines, so later numbers have something to be measured against. Every stage is
scored on the same 100 held-out studies.

Metrics are per-field F1, recall, and the fraction of outputs that parse as valid
JSON. Format compliance is tracked separately from extraction quality, since at
this model size they fail independently and a single score hides which one a
stage fixed.

| Row | field F1 | recall | JSON parse | notes |
|---|---|---|---|---|
| Human ceiling | 0.85 | | | two raters, same written rules, n=15 blind |
| **Base model (Run 0)** | **0.157 ± 0.019** | 0.332 | 0.510 | untouched SmolLM2-360M |
| Constant baseline | 0.26 | | | most common answer per field, no model |
| Empty baseline | 0.16 | | | empty record every time |

The base model scores below the constant baseline. The cause is format rather
than knowledge: about half its outputs fail to parse, and malformed records cost
precision, while a constant at least reproduces the base rates.

### Stage 01 — continued pretraining

One epoch over the corpus in fp32, 1024-token packed blocks, with the same
harness applied before and after.

| Model | Domain PPL | Domain bits/char | General PPL (WikiText-2) |
|---|---|---|---|
| SmolLM2-360M base | 10.51 | 1.639 | 13.12 |
| **SmolLM2-360M after CPT** | **8.83** | **1.580** | 14.35 |
| Qwen3-4B-Instruct (reference) | 8.89 | 1.573 | not measured |
| | **−16.0%** | | **+9.4%** |

Domain perplexity fell 16.0% and general-text perplexity rose 9.4% — the
trade-off continued pretraining exists to make. Measuring both is what separates
specialisation from degradation.

After CPT the 360M model reaches 1.580 bits per character on the domain against
Qwen3-4B-Instruct's 1.573, a model roughly ten times larger. On a corpus this
narrow, scale confers little advantage.

### Environments

Each stage runs in its own virtual environment — `env-eval.sh`, `env-sft.sh`,
`env-rl.sh` — because the evaluation, fine-tuning and RL toolchains have
conflicting dependencies. The RL environment pins vLLM's attention backend and
sampler explicitly; the defaults are unstable on this hardware.

---

## Ongoing

The remaining stages are designed and configured but not yet run. Each is scored
on the same harness and logged to `RUNS.md` with its config, wall-clock time and
peak VRAM.

- **Stage 02 — supervised fine-tuning.** Teaches the output format directly.
  Given the base model's 0.510 JSON parse rate, this is the stage most likely to
  produce the largest single jump.
- **Stage 03 — preference optimisation (DPO).** Trained on pairs drawn from the
  fine-tuned model's own outputs, targeting the errors SFT leaves behind.
- **Stage 04 — RL with verifiable rewards (GRPO).** The extraction task has a
  programmatic correctness check, so the reward needs no judge model.

---

## Repository layout

```
configs/         per-stage run configuration
data/stage00/    held-out evaluation set and baselines
cpt.ipynb        Stage 01 continued pretraining
env-*.sh         per-stage environment activation
RUNS.md          run log: stage, config, score, wall-clock, peak VRAM
```
