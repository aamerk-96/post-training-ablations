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

CPT alone does not transfer to the extraction task — it isn't trained on the
output format, and its field F1 actually drops below Run 0 (0.043 vs. 0.157).
Domain fluency and task performance are separate things at this model size, and
Stage 02 is what closes the gap.

### Stage 02 — supervised fine-tuning

LoRA (r=64, alpha 16, lr 2e-4) over the SFT pool, 2 epochs, effective batch 16,
run from both the base model and the CPT checkpoint, scored on the same harness.

| Model | field F1 (strict) | field recall | JSON parse | schema complete |
|---|---|---|---|---|
| Base model (Run 0) | 0.157 ± 0.019 | 0.332 | 0.510 | 0.510 |
| SFT from base | 0.725 ± 0.020 | 0.758 ± 0.022 | 0.980 ± 0.014 | 0.980 ± 0.014 |
| **SFT from CPT** | **0.754 ± 0.020** | **0.810 ± 0.018** | 0.980 ± 0.014 | 0.980 ± 0.014 |

SFT is the stage that fixes format compliance (JSON parse rate 0.51 → 0.98),
as expected from Run 0's failure mode. Starting SFT from the CPT checkpoint
rather than the base model gives a further, consistent lift on every metric.
Full numbers, decoding-variant breakdowns, training cost and a RoPE
misconfiguration bug that invalidated an earlier round of runs are in
[`RUNS.md`](RUNS.md).

### Before / after: one held-out example

Same held-out trial description (`NCT06358482`), greedy decoding, run through
the untouched base model and the final CPT + SFT checkpoint.

> Electric Toothbrush Efficacy in Preventing Dental Plaque Development
>
> The present study will be the first to compared dental plaque quantitation
> between an oscillating-rotating electric toothbrush and a manual toothbrush
> with a new QLF technology. This is a randomized controlled trial of 12-weeks
> duration designed to assess whether a new commercially available
> oscillating-rotating electric toothbrush can objectively improve dental
> plaque parameters obtained via light induced fluorescence (QLF) technology.
> The study will be conducted in a dental clinic in Italy.

**Base model (Run 0)** — never terminates the object, hallucinates a disease and
an intervention that appear nowhere in the text, and then loops on the same
malformed shape until the token budget runs out:

```json
{
  "conditions": [
    "DISEASE": "DENTAL PLASMA",
    "INTERVENTION": "ELECTROCUTING",
    "INTERVENTION_TYPES": ["ELECTROCUTING"],
    "ALLOCATION": "NON_RANDOMIZED"
  ],
  "interventions": [ ... ],
  "allocation": "NON_RANDOMIZED"
}

JSON:
{
  "conditions": [
    "DISEASE": "DENTAL PLASMA",
    ...
  # repeats verbatim until generation is cut off
```

**CPT + SFT (`sft_cpt_rope`)** — well-formed, on-schema, and matches the study:

```json
{"conditions": ["dental plaque development"], "interventions": ["oscillating-rotating electric toothbrush", "manual toothbrush with a new QLF technology"], "intervention_types": ["DEVICE"], "allocation": "RANDOMIZED"}
```

**Gold:**

```json
{"conditions": ["dental plaque"], "interventions": ["oscillating-rotating electric toothbrush", "manual toothbrush"], "intervention_types": ["DEVICE"], "allocation": "RANDOMIZED"}
```

Full sample logs: `data/stage00/logs/` (base) and `logs/sft_cpt_rope/` (final).

### Environments

Each stage runs in its own virtual environment — `env-eval.sh`, `env-sft.sh`,
`env-rl.sh` — because the evaluation, fine-tuning and RL toolchains have
conflicting dependencies. The RL environment pins vLLM's attention backend and
sampler explicitly; the defaults are unstable on this hardware.

---

## Ongoing

Stages 01 and 02 are done and scored above. The remaining stages are designed
and configured but not yet run. Each is scored on the same harness and logged
to `RUNS.md` with its config, wall-clock time and peak VRAM.

- **Stage 03 — preference optimisation (DPO).** Trained on pairs drawn from the
  fine-tuned model's own outputs, targeting the errors SFT leaves behind.
- **Stage 04 — RL with verifiable rewards (GRPO).** The extraction task has a
  programmatic correctness check, so the reward needs no judge model.

---

## Repository layout

```
configs/         per-stage run configuration
data/stage00/    held-out evaluation set, baselines, dataset-build scripts
cpt.ipynb        Stage 01 continued pretraining
sft.ipynb        Stage 02 supervised fine-tuning (LoRA)
fp32.ipynb       merges an fp32 copy of the SFT-from-base adapter
env-*.sh         per-stage environment activation
RUNS.md          run log: stage, config, score, wall-clock, peak VRAM
```
