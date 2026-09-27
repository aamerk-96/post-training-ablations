# Runs

All scores are from `eval/eval_task.py` against the frozen 100-study held-out
set (`data/stage00/eval_gold.jsonl`), reported as field F1 (strict), field
recall, JSON parse rate and schema-complete rate.

## Headline results

Fixed RoPE config (`rope_theta=100000`), fixed scorer. These are the reported
numbers.

| run | log | field F1 (strict) | field recall | JSON parse | schema complete |
|---|---|---|---|---|---|
| Base model, greedy (Run 0) | `data/stage00/logs` (`19-15-09`) | 0.157 ± 0.019 | 0.332 | 0.510 | 0.510 |
| CPT, epoch 1 | `logs/cpt_e1_rope` | 0.043 ± 0.012 | 0.278 ± 0.028 | 0.150 ± 0.036 | 0.150 ± 0.036 |
| SFT from base | `logs/sft_base_rope` | 0.725 ± 0.020 | 0.758 ± 0.022 | 0.980 ± 0.014 | 0.980 ± 0.014 |
| **SFT from CPT** | `logs/sft_cpt_rope` | **0.754 ± 0.020** | **0.810 ± 0.018** | 0.980 ± 0.014 | 0.980 ± 0.014 |

CPT alone regresses extraction quality (it isn't trained on the task format),
but once SFT is layered on top, starting from the CPT checkpoint beats starting
from the base model on every metric.

### Run 0 decoding variants

Not headline numbers — kept to show sensitivity to decoding strategy.

| variant | field F1 (strict) | field recall | JSON parse | schema complete |
|---|---|---|---|---|
| pass@4, T=0.7, max reducer | 0.150 ± 0.021 | 0.488 | 0.570 | 0.510 |
| mean@4, T=0.7, mean reducer | 0.109 ± 0.016 | 0.388 | 0.482 | 0.448 |
| stop sequences, T=0 | 0.157 ± 0.019 | 0.298 | 0.510 | 0.510 |

### Training cost

| run | val loss (start → end) | steps | wall-clock | peak VRAM |
|---|---|---|---|---|
| SFT from base | 0.906 → 0.200 | 238 | 1,105 s | 13.21 GB |
| SFT from CPT | 1.0328 → 0.2000 | 238 | 646 s | 13.21 GB |

Both SFT runs: LoRA r=64, alpha 16, lr 2e-4, effective batch 16, 2 epochs,
1,891 train examples.

### Superseded runs (broken RoPE config)

Early CPT/SFT runs merged with the base model's default RoPE settings instead
of the corrected `rope_theta=100000`, which silently degraded long-context
behaviour. Kept as a bug record, not as reported results — superseded by the
`*_rope` runs above.

| run | log | field F1 (strict) | field recall | JSON parse | schema complete |
|---|---|---|---|---|---|
| CPT, epoch 1 | `logs/cpt_e1` | 0.000 | 0.294 ± 0.030 | 0.000 | 0.000 |
| SFT from base, bf16, old scorer | `logs/sft_base` | 0.550 ± 0.026 | 0.623 ± 0.023 | 0.930 ± 0.026 | 0.440 ± 0.050 |
| SFT from base, bf16, fixed scorer | `logs/sft_base` | 0.48 | 0.623 | 0.930 | 0.440 |
| SFT from base, fp32 | `logs/sft_base_fp32` | 0.504 ± 0.023 | 0.648 ± 0.022 | 0.930 ± 0.026 | 0.350 ± 0.048 |
| SFT from CPT | `logs/sft_cpt` | 0.380 ± 0.026 | 0.668 ± 0.021 | 0.790 ± 0.041 | 0.020 ± 0.014 |

The re-scored `sft_base` row was re-run against a fixed scorer version; it
printed without a stderr so none is reported for that row.

---

## Next

- **Stage 03 — preference optimisation (DPO).** Not yet run.
- **Stage 04 — RL with verifiable rewards (GRPO).** Not yet run.
