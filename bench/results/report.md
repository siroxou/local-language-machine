# Fine-tuning across model size and family

Dataset: `datasets/kestrel` — a fully invented domain, so base accuracy is chance by construction.
Shared config: {"iters":800,"learningRate":0.00001,"loraRank":16,"batchSize":4,"maxSeqLen":256,"numLayers":-1,"warmupSteps":20,"seed":42,"loraScale":20}

## Primary endpoint — MCQ on MEM (trained facts, unseen phrasing)

| model | family | params (B) | base | tuned | gain | n |
|---|---|---:|---:|---:|---:|---:|
| qwen2.5-0.5b | Qwen2.5 | 0.494 | 30% | 100% | 70pt | 126 |
| qwen2.5-3b | Qwen2.5 | 3.09 | 26% | 98% | 72pt | 126 |
| qwen2.5-7b | Qwen2.5 | 7.62 | 25% | 100% | 75pt | 126 |
| smollm2-135m | SmolLM2 | 0.135 | 21% | 50% | 29pt | 126 |
| smollm2-360m | SmolLM2 | 0.362 | 25% | 72% | 48pt | 126 |
| smollm2-1.7b | SmolLM2 | 1.711 | 24% | 94% | 71pt | 126 |
| falcon3-1b | Falcon3 | 1.669 | 17% | 83% | 66pt | 126 |
| falcon3-3b | Falcon3 | 3.228 | 22% | 90% | 68pt | 126 |
| falcon3-7b | Falcon3 | 7.456 | 18% | 100% | 82pt | 126 |
| smollm2-135m-s2 | SmolLM2 | 0.135 | 21% | 40% | 20pt | 126 |
| smollm2-135m-s3 | SmolLM2 | 0.135 | 21% | 46% | 25pt | 126 |

## All suites (tuned MCQ, gain over base)

| model | mem | para | comp | unseen |
|---|---:|---:|---:|---:|
| qwen2.5-0.5b | 100% (70pt) | 100% (71pt) | 17% (-25pt) | 50% (25pt) |
| qwen2.5-3b | 98% (72pt) | 99% (75pt) | 17% (-8pt) | 20% (-5pt) |
| qwen2.5-7b | 100% (75pt) | 100% (72pt) | 21% (-17pt) | 35% (15pt) |
| smollm2-135m | 50% (29pt) | 50% (28pt) | 21% (-12pt) | 15% (-5pt) |
| smollm2-360m | 72% (48pt) | 80% (60pt) | 17% (-17pt) | 30% (0pt) |
| smollm2-1.7b | 94% (71pt) | 95% (73pt) | 21% (-12pt) | 40% (15pt) |
| falcon3-1b | 83% (66pt) | 86% (63pt) | 17% (-17pt) | 30% (10pt) |
| falcon3-3b | 90% (68pt) | 90% (64pt) | 17% (-17pt) | 40% (25pt) |
| falcon3-7b | 100% (82pt) | 99% (81pt) | 29% (-17pt) | 30% (5pt) |
| smollm2-135m-s2 | 40% (20pt) | 34% (12pt) | 13% (-21pt) | 5% (-15pt) |
| smollm2-135m-s3 | 46% (25pt) | 47% (25pt) | 8% (-25pt) | 30% (10pt) |

## Cost and fit

| model | train (s) | best iter | best val | adapter (MB) | bits/byte base→tuned |
|---|---:|---:|---:|---:|---:|
| qwen2.5-0.5b | 68 | final | — | 33.6 | 1.397 → 0.006 |
| qwen2.5-3b | 276 | final | — | 114.24 | 2.727 → 0.006 |
| qwen2.5-7b | 428 | 650 | 0.002 | 154.04 | 2.801 → 0.003 |
| smollm2-135m | 40 | final | — | 18.68 | 1.400 → 0.083 |
| smollm2-360m | 56 | final | — | 33.17 | 1.287 → 0.053 |
| smollm2-1.7b | 124 | final | — | 69.03 | 1.541 → 0.014 |
| falcon3-1b | 92 | final | — | 49.53 | 1.422 → 0.032 |
| falcon3-3b | 164 | 750 | 0.051 | 77.03 | 1.704 → 0.026 |
| falcon3-7b | 360 | 750 | 0.005 | 168.92 | 1.516 → 0.006 |
| smollm2-135m-s2 | 44 | 750 | 0.135 | 18.68 | 1.400 → 0.082 |
| smollm2-135m-s3 | 44 | final | — | 18.68 | 1.400 → 0.086 |

## Run-to-run noise floor

Same model and data, 3 seeds: MCQ-MEM 50%, 40%, 46% — SD 4.8pt.

**Treat any cross-model gap smaller than ~10pt as noise.**

## Reading these numbers

- **MCQ is the headline.** It is teacher-forced, so it cannot be skewed by how verbose a model is. Free-text accuracy is reported too but is verbosity-sensitive.
- **Per-token perplexity is NOT comparable across families** — it depends on the tokenizer. Bits-per-byte uses the same denominator for every model and is the safe cross-model likelihood number.
- **UNSEEN measures hallucination, not knowledge.** Those compounds were never trained, so accuracy there is chance; what matters is whether the abstain rate collapsed after tuning.
- **COMP is composition** — compound→class and class→antidote are trained separately and the direct link never appears in training.
- **Size is confounded with LR-optimality.** mlx applies the LoRA `scale` as a raw multiplier and inits `lora_a` with 1/√input_dims, so one fixed learning rate is not equally good across a 55× width span.
- **Families are only cleanly comparable Qwen2.5 ↔ Falcon3**; SmolLM2 tops out at 1.7B.
- **Training time is serial and thermal** — later cells ran on a hotter machine.
