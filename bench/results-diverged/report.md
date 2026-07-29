# Fine-tuning across model size and family

Dataset: `datasets/kestrel` — a fully invented domain, so base accuracy is chance by construction.
Shared config: {"iters":800,"learningRate":0.0002,"loraRank":16,"batchSize":4,"maxSeqLen":256,"numLayers":-1,"warmupSteps":20,"seed":42}

## Primary endpoint — MCQ on MEM (trained facts, unseen phrasing)

| model | family | params (B) | base | tuned | gain | n |
|---|---|---:|---:|---:|---:|---:|
| qwen2.5-0.5b | Qwen2.5 | 0.494 | 30% | 28% | -2pt | 126 |
| qwen2.5-3b | Qwen2.5 | 3.09 | 26% | 21% | -6pt | 126 |
| qwen2.5-7b | Qwen2.5 | 7.62 | 25% | 21% | -3pt | 126 |
| smollm2-135m | SmolLM2 | 0.135 | 21% | 99% | 79pt | 126 |
| smollm2-360m | SmolLM2 | 0.362 | 25% | 98% | 74pt | 126 |
| smollm2-1.7b | SmolLM2 | 1.711 | 24% | 95% | 71pt | 126 |
| falcon3-1b | Falcon3 | 1.669 | 17% | 33% | 16pt | 126 |
| falcon3-3b | Falcon3 | 3.228 | 22% | 29% | 7pt | 126 |
| falcon3-7b | Falcon3 | 7.456 | 18% | 25% | 7pt | 126 |
| smollm2-135m-s2 | SmolLM2 | 0.135 | 21% | 89% | 68pt | 126 |
| smollm2-135m-s3 | SmolLM2 | 0.135 | 21% | 95% | 75pt | 126 |

## All suites (tuned MCQ, gain over base)

| model | mem | para | comp | unseen |
|---|---:|---:|---:|---:|
| qwen2.5-0.5b | 28% (-2pt) | 34% (6pt) | 17% (-25pt) | 40% (15pt) |
| qwen2.5-3b | 21% (-6pt) | 25% (0pt) | 17% (-8pt) | 30% (5pt) |
| qwen2.5-7b | 21% (-3pt) | 20% (-8pt) | 17% (-21pt) | 50% (30pt) |
| smollm2-135m | 99% (79pt) | 99% (77pt) | 29% (-4pt) | 15% (-5pt) |
| smollm2-360m | 98% (74pt) | 96% (75pt) | 13% (-21pt) | 30% (0pt) |
| smollm2-1.7b | 95% (71pt) | 94% (71pt) | 8% (-25pt) | 35% (10pt) |
| falcon3-1b | 33% (16pt) | 31% (9pt) | 17% (-17pt) | 25% (5pt) |
| falcon3-3b | 29% (7pt) | 29% (2pt) | 33% (0pt) | 45% (30pt) |
| falcon3-7b | 25% (7pt) | 32% (14pt) | 17% (-29pt) | 25% (0pt) |
| smollm2-135m-s2 | 89% (68pt) | 89% (67pt) | 8% (-25pt) | 35% (15pt) |
| smollm2-135m-s3 | 95% (75pt) | 96% (74pt) | 8% (-25pt) | 45% (25pt) |

## Cost and fit

| model | train (s) | best iter | best val | adapter (MB) | bits/byte base→tuned |
|---|---:|---:|---:|---:|---:|
| qwen2.5-0.5b | 132 | final | — | 33.6 | 1.397 → 1.237 |
| qwen2.5-3b | 296 | final | — | 114.24 | 2.727 → 2.032 |
| qwen2.5-7b | 416 | final | — | 154.04 | 2.801 → 1.804 |
| smollm2-135m | 44 | 750 | 0.005 | 18.68 | 1.400 → 0.005 |
| smollm2-360m | 100 | final | — | 33.17 | 1.287 → 0.025 |
| smollm2-1.7b | 340 | final | — | 69.03 | 1.541 → 0.027 |
| falcon3-1b | 308 | final | — | 49.53 | 1.422 → 0.825 |
| falcon3-3b | 548 | final | — | 77.03 | 1.704 → 1.647 |
| falcon3-7b | 1240 | final | — | 168.92 | 1.516 → 1.953 |
| smollm2-135m-s2 | 40 | final | — | 18.68 | 1.400 → 0.018 |
| smollm2-135m-s3 | 4459 | final | — | 18.68 | 1.400 → 0.011 |

## Run-to-run noise floor

Same model and data, 3 seeds: MCQ-MEM 99%, 89%, 95% — SD 5.2pt.

**Treat any cross-model gap smaller than ~10pt as noise.**

## Reading these numbers

- **MCQ is the headline.** It is teacher-forced, so it cannot be skewed by how verbose a model is. Free-text accuracy is reported too but is verbosity-sensitive.
- **Per-token perplexity is NOT comparable across families** — it depends on the tokenizer. Bits-per-byte uses the same denominator for every model and is the safe cross-model likelihood number.
- **UNSEEN measures hallucination, not knowledge.** Those compounds were never trained, so accuracy there is chance; what matters is whether the abstain rate collapsed after tuning.
- **COMP is composition** — compound→class and class→antidote are trained separately and the direct link never appears in training.
- **Size is confounded with LR-optimality.** mlx applies the LoRA `scale` as a raw multiplier and inits `lora_a` with 1/√input_dims, so one fixed learning rate is not equally good across a 55× width span.
- **Families are only cleanly comparable Qwen2.5 ↔ Falcon3**; SmolLM2 tops out at 1.7B.
- **Training time is serial and thermal** — later cells ran on a hotter machine.
