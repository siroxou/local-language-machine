# Fine-tuning across model size and family

Dataset: `datasets/kestrel` — a fully invented domain, so base accuracy is chance by construction.
Shared config: {"iters":200,"loraRank":16,"batchSize":4,"maxSeqLen":256,"numLayers":-1,"warmupSteps":20,"seed":42}

## Primary endpoint — MCQ on MEM (trained facts, unseen phrasing)

| model | family | params (B) | base | tuned | gain | n |
|---|---|---:|---:|---:|---:|---:|
| q05-lr2e4-s2 | Qwen2.5 | 0.494 | 30% | 39% | 9pt | 126 |
| q05-lr5e5-s20 | Qwen2.5 | 0.494 | 30% | 33% | 3pt | 126 |
| q05-lr1e5-s20 | Qwen2.5 | 0.494 | 30% | 51% | 21pt | 126 |
| q05-lr1e4-s2 | Qwen2.5 | 0.494 | 30% | 45% | 15pt | 126 |
| sm135-lr2e4-s2 | SmolLM2 | 0.135 | 21% | 36% | 15pt | 126 |
| sm135-lr1e4-s2 | SmolLM2 | 0.135 | 21% | 29% | 8pt | 126 |

## All suites (tuned MCQ, gain over base)

| model | mem |
|---|---:|
| q05-lr2e4-s2 | 39% (9pt) |
| q05-lr5e5-s20 | 33% (3pt) |
| q05-lr1e5-s20 | 51% (21pt) |
| q05-lr1e4-s2 | 45% (15pt) |
| sm135-lr2e4-s2 | 36% (15pt) |
| sm135-lr1e4-s2 | 29% (8pt) |

## Cost and fit

| model | train (s) | best iter | best val | adapter (MB) | bits/byte base→tuned |
|---|---:|---:|---:|---:|---:|
| q05-lr2e4-s2 | 102 | final | — | 33.6 | 1.397 → 0.110 |
| q05-lr5e5-s20 | 20 | final | — | 33.6 | 1.397 → 0.129 |
| q05-lr1e5-s20 | 20 | final | — | 33.6 | 1.397 → 0.092 |
| q05-lr1e4-s2 | 20 | final | — | 33.6 | 1.397 → 0.095 |
| sm135-lr2e4-s2 | 12 | final | — | 18.68 | 1.400 → 0.124 |
| sm135-lr1e4-s2 | 16 | final | — | 18.68 | 1.400 → 0.149 |

## Reading these numbers

- **MCQ is the headline.** It is teacher-forced, so it cannot be skewed by how verbose a model is. Free-text accuracy is reported too but is verbosity-sensitive.
- **Per-token perplexity is NOT comparable across families** — it depends on the tokenizer. Bits-per-byte uses the same denominator for every model and is the safe cross-model likelihood number.
- **UNSEEN measures hallucination, not knowledge.** Those compounds were never trained, so accuracy there is chance; what matters is whether the abstain rate collapsed after tuning.
- **COMP is composition** — compound→class and class→antidote are trained separately and the direct link never appears in training.
- **Size is confounded with LR-optimality.** mlx applies the LoRA `scale` as a raw multiplier and inits `lora_a` with 1/√input_dims, so one fixed learning rate is not equally good across a 55× width span.
- **Families are only cleanly comparable Qwen2.5 ↔ Falcon3**; SmolLM2 tops out at 1.7B.
- **Training time is serial and thermal** — later cells ran on a hotter machine.
