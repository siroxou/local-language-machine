#!/usr/bin/env python3
"""Held-out perplexity for an MLX model, with or without a LoRA adapter.

Used by eval.ts for the base-vs-fine-tuned benchmark. mlx_lm.lora --test always tries to
load an adapter (so it can't score the base model), so we load directly via mlx_lm.load
(adapter_path=None → base) and average the per-token cross-entropy over the test set.
Prints `Test loss X, Test ppl Y` — the same format eval.ts's parseTestLoss expects.
"""
import argparse, json, math
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load


def text_of(row, tok):
    if isinstance(row.get("text"), str):
        return row["text"]
    if isinstance(row.get("messages"), list):
        return tok.apply_chat_template(row["messages"], tokenize=False)
    if isinstance(row.get("prompt"), str) and isinstance(row.get("completion"), str):
        return row["prompt"] + row["completion"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--data", required=True)          # a .jsonl file
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=1024)
    a = ap.parse_args()

    model, tok = load(a.model, adapter_path=a.adapter_path)

    rows = []
    with open(a.data) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = rows[: a.limit]

    total_loss, total_tok = 0.0, 0
    for row in rows:
        t = text_of(row, tok)
        if not t:
            continue
        ids = tok.encode(t)[: a.max_len]
        if len(ids) < 2:
            continue
        x = mx.array(ids[:-1])[None]
        y = mx.array(ids[1:])[None]
        logits = model(x).astype(mx.float32)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)
        total_loss += float(loss)
        total_tok += y.size

    avg = total_loss / max(total_tok, 1)
    print(f"Test loss {avg:.4f}, Test ppl {math.exp(avg):.4f}")


if __name__ == "__main__":
    main()
