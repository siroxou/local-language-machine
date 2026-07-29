#!/usr/bin/env python3
"""Held-out scoring for an MLX model, with or without a LoRA adapter.

One process per model variant — loading the model is the expensive part, so perplexity, MCQ and
generation all happen behind a single `load()`. Used by eval.ts for the base-vs-fine-tuned
benchmark (`--adapter-path` absent => base, present => tuned).

Emits one `EVAL_JSON {...}` line. A JSON payload survives newlines, quotes and stray `=` in
generated text; the previous `==========` delimiter scraping did not.

Three things here are deliberate and load-bearing:

1. Tokenisation mirrors `mlx_lm.tuner.datasets` EXACTLY — `apply_chat_template(..., return_dict=
   False)` for the full sequence and a second call with `add_generation_prompt` for the offset.
   Do NOT use `apply_chat_template(tokenize=False)` + `tok.encode(...)`: TokenizerWrapper has no
   own `encode`, so that path falls through to HF with `add_special_tokens=True` and prepends a
   SECOND BOS for families whose template already emits one (Llama/Gemma yes, Qwen no). That
   inflates loss for some families and not others — a silent cross-family confound.
2. Loss is scored on completion tokens only, matching `mask_prompt: true` in training.
3. Bits-per-byte is reported alongside perplexity. Per-token perplexity is tokenizer-dependent
   and must never be compared across model families; bits-per-byte has the same denominator for
   every model, so it is the only likelihood number that is safe cross-model.
"""
import argparse, json, math, sys
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, batch_generate

LN2 = math.log(2)


def to_messages(row):
    """Normalise a supported row to the message list mlx trains on. None => free `text` row."""
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        return msgs
    p, c = row.get("prompt"), row.get("completion")
    if isinstance(p, str) and isinstance(c, str):
        return [{"role": "user", "content": p}, {"role": "assistant", "content": c}]
    return None


def tokenize_row(tok, row):
    """(ids, offset) exactly as mlx_lm.tuner.datasets produces with mask_prompt=True.

    `offset` is the index of the first completion token; 0 means "score everything"."""
    msgs = to_messages(row)
    if msgs is None:
        t = row.get("text")
        if not isinstance(t, str):
            return None, 0
        return tok.encode(t), 0
    ids = tok.apply_chat_template(msgs, return_dict=False)
    add_gen = msgs[-1].get("role") == "assistant"
    offset = len(tok.apply_chat_template(msgs[:-1], add_generation_prompt=add_gen, return_dict=False))
    return ids, offset


def answer_text(row):
    """The completion string — used for the bits-per-byte denominator."""
    msgs = to_messages(row)
    if msgs is None:
        return row.get("text") or ""
    return msgs[-1].get("content") or ""


def prompt_ids(tok, row):
    """Token ids for the prompt alone, with the generation prompt appended (for generation)."""
    msgs = to_messages(row)
    if msgs is None:
        return None
    add_gen = msgs[-1].get("role") == "assistant"
    return tok.apply_chat_template(msgs[:-1], add_generation_prompt=add_gen, return_dict=False)


def sequence_loss(model, ids, start, max_len):
    """(summed nats, tokens scored) over ids[start+1:], i.e. the completion only."""
    ids = ids[:max_len]
    if len(ids) < 2:
        return 0.0, 0
    labels = ids[1:]
    start = max(min(start, len(ids)) - 1, 0)
    if start >= len(labels):
        return 0.0, 0
    x = mx.array(ids[:-1])[None]
    y = mx.array(labels[start:])[None]
    logits = model(x).astype(mx.float32)[:, start:, :]
    loss = nn.losses.cross_entropy(logits, y, reduction="sum")
    mx.eval(loss)
    return float(loss), int(y.size)


def mcq_pick(model, tok, row, max_len):
    """Teacher-forced multiple choice: score each option, pick the lowest mean nats.

    Length-normalised — unnormalised sum-loss systematically prefers the shortest option. No
    generation, so it is immune to verbosity differences between models, which is exactly why
    it is the headline cross-model metric."""
    choices = row.get("choices")
    gold = row.get("answer")
    if not isinstance(choices, list) or len(choices) < 2 or gold is None:
        return None
    msgs = to_messages(row)
    if msgs is None:
        return None
    # Substitute each choice INTO the reference answer sentence rather than scoring the bare
    # choice string. Training taught "X is a thalamic dampener." — scoring the bare "thalamic
    # dampener" as an entire reply is off-distribution for a tuned model and washes out the
    # signal (observed: bits/byte fell 8x while bare-string MCQ moved 1pt). Comparing full
    # sentences that differ only in the answer span isolates the fact being tested.
    ref = msgs[-1].get("content") or ""
    def as_reply(choice):
        c = str(choice)
        return ref.replace(str(gold), c) if str(gold) and str(gold) in ref else c
    best, best_i = None, 0
    for i, c in enumerate(choices):
        probe = dict(row)
        probe["messages"] = msgs[:-1] + [{"role": "assistant", "content": as_reply(c)}]
        probe.pop("prompt", None)
        probe.pop("completion", None)
        ids, off = tokenize_row(tok, probe)
        total, n = sequence_loss(model, ids, off, max_len)
        if n == 0:
            continue
        mean = total / n
        if best is None or mean < best:
            best, best_i = mean, i
    try:
        gold_i = choices.index(gold)
    except ValueError:
        return None
    return bool(best_i == gold_i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-path", default=None)
    # One or more .jsonl files (comma-separated). Multiple suites share a single model load,
    # which is the dominant cost — 4 suites x 2 variants would otherwise be 8 loads per cell.
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=0)      # 0 = every row (no silent truncation)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--generate", action="store_true")   # also produce free-text answers
    a = ap.parse_args()

    model, tok = load(a.model, adapter_path=a.adapter_path)

    files = [f for f in a.data.split(",") if f.strip()]
    per_file = {}
    for path in files:
        per_file[path] = score_file(model, tok, path, a)

    # Single file -> emit the payload flat (the shape eval.ts expects). Multiple -> keyed by path.
    payload = per_file[files[0]] if len(files) == 1 else {"suites": per_file}
    first = per_file[files[0]]
    print(f"Test loss {first['loss']:.4f}, Test ppl {first['ppl']:.4f}, rows {first['rows']}, tokens {first['tokens']}")
    print("EVAL_JSON " + json.dumps(payload))


def score_file(model, tok, path, a):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if a.limit > 0:
        rows = rows[: a.limit]

    total_loss, total_tok, total_bytes, scored = 0.0, 0, 0, 0
    mcq_right, mcq_n = 0, 0
    for row in rows:
        ids, off = tokenize_row(tok, row)
        if not ids:
            continue
        loss, n = sequence_loss(model, ids, off, a.max_len)
        if n:
            total_loss += loss
            total_tok += n
            total_bytes += len(answer_text(row).encode("utf-8"))
            scored += 1
        got = mcq_pick(model, tok, row, a.max_len)
        if got is not None:
            mcq_n += 1
            mcq_right += int(got)

    avg = total_loss / max(total_tok, 1)
    out = {
        "loss": round(avg, 4),
        "ppl": round(math.exp(avg), 4),
        # nats -> bits, over the answer's utf-8 bytes: the same denominator for every tokenizer.
        "bitsPerByte": round(total_loss / (LN2 * total_bytes), 4) if total_bytes else None,
        "rows": scored,
        "tokens": total_tok,
        "bytes": total_bytes,
    }
    if mcq_n:
        out["mcq"] = {"correct": mcq_right, "n": mcq_n, "acc": round(mcq_right / mcq_n, 4)}

    if a.generate:
        prompts, keep = [], []
        for i, row in enumerate(rows):
            p = prompt_ids(tok, row)
            if p:
                prompts.append(p)
                keep.append(i)
        if prompts:
            # Greedy (temp 0) so the comparison is base-vs-tuned, not sampling noise.
            resp = batch_generate(model, tok, prompts, max_tokens=a.max_tokens, verbose=False)
            gens = []
            for idx, text in zip(keep, resp.texts):
                gens.append({
                    "i": idx,
                    "prompt": (to_messages(rows[idx]) or [{}])[0].get("content", ""),
                    "reference": answer_text(rows[idx]),
                    "text": text.strip(),
                })
            out["generations"] = gens
            tps = getattr(resp.stats, "generation_tps", None)
            if tps is not None:
                out["genTokensPerSec"] = round(float(tps), 2)

    return out


def _self_check():
    """No-model checks for the pure logic: masking, row normalisation, bpb math."""
    class FakeTok:
        """Mimics the contract that matters: templating adds wrapper tokens, and the
        prompt-only render is a prefix of the full render."""
        def encode(self, s):
            return [ord(c) % 97 for c in s.split()[0]] if s else []

        def apply_chat_template(self, msgs, add_generation_prompt=False, return_dict=False, **kw):
            ids = [1]  # BOS emitted by the template itself
            for m in msgs:
                ids += [2] + [ord(c) % 97 for c in m.get("content", "").split()[0]] + [3]
            if add_generation_prompt:
                ids += [2]
            return ids

    tok = FakeTok()

    # prompt+completion normalises to the same two-message shape the trainer builds
    row = {"prompt": "question here", "completion": "answer here"}
    msgs = to_messages(row)
    assert msgs and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant", msgs
    ids, off = tokenize_row(tok, row)
    assert 0 < off < len(ids), (off, len(ids))
    # the offset marks the answer, so fewer labels are scored than the full sequence
    assert (len(ids) - 1) - max(off - 1, 0) < len(ids) - 1

    # messages rows: offset lands after the generation prompt
    row2 = {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]}
    ids2, off2 = tokenize_row(tok, row2)
    assert ids2[:off2] == tok.apply_chat_template(row2["messages"][:-1], add_generation_prompt=True)

    # text rows are unmaskable -> score everything
    ids3, off3 = tokenize_row(tok, {"text": "hello"})
    assert off3 == 0 and ids3

    assert answer_text(row) == "answer here"
    assert answer_text({"text": "raw"}) == "raw"

    # bits-per-byte: 1 nat over 1 byte == 1/ln2 bits
    assert abs((1.0 / (LN2 * 1)) - 1.4426950408889634) < 1e-9
    print("self-check ok: masking, row shapes, bpb")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _self_check()
    else:
        main()
