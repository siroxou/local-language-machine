#!/usr/bin/env python3
"""LoRA fine-tune via Unsloth (CUDA/NVIDIA path).

Invoked by src/native/training/train.ts as a subprocess with `--config cfg.json`. Emits
a machine-parseable `TRAIN_PROGRESS step=<N> loss=<X>` line on every log step so the Node
runner can stream live loss to the UI (see parseProgress() in train.ts). Not used on
Apple Silicon — that path drives mlx_lm.lora directly, no script.

Runs only where Unsloth's kernels are supported (an NVIDIA GPU). Version-sensitive
across trl/peft releases; pin versions in venv.ts if a release breaks the API.
"""
import argparse
import json
import sys


def log(msg):
    print(msg, flush=True)


def build_texts(rows, tokenizer):
    """Normalize supported row shapes into a single training string per row."""
    out = []
    for r in rows:
        if isinstance(r.get("text"), str):
            out.append(r["text"])
        elif isinstance(r.get("messages"), list):
            out.append(tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False))
        elif isinstance(r.get("prompt"), str) and isinstance(r.get("completion"), str):
            out.append(r["prompt"] + r["completion"] + (tokenizer.eos_token or ""))
        else:
            raise ValueError("row must have text, messages, or prompt+completion")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))

    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments, TrainerCallback

    log(f"▸ Loading base model {cfg['base_model']} (4-bit) …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_rank"] * 2,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    rows = [json.loads(l) for l in open(cfg["data"]) if l.strip()]
    dataset = Dataset.from_dict({"text": build_texts(rows, tokenizer)})

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                log(f"TRAIN_PROGRESS step={state.global_step} loss={logs['loss']:.4f}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        callbacks=[Progress()],
        args=TrainingArguments(
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=1,
            max_steps=cfg["iters"],
            learning_rate=cfg["learning_rate"],
            logging_steps=1,
            output_dir=cfg["adapter_path"],
            optim="adamw_8bit",
            report_to="none",
        ),
    )
    log("▸ Training …")
    trainer.train()

    model.save_pretrained(cfg["adapter_path"])
    tokenizer.save_pretrained(cfg["adapter_path"])
    log(f"▸ Saved adapter → {cfg['adapter_path']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # surface the real error to the streamed log
        log(f"ERROR: {e}")
        sys.exit(1)
