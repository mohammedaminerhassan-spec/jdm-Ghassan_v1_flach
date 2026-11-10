#!/usr/bin/env python3
"""Fine-tune a model with QLoRA (or plain LoRA) and export it back to GGUF.

This is the training half of the "body": the engine serves GGUF, this script
turns a GGUF (or a HuggingFace checkpoint) into a trainable model, trains it
on your data, merges the adapter and hands the result back to the engine.

    # 1. engine model -> trainable checkpoint (only if you start from GGUF)
    python tools/py/gguf_to_safetensors.py --gguf m.gguf --out models/m-hf

    # 2. data -> tokenized shards
    python tools/py/prepare_dataset.py --tokenizer models/m-hf \
        --lake /kaggle/input/english_parquet --out artifacts/shards_en

    # 3. fine-tune
    python tools/py/qlora_finetune.py --model models/m-hf \
        --data artifacts/shards_en/train.jsonl --out runs/exp1 \
        --epochs 2 --batch-size 1 --grad-accum 8

    # 4. back to the engine
    python tools/convert_safetensors_to_gguf.py --model-dir runs/exp1/merged \
        --out artifacts/my-model-f16.gguf

It runs on a single 16 GB T4 (Kaggle) with 4-bit weights and gradient
checkpointing.
"""
import argparse
import json
import os
import shutil
import sys


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

class JsonlTokenDataset:
    """Reads the tokenized documents written by prepare_dataset.py.

    Each line is {"ids": [...], "mask": [...]}. The mask is 1 on the tokens
    that must be predicted (the assistant spans for a chat corpus, every
    token for a plain text corpus), so the same file serves pretraining and
    supervised fine-tuning.
    """

    def __init__(self, path, seq_len):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ids = obj.get("ids")
                if not ids:
                    continue
                mask = obj.get("mask")
                if mask is None or len(mask) != len(ids):
                    mask = [1] * len(ids)
                self.rows.append((ids, mask))
        self.seq_len = seq_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, mask = self.rows[i]
        ids = ids[: self.seq_len]
        mask = mask[: self.seq_len]
        # The model predicts the next token, so the target is the ids shifted
        # by one and the loss mask is shifted with them.
        input_ids = ids[:-1]
        labels = ids[1:]
        loss_mask = mask[1:]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "loss_mask": loss_mask,
        }


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        import torch
        n = max(len(f["input_ids"]) for f in feats)
        batch = {"input_ids": [], "attention_mask": [], "labels": [], "loss_mask": []}
        for f in feats:
            k = n - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_id] * k)
            batch["attention_mask"].append(f["attention_mask"] + [0] * k)
            batch["labels"].append(f["labels"] + [-100] * k)
            batch["loss_mask"].append(f["loss_mask"] + [0] * k)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


class MaskedTrainer:
    """A Trainer subclass that honours the per-token loss mask.

    The stock Trainer computes the mean over every label that is not -100.
    Our shards already encode the supervised span in `loss_mask`, so the
    labels of the unsupervised positions are set to -100 here and the loss
    is a plain mean over the supervised tokens.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        import torch
        labels = inputs.pop("labels", None)
        loss_mask = inputs.pop("loss_mask", None)
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        if labels is None:
            loss = outputs.loss if hasattr(outputs, "loss") else None
            return (loss, outputs) if return_outputs else loss
        if loss_mask is not None:
            labels = labels.clone()
            labels[loss_mask == 0] = -100
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def build_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(args.model)
    is_moe = getattr(config, "num_experts", 0) > 0

    kwargs = {}
    if args.qlora:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        **kwargs,
    )
    model.config.use_cache = False

    if args.lora:
        from peft import LoraConfig, get_peft_model
        targets = args.target_modules
        if targets == "auto":
            if is_moe:
                # Adapters on every expert would mean 128 x 3 adapters per
                # layer: they do not fit on a T4 and they are not what you
                # want to steer. Attention is the sane target for a MoE
                # model.
                targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
            else:
                targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"]
        if isinstance(targets, str):
            targets = [t.strip() for t in targets.split(",") if t.strip()]
        peft_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model, is_moe


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HuggingFace model directory")
    ap.add_argument("--data", required=True,
                    help="tokenized jsonl (prepare_dataset.py) or a HuggingFace dataset name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default="auto",
                    help="'auto', or a comma separated list")
    ap.add_argument("--no-lora", dest="lora", action="store_false",
                    help="full fine-tuning instead of LoRA")
    ap.add_argument("--qlora", action="store_true", help="4-bit base weights")
    ap.add_argument("--merge", action="store_true", help="merge the adapter and save it")
    ap.add_argument("--export-gguf", default=None,
                    help="merge, then export to this GGUF path")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--fp16", dest="bf16", action="store_false")
    ap.add_argument("--grad-checkpointing", action="store_true", default=True)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("=== model ===")
    model, is_moe = build_model(args)
    print(f"  loaded {args.model} | moe={is_moe} | "
          f"{'qlora' if args.qlora else 'lora' if args.lora else 'full'}")

    print("=== data ===")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.data.endswith(".jsonl"):
        ds = JsonlTokenDataset(args.data, args.seq_len)
    else:
        from datasets import load_dataset
        raw = load_dataset(args.data, split="train")
        cols = raw.column_names
        text_col = "text" if "text" in cols else ("content" if "content" in cols else cols[0])

        def _tok(batch):
            enc = tok(batch[text_col], truncation=True, max_length=args.seq_len)
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": [list(ids) for ids in enc["input_ids"]],
                "loss_mask": [[1] * len(ids) for ids in enc["input_ids"]],
            }
        ds = raw.map(_tok, batched=True, remove_columns=cols)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "loss_mask"])
    print(f"  {len(ds)} documents | seq_len {args.seq_len}")

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    from transformers import Trainer, TrainingArguments
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=not args.bf16,
        optim="paged_adamw_8bit" if args.qlora else "adamw_torch",
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        gradient_checkpointing=False,   # enabled on the model itself
    )

    trainer = MaskedTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=Collator(pad_id),
    )

    print("=== training ===")
    trainer.train()

    print("=== saving ===")
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "gai_finetune_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.merge or args.export_gguf:
        print("=== merging the adapter ===")
        merged_dir = os.path.join(args.out, "merged")
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tok.save_pretrained(merged_dir)
        print(f"  merged model -> {merged_dir}")

        if args.export_gguf:
            print("=== exporting to GGUF ===")
            here = os.path.dirname(os.path.abspath(__file__))
            converter = os.path.join(here, "..", "convert_safetensors_to_gguf.py")
            import subprocess
            subprocess.run([sys.executable, converter,
                            "--model-dir", merged_dir,
                            "--out", args.export_gguf], check=True)
            print(f"  GGUF -> {args.export_gguf}")

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
