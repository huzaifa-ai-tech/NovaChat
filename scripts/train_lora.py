import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune a small language model (CPU or GPU)."
    )
    parser.add_argument(
        "--model_name",
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        help="Base model. For faster CPU training try HuggingFaceTB/SmolLM2-360M-Instruct.",
    )
    parser.add_argument(
        "--data_path",
        default=str(ROOT / "data" / "training_data.json"),
        help="JSON file with instruction/response pairs.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(ROOT / "models" / "lora"),
        help="Where to save the LoRA adapter.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    with open(args.data_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    def tokenize(example):
        prefix = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(
            f"{example['response']}\n", add_special_tokens=False
        )["input_ids"]
        if not response_ids:
            response_ids = [tokenizer.eos_token_id]
        combined = (prefix_ids + response_ids)[: args.max_length]
        combined = combined + [tokenizer.pad_token_id] * (
            args.max_length - len(combined)
        )
        labels = ([-100] * len(prefix_ids) + response_ids)[: args.max_length]
        labels = labels + [-100] * (len(combined) - len(labels))
        return {
            "input_ids": combined,
            "labels": labels,
            "attention_mask": [
                1 if t != tokenizer.pad_token_id else 0 for t in combined
            ],
        }

    dataset = Dataset.from_list(raw_data)
    tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="max_length", max_length=args.max_length
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        fp16=(device == "cuda"),
        dataloader_pin_memory=(device == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
