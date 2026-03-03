"""Train the pairwise SQL selector model using LoRA fine-tuning.

Uses Qwen2.5-1.5B-Instruct as base model with LoRA for binary classification.
Designed to run on EKS Ray cluster or locally with GPU.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def load_data(data_path: str) -> Dataset:
    """Load training data from JSON file."""
    with open(data_path) as f:
        data = json.load(f)
    return Dataset.from_list(data)


def tokenize_fn(examples, tokenizer, max_length=1024):
    """Tokenize text for classification."""
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def main():
    parser = argparse.ArgumentParser(description="Train pairwise SQL selector")
    parser.add_argument("--train_data", default="optimization/training/selector_data_train.json")
    parser.add_argument("--val_data", default="optimization/training/selector_data_val.json")
    parser.add_argument("--output_dir", default="optimization/training/selector_model")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        torch_dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    print(f"Loading training data: {args.train_data}")
    train_dataset = load_data(args.train_data)
    val_dataset = load_data(args.val_data) if Path(args.val_data).exists() else None

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    train_dataset = train_dataset.rename_column("label", "labels")
    train_dataset.set_format("torch")

    if val_dataset:
        val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=["text"])
        val_dataset = val_dataset.rename_column("label", "labels")
        val_dataset.set_format("torch")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        bf16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=4,
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting training...")
    trainer.train()

    # Save the LoRA adapter
    adapter_path = Path(args.output_dir) / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"LoRA adapter saved to: {adapter_path}")

    # Evaluate on validation set
    if val_dataset:
        eval_results = trainer.evaluate()
        print(f"Validation loss: {eval_results['eval_loss']:.4f}")

    print("Training complete!")


if __name__ == "__main__":
    main()
