import argparse
from typing import Dict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM


def get_prompt(article: str) -> str:
    return (
        "You are a helpful assistant that writes concise news summaries.\n"
        "Summarize the following news article in 3-5 sentences focusing on the key facts.\n\n"
        f"Article:\n{article}\n\n"
        "Summary:"
    )


def parse_args():
    p = argparse.ArgumentParser(description="QLoRA SFT for summarization (CNN/DailyMail) with Gemma 3 1B IT")
    # Data/model
    p.add_argument("--model_id", type=str, default="google/gemma-3-1b-it")
    p.add_argument("--output_dir", type=str, default="runs/gemma3-1b-it-sft-cnn")
    p.add_argument("--dataset_name", type=str, default="cnn_dailymail")
    p.add_argument("--dataset_config", type=str, default="3.0.0")
    p.add_argument("--split_train", type=str, default="train[:2%]")
    p.add_argument("--split_val", type=str, default="validation[:2%]")

    # Training
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--train_batch", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)

    # LoRA / optimization toggles
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--packing", action="store_true", default=True)
    p.add_argument("--no_packing", dest="packing", action="store_false")
    p.add_argument("--use_4bit", action="store_true", default=True)
    p.add_argument("--no_use_4bit", dest="use_4bit", action="store_false")

    return p.parse_args()


def main():
    args = parse_args()
    # Load dataset
    train_ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split_train)
    val_ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split_val)

    # Map dataset to text with prompt + target
    def format_row(row: Dict) -> Dict:
        article = row.get("article") or row.get("document")
        summary = row.get("highlights") or row.get("summary")
        if article is None or summary is None:
            raise ValueError("Expected 'article'/'document' and 'highlights'/'summary' fields.")
        text = get_prompt(article) + " " + summary
        return {"text": text}

    train_ds = train_ds.map(format_row, remove_columns=train_ds.column_names, desc="Formatting train")
    val_ds = val_ds.map(format_row, remove_columns=val_ds.column_names, desc="Formatting val")

    # Load tokenizer and model (QLoRA)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantized base weights for memory efficiency
    load_kwargs = {
        "device_map": "auto",
    }
    if args.use_4bit:
        load_kwargs.update(
            {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            }
        )
    else:
        load_kwargs.update({
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        })

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)

    # PEFT LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    # Only compute loss on the completion (after "Summary:")
    collator = DataCollatorForCompletionOnlyLM(
        response_template="Summary:",
        instruction_template=None,
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=2,
        gradient_checkpointing=args.gradient_checkpointing,
        max_seq_length=args.max_seq_len,
        packing=args.packing,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        report_to=["none"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        formatting_func=None,  # we already produced a 'text' field
        dataset_text_field="text",
        data_collator=collator,
        args=training_args,
    )

    trainer.train()

    # Save PEFT adapter and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
