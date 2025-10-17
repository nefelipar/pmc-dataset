**Overview**
- Supervised fine-tuning (SFT) of `google/gemma-3-1b-it` for text summarization using the CNN/DailyMail dataset.
- Uses QLoRA (4-bit) with TRL’s `SFTTrainer` for efficient finetuning on a single GPU.

**Environment**
- Python 3.10+
- GPU with CUDA recommended (BF16 or FP16)
- Install deps:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`

**Notes on Model and Data**
- Model: `google/gemma-3-1b-it` (instruction-tuned, causal LM)
- Dataset: `cnn_dailymail` config `3.0.0` with fields `article` and `highlights`
- Prompting style avoids dependency on chat templates and uses a simple instruction + response marker (`Summary:`) with completion-only loss.

**Training (SFT)**
- CLI args (defaults shown):
  - `--model_id google/gemma-3-1b-it`
  - `--output_dir runs/gemma3-1b-it-sft-cnn`
  - `--dataset_name cnn_dailymail`
  - `--dataset_config 3.0.0`
  - `--split_train "train[:2%]"` (use `train` for full)
  - `--split_val "validation[:2%]"`
  - `--max_seq_len 2048`
  - `--train_batch 2`
  - `--grad_accum 8`
  - `--epochs 1`
  - `--learning_rate 2e-4`
  - `--warmup_ratio 0.03`
  - `--log_steps 10 --save_steps 500 --eval_steps 500`
  - `--lora_r 16 --lora_alpha 16 --lora_dropout 0.1`
  - `--[no_]gradient_checkpointing` (default on)
  - `--[no_]packing` (default on)
  - `--[no_]use_4bit` (default on)

- Example: quick smoke run on 2% of data
  - `python scripts/train_sft.py --split_train "train[:2%]" --split_val "validation[:2%]"`

- Full-dataset example:
  - `python scripts/train_sft.py --split_train train --split_val validation --epochs 2 --train_batch 1 --grad_accum 16`

This trains a PEFT LoRA adapter and saves it into `--output_dir` alongside the tokenizer.

**Inference**
- Args (defaults shown):
  - `--model_id google/gemma-3-1b-it`
  - `--adapter_dir runs/gemma3-1b-it-sft-cnn`
  - `--max_new_tokens 256 --temperature 0.2 --top_p 0.9`
- Examples:
  - `python scripts/infer.py --article "<paste article text>"`
  - `python scripts/infer.py --article_file /path/to/article.txt --adapter_dir runs/gemma3-1b-it-sft-cnn`

**Tips**
- If you hit OOM:
  - Reduce `TRAIN_BATCH` or increase `GRAD_ACCUM`.
  - Reduce `MAX_SEQ_LEN`.
  - Ensure 4-bit loading is active and gradient checkpointing is enabled.
- For better quality:
  - Train for more steps/epochs or expand `SPLIT_TRAIN` to full dataset.
  - Increase context length if your GPU allows.
  - Tune LoRA ranks (`r`) and dropout.

**Evaluation (optional)**
- For quick validation, you can sample a handful of validation items and compute ROUGE with `evaluate`. For full-scale evaluation, script up a small loop to generate summaries and score them; avoid running during training to save time.
