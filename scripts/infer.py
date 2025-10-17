import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def build_prompt(article: str) -> str:
    return (
        "You are a helpful assistant that writes concise news summaries.\n"
        "Summarize the following news article in 3-5 sentences focusing on the key facts.\n\n"
        f"Article:\n{article}\n\n"
        "Summary:"
    )


def load_model(model_id: str, adapter_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tokenizer, model


def summarize(article: str, model_id: str, adapter_dir: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    tokenizer, model = load_model(model_id, adapter_dir)
    prompt = build_prompt(article)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    # Return only the part after "Summary:"
    if "Summary:" in output_text:
        return output_text.split("Summary:", 1)[1].strip()
    return output_text.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize text with Gemma+LoRA adapter")
    parser.add_argument("--article", type=str, help="Raw article text to summarize")
    parser.add_argument("--article_file", type=str, help="Path to file containing the article text")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--adapter_dir", type=str, default="runs/gemma3-1b-it-sft-cnn")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    if not args.article and not args.article_file:
        raise SystemExit("Please provide --article or --article_file")
    article_text = args.article
    if args.article_file:
        with open(args.article_file, "r", encoding="utf-8") as f:
            article_text = f.read()

    print(
        summarize(
            article=article_text,
            model_id=args.model_id,
            adapter_dir=args.adapter_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    )
