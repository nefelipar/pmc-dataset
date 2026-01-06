python utils/create_random_dataset.py \
  --input "__dataset__/filtered_merged.jsonl" \
  --output "__dataset__/random_500.jsonl" \
  --samples 500

python utils/create_random_dataset.py \
  --input "__dataset__/filtered_merged.jsonl" \
  --output "__dataset__/random_1_000.jsonl" \
  --samples 1000

  python utils/create_random_dataset.py \
  --input "__dataset__/filtered_merged.jsonl" \
  --output "__dataset__/random_5_000.jsonl" \
  --samples 5000

  python utils/create_random_dataset.py \
  --input "__dataset__/filtered_merged.jsonl" \
  --output "__dataset__/random_10_000.jsonl" \
  --samples 10000
