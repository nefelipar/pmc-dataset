python create_dataset/b_token_stats.py \
  --files "oa_comm_xml.PMC000xxxxxx.baseline.2025-06-26.jsonl" \
  --tokenizer "Qwen/Qwen1.5-1.8B" &

python create_dataset/b_token_stats.py \
  --files "oa_comm_xml.PMC001xxxxxx.baseline.2025-06-26.jsonl" \
  --tokenizer "Qwen/Qwen1.5-1.8B" &
wait


python create_dataset/c_aggregate_stats_records.py \
  --tokenizer "Qwen/Qwen1.5-1.8B" \
