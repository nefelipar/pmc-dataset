python create_dataset/b_token_stats.py \
  --files "oa_comm_xml.PMC000xxxxxx.baseline.2025-06-26.jsonl" \
  --tokenizer "Qwen/Qwen1.5-1.8B" &

python create_dataset/b_token_stats.py \
  --files "oa_comm_xml.PMC001xxxxxx.baseline.2025-06-26.jsonl" \
  --tokenizer "Qwen/Qwen1.5-1.8B" &

python create_dataset/b_token_stats.py \
  --files "oa_comm_xml.PMC002xxxxxx.baseline.2025-06-26.jsonl" \
  --tokenizer "Qwen/Qwen1.5-1.8B" &

python create_dataset/b_token_stats.py \
  --files "oa_comm_xml.PMC003xxxxxx.baseline.2025-06-26.jsonl" \
  --tokenizer "Qwen/Qwen1.5-1.8B" &

python create_dataset/b_token_stats.py \
  --files "oa_comm_xml.PMC004xxxxxx.baseline.2025-06-26.jsonl" \
  --tokenizer "Qwen/Qwen1.5-1.8B" &

python create_dataset/b_token_stats.py \
  --files "oa_comm_xml.PMC005xxxxxx.baseline.2025-06-26.jsonl" \
  --tokenizer "Qwen/Qwen1.5-1.8B" &

python create_dataset/b_token_stats.py \
  --files "oa_comm_xml.PMC006xxxxxx.baseline.2025-06-26.jsonl" \
  --tokenizer "Qwen/Qwen1.5-1.8B" &  

python create_dataset/b_token_stats.py \
  --files "oa_comm_xml.PMC007xxxxxx.baseline.2025-06-26.jsonl" \
  --tokenizer "Qwen/Qwen1.5-1.8B" &

python create_dataset/b_token_stats.py \
  --files "oa_comm_xml.PMC008xxxxxx.baseline.2025-06-26.jsonl" \
  --tokenizer "Qwen/Qwen1.5-1.8B" &

python create_dataset/b_token_stats.py \
  --files "oa_comm_xml.PMC009xxxxxx.baseline.2025-06-26.jsonl" \
  --tokenizer "Qwen/Qwen1.5-1.8B" &

python create_dataset/b_token_stats.py \
  --files "oa_comm_xml.PMC010xxxxxx.baseline.2025-06-26.jsonl" \
  --tokenizer "Qwen/Qwen1.5-1.8B" &

python create_dataset/b_token_stats.py \
  --files "oa_comm_xml.PMC011xxxxxx.baseline.2025-06-26.jsonl" \
  --tokenizer "Qwen/Qwen1.5-1.8B" &

python create_dataset/b_token_stats.py \
  --files "oa_comm_xml.PMC012xxxxxx.baseline.2025-06-26.jsonl" \
  --tokenizer "Qwen/Qwen1.5-1.8B" &
wait


python create_dataset/c_aggregate_stats_records.py \
  --tokenizer "Qwen/Qwen1.5-1.8B" \
