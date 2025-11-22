# Step 1: Convert tar.gz files to JSONL format
python create_dataset/a_create_jsonl_from_tar.py oa_comm_xml.PMC000xxxxxx.baseline.2025-06-26.tar.gz
python create_dataset/a_create_jsonl_from_tar.py oa_comm_xml.PMC001xxxxxx.baseline.2025-06-26.tar.gz 
python create_dataset/a_create_jsonl_from_tar.py oa_comm_xml.PMC002xxxxxx.baseline.2025-06-26.tar.gz
python create_dataset/a_create_jsonl_from_tar.py oa_comm_xml.PMC003xxxxxx.baseline.2025-06-26.tar.gz
python create_dataset/a_create_jsonl_from_tar.py oa_comm_xml.PMC004xxxxxx.baseline.2025-06-26.tar.gz
python create_dataset/a_create_jsonl_from_tar.py oa_comm_xml.PMC005xxxxxx.baseline.2025-06-26.tar.gz
python create_dataset/a_create_jsonl_from_tar.py oa_comm_xml.PMC006xxxxxx.baseline.2025-06-26.tar.gz
python create_dataset/a_create_jsonl_from_tar.py oa_comm_xml.PMC007xxxxxx.baseline.2025-06-26.tar.gz
python create_dataset/a_create_jsonl_from_tar.py oa_comm_xml.PMC008xxxxxx.baseline.2025-06-26.tar.gz
python create_dataset/a_create_jsonl_from_tar.py oa_comm_xml.PMC009xxxxxx.baseline.2025-06-26.tar.gz            
python create_dataset/a_create_jsonl_from_tar.py oa_comm_xml.PMC010xxxxxx.baseline.2025-06-26.tar.gz    
python create_dataset/a_create_jsonl_from_tar.py oa_comm_xml.PMC011xxxxxx.baseline.2025-06-26.tar.gz
python create_dataset/a_create_jsonl_from_tar.py oa_comm_xml.PMC012xxxxxx.baseline.2025-06-26.tar.gz

# Step 2: Clean JSONL files. Keep only the records with non-empty 'abstract' and 'body_text' fields.
python create_dataset/b_clean_jsonl.py
