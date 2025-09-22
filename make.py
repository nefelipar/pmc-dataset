import gzip
import json
import logging
import os
import re
from pathlib import Path

# Αλλαγή στο path αν χρειάζεται
jsonl_path = "data/jsonl/oa_comm_xml.PMC000xxxxxx.baseline.2025-06-26.jsonl.gz"

# Configure logging to file named like the JSONL base
base_name = re.sub(r"\.jsonl(\.gz)?$", "", Path(jsonl_path).name)
logs_dir = Path("data/log")
logs_dir.mkdir(parents=True, exist_ok=True)
log_path = logs_dir / f"{base_name}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

total = 0
missing_abstract = 0
missing_text = 0

logging.info("Processing JSONL: %s", jsonl_path)

with gzip.open(jsonl_path, "rt", encoding="utf-8") as f:
    for line in f:
        total += 1
        try:
            obj = json.loads(line)
            if not obj.get("abstract"):
                missing_abstract += 1
            if not obj.get("text"):
                missing_text += 1
        except Exception:
            logging.exception("Error parsing line")

logging.info("Total records: %d", total)
logging.info("Records missing abstract: %d", missing_abstract)
logging.info("Records missing text: %d", missing_text)

logging.info("Logs written to: %s", log_path)
