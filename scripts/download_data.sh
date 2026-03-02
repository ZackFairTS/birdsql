#!/bin/bash
# Download BIRD benchmark dev set
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_DIR}/data"

mkdir -p "${DATA_DIR}"

echo "=== Downloading BIRD dev set ==="
if [ ! -f "/tmp/bird_dev.zip" ]; then
    wget -O /tmp/bird_dev.zip "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
fi

echo "=== Extracting data ==="
unzip -o /tmp/bird_dev.zip -d /tmp/bird_dev_extract/

# Copy files
cp /tmp/bird_dev_extract/dev_20240627/dev.json "${DATA_DIR}/"
cp /tmp/bird_dev_extract/dev_20240627/dev.sql "${DATA_DIR}/"
cp /tmp/bird_dev_extract/dev_20240627/dev_tables.json "${DATA_DIR}/"

# Extract databases
unzip -o /tmp/bird_dev_extract/dev_20240627/dev_databases.zip -d "${DATA_DIR}/"

# Generate gold SQL
python3 -c "
import json
with open('${DATA_DIR}/dev.json') as f:
    data = json.load(f)
with open('${DATA_DIR}/dev_gold.sql', 'w') as f:
    for entry in data:
        f.write(f\"{entry['SQL']}\t{entry['db_id']}\n\")
print(f'Generated dev_gold.sql with {len(data)} entries')
"

echo "=== Data setup complete ==="
echo "  dev.json: $(wc -l < "${DATA_DIR}/dev.json") lines"
echo "  Databases: $(ls "${DATA_DIR}/dev_databases/" | wc -l)"
