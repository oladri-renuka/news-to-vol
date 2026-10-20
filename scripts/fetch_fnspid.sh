#!/usr/bin/env bash
# Download the FNSPID headline files (~29 GB total) and build the filtered, trading-day-aligned
# headline table. Run this on the pod; nothing it produces is committed.
#   bash scripts/fetch_fnspid.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=data/raw/fnspid
mkdir -p "$DEST"
BASE=https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news
for f in nasdaq_exteral_data.csv All_external.csv; do
  echo "downloading $f"
  curl -L -C - --retry 5 -o "$DEST/$f" "$BASE/$f"
done
# Remove any stale filtered cache so it is rebuilt from the full files.
rm -f data/interim/headlines_raw.parquet
python -m src.data.news --config configs/base.yaml --audit
python -m src.data.news --config configs/base.yaml
echo "done: data/interim/headlines_aligned.parquet, reports/tables/news_coverage.csv"
