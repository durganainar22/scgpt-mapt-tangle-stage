#!/bin/bash
# v3 Step 0: download SEA-AD MTG snRNA-seq files (public S3, no login).
#
#   bash seaad/00_download.sh meta     # 1.2 GB per-cell metadata  -> enough for Step 1 audit
#   bash seaad/00_download.sh donors   # 84 per-donor .h5ad files, ~0.5 GB each (~42 GB) -> Step 2+
#
# Run from the project directory. Files go to data/ (gitignored). Re-running skips finished files.
set -euo pipefail
BASE="https://sea-ad-single-cell-profiling.s3.amazonaws.com/MTG/RNAseq"
mkdir -p data
cd data

case "${1:-meta}" in
  meta)
    wget -c "${BASE}/SEAAD_MTG_RNAseq_final-nuclei_metadata.2026-06-22.csv"
    ;;
  donors)
    mkdir -p donors
    # List the donor objects from the bucket, then fetch each one
    curl -s "https://sea-ad-single-cell-profiling.s3.amazonaws.com/?list-type=2&prefix=MTG/RNAseq/donors_objects/" \
      | grep -o '<Key>[^<]*\.h5ad</Key>' | sed 's/<Key>MTG\/RNAseq\///; s/<\/Key>//' > donors/files.txt
    echo "$(wc -l < donors/files.txt) donor files"
    while read -r f; do
      wget -c -q --show-progress -P donors "${BASE}/${f}"
    done < donors/files.txt
    ;;
  *) echo "usage: $0 meta|donors"; exit 1 ;;
esac
ls -lh
