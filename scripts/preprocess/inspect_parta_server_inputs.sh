#!/usr/bin/env bash
set -u

DATA_ROOT="${1:-/data2/wlx/data}"
PROJECT_ROOT="${2:-/data2/wlx/projects/space}"

echo "=== SYSTEM ==="
date
hostname
df -h "${DATA_ROOT}" 2>&1 || true
python --version 2>&1 || true
hf --version 2>&1 || true

echo "=== PROJECT ==="
git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>&1 || true
git -C "${PROJECT_ROOT}" status --short 2>&1 | head -n 100 || true

echo "=== TOP-LEVEL DATA DIRECTORIES ==="
find "${DATA_ROOT}" -maxdepth 3 -type d \
  \( -iname '*vsi*' -o -iname '*scannet*' -o -iname '*arkit*' \
     -o -iname '*adt*' -o -iname '*hypersim*' -o -iname '*procthor*' \) \
  -print 2>/dev/null | sort | head -n 300

echo "=== KEY ANNOTATION/METADATA FILES ==="
find "${DATA_ROOT}" -maxdepth 6 -type f \
  \( -iname 'vsi_590k.jsonl' -o -iname '*meta*info*' \
     -o -iname '*aggregation.json' -o -iname '*.sens' \
     -o -iname '*pose*.txt' -o -iname '*intrinsic*.txt' \
     -o -iname '*trajectory*' -o -iname '*.glb' -o -iname '*.ply' \) \
  -printf '%s\t%p\n' 2>/dev/null | sort -n | head -n 500

echo "=== VSI VIDEO COUNTS AND EXAMPLES ==="
for dataset_name in scannet scannetpp scannetppv2 arkitscenes adt hypersim procthor s3dis robotics ytb_roomtour; do
  count="$(find "${DATA_ROOT}" -type f \
    \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' \) \
    -ipath "*${dataset_name}*" 2>/dev/null | wc -l)"
  echo "${dataset_name}: ${count}"
  find "${DATA_ROOT}" -type f \
    \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' \) \
    -ipath "*${dataset_name}*" -print 2>/dev/null | head -n 3
done

echo "=== KNOWN JOIN PROBE ==="
find "${DATA_ROOT}" -type f \
  \( -iname 'scene0191_00.mp4' -o -iname 'scene0191_00.sens' \
     -o -ipath '*scene0191_00*pose*' -o -ipath '*scene0191_00*intrinsic*' \) \
  -print 2>/dev/null | head -n 100

echo "=== DATASET SIZES ==="
find "${DATA_ROOT}" -maxdepth 2 -mindepth 1 -type d -print0 2>/dev/null |
  xargs -0 -r du -sh 2>/dev/null | sort -h | tail -n 100
