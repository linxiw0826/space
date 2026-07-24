#!/usr/bin/env bash
set -euo pipefail

AUDIT_ROOT="${1:-/u/lwu9/Space_sensing/data/vsi590k_parta_audit}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${AUDIT_ROOT}/VSI-590K-MetaInfo" "${AUDIT_ROOT}/VSI-590K"

# Part A feasibility gate: metadata + QA schema only (~4.1 GB).
# Do not download the ~236 GB media archives until this audit identifies
# which source subsets can be joined to usable geometry annotations.
hf download nyu-visionx/VSI-590K-MetaInfo \
  --repo-type dataset \
  --local-dir "${AUDIT_ROOT}/VSI-590K-MetaInfo"

hf download nyu-visionx/VSI-590K \
  vsi_590k.jsonl README.md \
  --repo-type dataset \
  --local-dir "${AUDIT_ROOT}/VSI-590K"

python "${SCRIPT_DIR}/inspect_vsi590k_parta_metadata.py" \
  --root "${AUDIT_ROOT}" \
  --output "${AUDIT_ROOT}/schema_audit.json"

echo "Audit written to ${AUDIT_ROOT}/schema_audit.json"
