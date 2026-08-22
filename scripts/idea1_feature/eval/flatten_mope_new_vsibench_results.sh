#!/usr/bin/env bash
set -euo pipefail

NAME="${1:?usage: $0 EXPERIMENT_OUTPUT_NAME}"
case "${NAME}" in
  e00b_mope_new_projector_only_4b|e02c_mope_new_crossattn_joint_4b|e03a_mope_new_crossattn_two_stage_4b) ;;
  *) echo "Unsupported MoPE-new experiment output: ${NAME}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACE_ROOT="${SPACE_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
source "${SPACE_ROOT}/scripts/_common/env/activate.sh"
OUTPUT_ROOT="${SPACE_OUTPUT_ROOT:-${SPACE_ROOT}/output}"
RESULTS_DIR="${OUTPUT_ROOT}/eval/vsibench/${NAME}"

mapfile -t RESULT_FILES < <(find "${RESULTS_DIR}" -type f -name '*_results.json' -print 2>/dev/null)
mapfile -t SAMPLE_FILES < <(find "${RESULTS_DIR}" -type f -name '*_samples_vsibench_mope_new.jsonl' -print 2>/dev/null)
[[ "${#RESULT_FILES[@]}" -eq 1 && "${#SAMPLE_FILES[@]}" -eq 1 ]] || {
  echo "Refusing to flatten: expected one result and one sample artifact; found ${#RESULT_FILES[@]} and ${#SAMPLE_FILES[@]}" >&2
  exit 3
}
[[ -s "${RESULT_FILES[0]}" && -s "${SAMPLE_FILES[0]}" ]] || {
  echo "Refusing to flatten empty artifacts" >&2
  exit 3
}
python -m json.tool "${RESULT_FILES[0]}" >/dev/null

STAGED="${RESULTS_DIR}.staged.$$"
BACKUP="${RESULTS_DIR}.backup.$$"
trap 'rm -rf -- "${STAGED}"' EXIT
mkdir -p "${STAGED}"
cp -- "${RESULT_FILES[0]}" "${STAGED}/${NAME}_results.json"
cp -- "${SAMPLE_FILES[0]}" "${STAGED}/${NAME}_samples.jsonl"
mv -- "${RESULTS_DIR}" "${BACKUP}"
if mv -- "${STAGED}" "${RESULTS_DIR}"; then
  rm -rf -- "${BACKUP}"
else
  [[ ! -e "${RESULTS_DIR}" ]] && mv -- "${BACKUP}" "${RESULTS_DIR}"
  exit 3
fi
echo "Flattened=${RESULTS_DIR}"
find "${RESULTS_DIR}" -maxdepth 1 -type f -printf '%f\n' | sort
