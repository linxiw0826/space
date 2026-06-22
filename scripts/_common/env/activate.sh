#!/bin/bash
# =============================================================================
# Per-server profile dispatcher.
#
# Sourced by every train/eval/preprocess script. Decides which
# profile_<name>.sh to source based on:
#   1. $SPACE_PROFILE env var (explicit override, highest priority)
#   2. Auto-detect by directory marker:
#        /data2/wlx           -> new server
#        /home/nvme03/wlx     -> old server
#   3. Fall back silently if neither matches (lets scripts' hardcoded
#      defaults take over, useful when reading code on a third host).
#
# Usage (inside a script under scripts/ideaX/{train,eval}/):
#   source "$(dirname "${BASH_SOURCE[0]}")/../../_common/env/activate.sh"
# For scripts directly under scripts/ (preprocess/, probe/):
#   source "$(dirname "${BASH_SOURCE[0]}")/../_common/env/activate.sh"
# =============================================================================

_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${SPACE_PROFILE:-}" ]; then
    _profile="profile_${SPACE_PROFILE}.sh"
elif [ -d "/data2/wlx" ]; then
    _profile="profile_new.sh"
elif [ -d "/home/nvme03/wlx" ]; then
    _profile="profile_old.sh"
else
    echo "[activate] WARNING: no profile matched (SPACE_PROFILE unset, /data2/wlx and /home/nvme03/wlx both missing). Hardcoded script defaults will be used." >&2
    unset _ENV_DIR
    return 0 2>/dev/null || exit 0
fi

if [ ! -f "${_ENV_DIR}/${_profile}" ]; then
    echo "[activate] ERROR: profile file not found: ${_ENV_DIR}/${_profile}" >&2
    unset _ENV_DIR _profile
    return 1 2>/dev/null || exit 1
fi

source "${_ENV_DIR}/${_profile}"
echo "[activate] sourced ${_profile}"

unset _ENV_DIR _profile
