#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Project environment bootstrap for CSD3 / similar HPC.
#
# Goals:
#  - Keep /home under quota by pushing caches + heavy artifacts to /local
#  - Disable HF Xet/CAS for stable downloads
# ============================================================================

export MIQ_USER_ALIAS="${MIQ_USER_ALIAS:-dei32}" # replaceable
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" # this file is in the repo/scripts folder
export MIQ_REPO_ROOT="$REPO_ROOT"

# ---- Scratch base on node-local disk (/local) ----
export MIQ_SCRATCH_BASE="/local/${USER}/p28"
export MIQ_CACHE_DIR="${MIQ_SCRATCH_BASE}/cache"
export MIQ_RUNS_DIR="${MIQ_SCRATCH_BASE}/runs"
export MIQ_TMP_DIR="${MIQ_SCRATCH_BASE}/tmp"

mkdir -p "$MIQ_CACHE_DIR" "$MIQ_RUNS_DIR" "$MIQ_TMP_DIR"

# ---- Hugging Face / Transformers caches (into /local) ----
export HF_HOME="${MIQ_CACHE_DIR}/hf"
export HF_DATASETS_CACHE="${MIQ_CACHE_DIR}/hf/datasets"
export HUGGINGFACE_HUB_CACHE="${MIQ_CACHE_DIR}/hf/hub"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"

# ---- Fix CAS/Xet flakiness ----
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=300

# ---- Torch + temp (into /local) ----
export TORCH_HOME="${MIQ_CACHE_DIR}/torch"
export TMPDIR="$MIQ_TMP_DIR"
mkdir -p "$TORCH_HOME" "$TMPDIR"

# ---- Tokenizers perf / stability ----
export TOKENIZERS_PARALLELISM=false

echo "[miq-env] alias     : ${MIQ_USER_ALIAS}"
echo "[miq-env] repo      : ${MIQ_REPO_ROOT}"
echo "[miq-env] hf cache  : ${HF_HOME}"
echo "[miq-env] runs dir  : ${MIQ_RUNS_DIR}"
echo "[miq-env] tmp       : ${TMPDIR}"
echo "[miq-env] xet off   : ${HF_HUB_DISABLE_XET}"
