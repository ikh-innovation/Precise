#!/usr/bin/env bash
# One-shot installer for Module 1 (literal SEEM).
# Creates a dedicated conda env, clones the SEEM repo, installs pinned deps,
# and downloads the seem_focall_v1.pt checkpoint.
#
# Idempotent: safe to re-run. Skips any step whose output already exists.
#
# Usage:  bash src/facade_parsing/setup_seem.sh

set -euo pipefail

PRECISE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THIRD_PARTY="$PRECISE_ROOT/third_party"
SEEM_DIR="$THIRD_PARTY/SEEM"
WEIGHTS_DIR="$THIRD_PARTY/weights"
WEIGHT_FILE="$WEIGHTS_DIR/seem_focall_v1.pt"
CONDA_ENV="${CONDA_ENV:-precise-seem}"
PY_VER="3.10"
SEEM_BRANCH="v1.0"

log() { printf '\033[1;36m[seem-setup]\033[0m %s\n' "$*"; }
log "Precise root      : $PRECISE_ROOT"
log "Conda env         : $CONDA_ENV"
log "Vendor dir        : $SEEM_DIR"
log "Weights file      : $WEIGHT_FILE"

# --- conda env ----------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not on PATH" >&2; exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qxE "$CONDA_ENV"; then
    log "creating conda env"
    conda create -y -n "$CONDA_ENV" "python=$PY_VER"
else
    log "conda env already exists, reusing"
fi

# Use absolute paths to the env's interpreter / pip — `conda activate` in
# non-interactive subshells is unreliable across distros, so we don't rely on it.
ENV_BIN="$CONDA_BASE/envs/$CONDA_ENV/bin"
PY="$ENV_BIN/python"
PIP="$ENV_BIN/pip"
if [ ! -x "$PY" ]; then
    echo "ERROR: env python not found at $PY" >&2; exit 1
fi
log "env python: $($PY --version)"

# Pin setuptools<81 throughout the env: torch 2.1.0's `torch.utils.cpp_extension`
# imports `pkg_resources` (`from pkg_resources import packaging`), which was
# removed in setuptools 81. detectron2-xyz's setup.py loads cpp_extension, so
# without the pin its metadata generation crashes with ModuleNotFoundError.
"$PY" -m pip install -U pip wheel >/dev/null
"$PY" -m pip install 'setuptools<81' >/dev/null

# mpi4py 3.1.5 (SEEM's pin) fails to build under setuptools>=80 — its setup.py
# calls distutils' new_compiler() with a `dry_run` kwarg that newer setuptools
# no longer accepts. Install mpi4py from conda-forge instead so we get a
# pre-built binary against the env's MPI.
if ! "$PY" -c "import mpi4py" 2>/dev/null; then
    log "installing mpi4py from conda-forge (avoids SEEM's broken 3.1.5 pin)"
    conda install -n "$CONDA_ENV" -c conda-forge -y mpi4py
else
    log "mpi4py already importable"
fi

# --- clone SEEM ---------------------------------------------------------------
mkdir -p "$THIRD_PARTY" "$WEIGHTS_DIR"
# Verify by a known file in the working tree, not just `.git/`, since a
# previously interrupted clone can leave behind an empty working dir.
if [ ! -f "$SEEM_DIR/configs/seem/focall_unicl_lang_demo.yaml" ]; then
    log "cloning SEEM ($SEEM_BRANCH)"
    rm -rf "$SEEM_DIR"
    git clone --depth 1 --branch "$SEEM_BRANCH" \
        https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git \
        "$SEEM_DIR"
else
    log "SEEM repo already present, skipping clone"
fi

# --- torch (CPU build, matches SEEM's pinned 2.1.0) ---------------------------
if ! "$PY" -c "import torch; assert torch.__version__.startswith('2.1.')" 2>/dev/null; then
    log "installing torch 2.1.0 + torchvision 0.16.0 (CPU build)"
    "$PIP" install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.1.0 torchvision==0.16.0
else
    log "torch 2.1.x already installed"
fi

# --- SEEM base requirements (drop training-only / problematic pins) ----------
REQ_BASE="$SEEM_DIR/assets/requirements/requirements.txt"
REQ_TMP="$(mktemp)"
# skip: deepspeed (training, hard to build), wandb (logging), infinibatch (training),
# torch/torchvision (already installed from cpu index), mpi4py (installed via
# conda above to avoid the 3.1.5 build break on modern setuptools).
grep -vE '^(deepspeed|wandb|infinibatch|torch==|torchvision==|mpi4py)' "$REQ_BASE" > "$REQ_TMP"
log "installing SEEM base requirements (training-only deps skipped)"
"$PIP" install -r "$REQ_TMP"
rm "$REQ_TMP"

# --- SEEM custom (git-based) requirements ------------------------------------
REQ_CUSTOM="$SEEM_DIR/assets/requirements/requirements_custom.txt"

# When --no-build-isolation is used, the build backend has to already be
# installed in the env. Install the union of backends used by SEEM's custom
# deps (hatchling for einops fork, setuptools for whisper + detectron2-xyz)
# plus the C-extension build tools (ninja, cython) detectron2-xyz needs.
# Re-pin setuptools<81 here in case anything pulled in a newer version.
log "preinstalling build backends + native-build tools"
"$PIP" install --upgrade wheel hatchling editables ninja cython packaging
"$PIP" install 'setuptools<81'

# Install einops fork + whisper without build isolation (they only need
# setuptools/hatchling which we just installed). Install detectron2-xyz the
# same way (it imports our env's torch at setup time and builds C++ ext).
log "installing SEEM custom requirements (no build isolation)"
"$PIP" install --no-build-isolation -r "$REQ_CUSTOM"

# --- project requirements (modules 2 + 3) ------------------------------------
# The user runs all three modules in this single env. Install the project's
# main requirements.txt with --upgrade-strategy only-if-needed so SEEM's
# already-installed pins (numpy 1.23, pillow 9.4) are not bumped past what
# the project requires.
PROJECT_REQ="$PRECISE_ROOT/requirements.txt"
if [ -f "$PROJECT_REQ" ]; then
    log "installing project requirements (modules 2 + 3) into the same env"
    "$PIP" install --upgrade-strategy only-if-needed -r "$PROJECT_REQ"
else
    log "WARN: $PROJECT_REQ not found, skipping project requirements"
fi

# --- weights ------------------------------------------------------------------
if [ ! -s "$WEIGHT_FILE" ]; then
    log "downloading seem_focall_v1.pt (~1.5 GB)"
    curl -L --fail -o "$WEIGHT_FILE.tmp" \
        https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt
    mv "$WEIGHT_FILE.tmp" "$WEIGHT_FILE"
else
    log "weight file already present, skipping download"
fi

# --- smoke import -------------------------------------------------------------
log "smoke-importing SEEM modeling package"
PYTHONPATH="$SEEM_DIR" "$PY" -c "
import torch
from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES
print('OK torch=' + torch.__version__ + '  classes=' + str(len(COCO_PANOPTIC_CLASSES)))
"

cat <<EOF

\033[1;32m[seem-setup] done.\033[0m

Activate the env:    conda activate $CONDA_ENV
Run the pipeline:    python src/client.py
EOF
