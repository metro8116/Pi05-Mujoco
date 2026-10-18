#!/usr/bin/env bash
# Install the cloud GPU training and deployment environment with uv.
#
# Create or reuse the project's Python 3.12 .venv and install
# requirements_train.txt, including the configured CUDA 12.8 packages.
# Try the retained wheel cache first, then use the configured package
# mirrors. Clear proxy variables only within this installer and finish
# with a dependency check. Activate .venv separately in the calling shell.
# Usage: bash install_remote.sh

set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PROJECT_DIR"

if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    UV_BIN="$HOME/.local/bin/uv"
else
    echo 'uv is missing. Install it first:' >&2
    echo 'python -m pip install --user uv -i https://mirrors.aliyun.com/pypi/simple' >&2
    exit 1
fi

# Downloads use AutoDL directly. This only changes this script's environment,
# not the parent terminal or the Codex connection.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

# Use the standard system cache for installer downloads.
export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
mkdir -p -- "$UV_CACHE_DIR"

if [[ ! -e .venv ]]; then
    "$UV_BIN" venv --python 3.12 .venv
elif [[ ! -x .venv/bin/python || ! -f .venv/pyvenv.cfg ]]; then
    echo 'Existing .venv is invalid; inspect it or rename it before retrying.' >&2
    exit 1
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), ".venv requires Python 3.12"'

# Use domestic mirrors for ordinary packages and pinned CUDA 12.8 wheels.
# Keep pip-compatible candidate selection for index / flat-wheel sources.
WHEEL_CACHE_DIR="${PI05_WHEEL_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/pi05-wheel-cache}"
RESTORED_FROM_CACHE=false
if [[ -d "$WHEEL_CACHE_DIR" ]]; then
    echo "Trying retained local wheel cache: $WHEEL_CACHE_DIR"
    # Ignore the online wheel listing during an offline cache restore.
    CACHE_REQUIREMENTS="$(mktemp "$PROJECT_DIR/.requirements-cache.XXXXXX")"
    trap 'rm -f -- "$CACHE_REQUIREMENTS"' EXIT
    sed '/^[[:space:]]*--find-links[ =]/d' requirements_train.txt > "$CACHE_REQUIREMENTS"
    if "$UV_BIN" pip install --python "$PROJECT_DIR/.venv/bin/python" \
        --offline --no-index --find-links "$WHEEL_CACHE_DIR" -r "$CACHE_REQUIREMENTS"; then
        RESTORED_FROM_CACHE=true
    fi
fi
if [[ "$RESTORED_FROM_CACHE" != true ]]; then
    "$UV_BIN" pip install --python "$PROJECT_DIR/.venv/bin/python" \
        --default-index https://mirrors.aliyun.com/pypi/simple \
        --index-strategy unsafe-best-match \
        -r requirements_train.txt
fi
"$UV_BIN" pip check --python "$PROJECT_DIR/.venv/bin/python"

echo
echo 'Installed. Activate in your terminal: source .venv/bin/activate'
echo 'EGL/GLFW system libraries, if missing, are documented in readme.md.'
