#!/bin/bash
# Run once on ui.pic.es after syncing the tree. Installs packages only.
set -euo pipefail
ROOT=${ROOT:-/data/cvcqml/scratch/$USER/iqp-precision}
export UV_CACHE_DIR=$ROOT/uv-cache UV_PYTHON_INSTALL_DIR=$ROOT/uv-python
cd "$ROOT/tree"
UV_PROJECT_ENVIRONMENT=$ROOT/env-gpu uv sync --frozen
jax_version=$("$ROOT/env-gpu/bin/python" -c "import jax; print(jax.__version__)")
uv pip install --python "$ROOT/env-gpu/bin/python" "jax[cuda12]==$jax_version"
echo "environment ready; jax $jax_version"
