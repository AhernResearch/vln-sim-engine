#!/usr/bin/env bash
# Build the pinned Habitat-GS runtime into a new, owned deployment directory: its own venv, offline
# wheels and sysroot. No global packages and no existing service or environment is modified.
# Usage: build_habitat_gs.sh DEPLOYMENT_DIR OWNER_ID   (DEPLOYMENT_DIR/owner must contain OWNER_ID)
# Required: HABITAT_GS_PYTHON, a Python 3.11 interpreter used only to create the venv.
# Inputs in DEPLOYMENT_DIR: build-inputs, upstream-source, source-resources, glm-headers,
# imath-source and extra-wheels archives (*.tgz) plus the offline wheel set in wheels/.
set -euo pipefail
root=$(cd "${1:?deployment directory required}" && pwd)
owner=${2:?generated ownership identity required}
test "$(cat "$root/owner")" = "$owner"
test ! -e "$root/venv"
tar -xzf "$root/build-inputs.tgz" -C "$root"
tar -xzf "$root/upstream-source.tgz" -C "$root"
tar -xzf "$root/source-resources.tgz" -C "$root"
tar -xzf "$root/glm-headers.tgz" -C "$root/sysroot"
tar -xzf "$root/imath-source.tgz" -C "$root"
tar -xzf "$root/extra-wheels.tgz" -C "$root"
python_base=${HABITAT_GS_PYTHON:?Python 3.11 interpreter required}
"$python_base" -m venv "$root/venv"
"$root/venv/bin/python" -m pip install --no-index --find-links "$root/wheels" \
  'numpy==2.2.6' 'numpy-quaternion==2024.0.13' 'scipy==1.15.3' 'numba==0.61.2' \
  'pillow==10.4.0' attrs imageio imageio-ffmpeg matplotlib gitpython tqdm \
  scikit-build-core 'cmake==3.31.6' ninja pybind11 pyyaml requests
"$root/venv/bin/python" -m pip install --no-index --find-links "$root/extra-wheels" \
  'pydantic>=2.14,<3' wheel
lib="$root/sysroot/usr/lib/x86_64-linux-gnu"
for name in EGL GL GLX OpenGL; do
  if test -f "/usr/lib/x86_64-linux-gnu/lib${name}.so.1"; then
    ln -sfn "/usr/lib/x86_64-linux-gnu/lib${name}.so.1" "$lib/lib${name}.so"
  fi
done
# Static scan rendering does not need torch, SMPL-X, or avatar dependencies.
# Runtime dependencies above are explicitly installed from the offline wheel set.
bash "$(dirname "$0")/compile_habitat_gs.sh" "$root" "$owner"
