#!/usr/bin/env bash
# Apply the recorded compatibility patches to root/upstream, then compile and install Habitat-GS
# into root/venv. Environment: CUDA_ARCHITECTURES (default 80), BUILD_JOBS (default 6).
set -euo pipefail
root=$(cd "${1:?deployment directory required}" && pwd)
owner=${2:?generated ownership identity required}
test "$(cat "$root/owner")" = "$owner"
test -d "$root/Imath"
# CUDA 13 dropped sm_70 / sm_75. Respect the selected target architecture.
for compat_patch in "$(dirname "$0")"/patches/*.patch; do
  if patch --batch --forward --dry-run -d "$root/upstream" -p1 < "$compat_patch" > /dev/null; then
    patch --batch --forward -d "$root/upstream" -p1 < "$compat_patch"
  elif ! patch --batch --dry-run --reverse -d "$root/upstream" -p1 < "$compat_patch" > /dev/null; then
    echo "Unexpected Habitat-GS source; cannot apply the recorded compatibility patch" >&2
    exit 1
  fi
done
export PATH="$root/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDACXX=/usr/local/cuda/bin/nvcc
export CMAKE_PREFIX_PATH="$root/sysroot/usr"
export CMAKE_INCLUDE_PATH="$root/sysroot/usr/include"
export CMAKE_LIBRARY_PATH="$root/sysroot/usr/lib/x86_64-linux-gnu"
export CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS:-6}
export HABITAT_WITH_CUDA=ON HABITAT_WITH_BULLET=OFF HABITAT_BUILD_GUI_VIEWERS=OFF
log="$root/compile-$(date -u +%Y%m%dT%H%M%S).log"
"$root/venv/bin/python" -m pip install --no-deps --no-build-isolation "$root/upstream" \
  -C cmake.build-type=Release -C cmake.define.CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES:-80} \
  -C cmake.define.REL_BUILD_RPATH=ON \
  -C cmake.define.FETCHCONTENT_SOURCE_DIR_IMATH="$root/Imath" \
  -C cmake.define.CMAKE_CXX_FLAGS="-I$root/sysroot/usr/include" \
  -C cmake.define.CMAKE_C_FLAGS="-I$root/sysroot/usr/include" \
  -C cmake.define.CMAKE_CUDA_FLAGS="-I$root/sysroot/usr/include" \
  -v > "$log" 2>&1
"$root/venv/bin/python" -m pip freeze > "$root/runtime-freeze.txt"
"$root/venv/bin/python" -c 'import habitat_sim; print(habitat_sim.__version__, habitat_sim.cuda_enabled)'
