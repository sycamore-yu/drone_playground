#!/usr/bin/env bash
# Project-local build matching LSY's pinned acados release.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/tmp/p3p4/optimization/acados"
mkdir -p "$(dirname "$DEST")"
if [ ! -d "$DEST/.git" ]; then
  git clone --branch v0.5.1 --depth 1 --recursive --shallow-submodules https://github.com/acados/acados.git "$DEST"
fi
test "$(git -C "$DEST" describe --tags --exact-match)" = "v0.5.1"
cmake -S "$DEST" -B "$DEST/build" -DCMAKE_INSTALL_PREFIX="$DEST" \
  -DCMAKE_BUILD_TYPE=Release -DACADOS_WITH_QPOASES=ON \
  -DCMAKE_INSTALL_RPATH='$ORIGIN' \
  -DBLASFEO_TARGET=GENERIC -DHPIPM_TARGET=GENERIC -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$DEST/build" --target install --parallel 4
if [ ! -f "$DEST/bin/renderer-0.2.0" ]; then
  curl --fail --location --retry 2 \
    https://github.com/acados/tera_renderer/releases/download/v0.2.0/t_renderer-v0.2.0-linux-amd64 \
    -o "$DEST/bin/t_renderer"
  chmod +x "$DEST/bin/t_renderer"
  touch "$DEST/bin/renderer-0.2.0"
fi
"$ROOT/.pixi/envs/default/bin/python" -m pip install --no-deps \
  --target "$(dirname "$DEST")/python-deps" 'future-fstrings==1.2.0' 'matplotlib==3.10.6'
printf '\nACADOS_SOURCE_DIR=%s\nACADOS_INSTALL_DIR=%s\n' "$DEST" "$DEST"
git -C "$DEST" rev-parse HEAD
sha256sum "$DEST/bin/t_renderer"
