#!/usr/bin/env bash
# Build in a prepared, patched source directory; no GPU, driver, weights or running service is used.
set -euo pipefail
[[ $# == 2 ]] || { echo 'usage: build-native-fp8.sh SOURCE OUTPUT' >&2; exit 2; }
source_dir=$(realpath "$1")
output_dir=$(realpath -m "$2")
zig_bin=${ZIG_BIN:-zig}
nvcc_bin=${NVCC_BIN:-nvcc}
[[ "$("$zig_bin" version)" == 0.17.0 ]] || { echo 'FP8 build requires Zig 0.17.0' >&2; exit 1; }
[[ "$(cat "$source_dir/.zig-version")" == 0.17.0 ]] || exit 1
[[ -f "$source_dir/zig/src/families/flashnext/cuda_kv8.zig" ]] || { echo 'apply native-patches first' >&2; exit 1; }
mkdir -p "$output_dir"
"$nvcc_bin" --version > "$output_dir/nvcc-version.txt"
cd "$source_dir"
# -Dnvcc compiles the whole upstream catalog, including the modified attention image; never reuse BF16 fatbins.
"$zig_bin" build native -Dnvcc="$nvcc_bin" -Dsm=121 -Dstrip=true -Doptimize=fast \
    --prefix "$output_dir" -j"${BUILD_JOBS:-2}"
mv "$output_dir/native/bin/tensorfold-native" "$output_dir/native/bin/tensorfold-native-fp8"
(cd "$output_dir/native"; sha256sum bin/tensorfold-native-fp8) > "$output_dir/binary.sha256"
