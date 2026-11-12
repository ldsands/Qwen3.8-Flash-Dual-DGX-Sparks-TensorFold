# Native Flash Next FP8 candidate

This change implements scaled E4M3 KV storage in a separate TensorFold 1.2.7
executable. It has not run on a GPU or loaded a checkpoint. It does not establish
TP=2 correctness, memory fit, fidelity or performance. Existing services were left
alone. No SSH, Docker lifecycle, model weights or inference were used for this work.

BF16 remains the native default because the GPU gates below are outstanding.
The Dockerfile invokes the source builder through Bash, so build contexts do not
need to preserve executable file modes.
Context stays 262144 and concurrency stays 16 in this standalone recipe. The legacy
engine and its twelve patches are unchanged. Changing the native KV default to FP8
is deferred until qualification, rather than treating compilation as a model receipt.

## Changed fork files

- `native-patches/0001-flashnext-scaled-e4m3-cache.patch` contains the nineteen source
  changes for storage, kernels, accounting, snapshots, rank agreement and contracts.
- `native/inputs.json`, `native/kv8_contract.cpp` and `tests/test_native_fp8.py` record
  pins and verify the codec, source application, admission and launcher selection.
- `docker/native.Dockerfile`, `scripts/build-native-fp8.sh` and `.dockerignore` build
  the separate executable and exclude local files from the build context.
- `scripts/engine.sh`, `scripts/config.sh` and `start.sh` select the binary, pass the
  precision flag on both ranks and include native source inputs in image hashing.
- `README.md`, `docs/native-upgrade.md`, `docs/validation.md` and this file document
  the implementation and its limits.

## Selection and integration interface

The interface remains `ENGINE=native` and `KV_DTYPE=bf16|fp8`.
BF16 selects the untouched official `tensorfold-native` executable, with implicit
BF16 caches. FP8 requires `NATIVE_FP8_EXPERIMENTAL=1` and selects
`tensorfold-native-fp8 serve ... --kv-cache fp8` on both ranks, including the
launcher parser check. The FP8 executable refuses BF16 selection for Flash Next.
Other context, concurrency, model IDs, checkpoint revisions, host and port defaults
are unchanged. The gate is a qualification acknowledgement, not a runtime fallback.
An FP8 failure never silently changes precision.

`validate_engine` rejects an unacknowledged FP8 profile before preparation or
lifecycle calls. It retains the ModelOpt NVFP4, context, media, TP paging, YaRN,
KEEP_LATE and decode graph guards. Source admission additionally refuses captured
attention glue, learning, disk persistence/spill, paging and graphs in FP8 mode. Window graphs default off in the FP8 specialization, and an explicit
request for either window or decode graphs is refused.
In-memory prompt snapshot reuse is implemented; disk persistence is refused.
Qwen4 is refused by the specialized executable because it shares the attention
image. Other families are outside this fork's supported interface.

The experimental flag selects code. It does not certify a deployment or authorize
one. Use the same image and settings on both ranks. Do not apply this change to a
running pair during source review.

## Storage and data flow

Each K or V record contains 256 E4M3 bytes followed by a float32 dequantization
scale. The 260-byte head record is aligned for its trailing scale. Records are
ordered by token, then KV head. K and V have separate buffers and separate scales.
A zero row stores scale 1. Finite nonzero rows use `max(abs(row)) / 448`, floored
at the smallest normal float32 scale. Encoding uses CUDA E4M3 round-to-nearest,
ties-to-even with finite saturation. The inputs remain BF16 at their original
rounding points. Indexer keys, pooled keys, queries and recurrent state retain
upstream precision.

The preparer computes each scale over all 256 head elements, writes the scale and
encodes its payload. Every contiguous attention reader dequantizes the record into
the existing BF16 MMA tiles before the original accumulation and softmax. Encoded
bytes are never interpreted as BF16 elements. Existing chunk, one-buffer,
ahead-buffer and fused attention bodies share this reader. The media preparer and
shared-lane entry points share the writer and reader as well. Paged exports trap
if called accidentally, and production admission refuses paging before loading.

Allocation and accounting use `Heads.kvRow()`, now 260 bytes per KV head, including
the scale. This feeds main sequences, MTP head caches, per-stream admission and
snapshot byte counts. Under TP=2 the model has one KV head per rank; the draft
head retains its two heads. The draft-cache charge therefore remains whole-head
on each rank, including the conservative worker budget. Recurrent-state and
indexer costs are unchanged.

Prefill, ordinary decode, concurrent lane-by-lane passes, shared lane entry points,
MTP draft generation and speculative verification all use those storage functions.
Draft rejection overwrites later token records with new payloads and scales, just
as the original path overwrote later BF16 rows. Snapshot copies retain complete
records, including scales, for both model and draft caches. They never recalculate
scales during reuse. Existing rank snapshot ownership, draft-row and restore guards
remain in place.

The rank hello includes the explicit `E4M3ROW1` storage ABI alongside the existing
executable hash, checkpoint identity and execution settings. A mismatch aborts the
link. The FP8 glue loader also requires the kernel symbol
`fn_kv8_abi_e4m3row1`, so an accidentally reused BF16 attention fatbin fails before
sequence execution. The original single-token attention reference utility now
allocates records correctly and compares against an independent scalar quantized
reference. Its GPU check remains unrun.

## Build inputs and packaging

`native/inputs.json` records the reviewed inputs:

- TensorFold 1.2.7 source commit `93a2e93ecebff8aa52b88e58c5dd222516e69c5f`.
- GitHub source archive SHA-256 `d5c11971e9f20d549ae741fecb353bb762f0a4de4d5b0f17f23722754ae5f851`.
- Zig 0.17.0. Its aarch64 archive SHA-256 is
  `9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8`.
- SM121. The container base remains `nvcr.io/nvidia/pytorch:26.07-py3`.

`native-patches/0001-flashnext-scaled-e4m3-cache.patch` is a separate patch stack
against that exact source. It does not reuse or rewrite the twelve legacy patches.
The Dockerfile verifies source and toolchain archives, applies the patch, then
calls `scripts/build-native-fp8.sh`. That script checks Zig's exact version and
uses upstream `zig build native -Dnvcc=... -Dsm=121 -Dstrip=true`. This compiles and
embeds the upstream kernel catalog, including the changed attention image. It does
not accept prebuilt BF16 fatbins as FP8 inputs or use captured Triton packing.
The upstream build tracks compiler version and CUDA header dependencies.

The final image retains the verified official BF16 archive, grammar library,
licenses and JPEG/WebP libraries. The FP8 executable lives beside the original,
so it discovers the same grammar ABI at `../lib/libtfgrammar.so`. Grammar masked
sampling, model loading and the original server correctness guards are unchanged.
The image also records the FP8 binary hash, nvcc version, input pins and source
patch. Image-input hashing includes the source patch and build script. Build
output is not bit-for-bit reproducible across arbitrary CUDA compilers or mutable
container tags; pin a qualified base digest before release. No image was built.

## Offline validation

Commands use local source staging and synthetic CPU inputs only:

```sh
# Apply the patch to the verified source archive, then run from that source directory.
zig build test-flashnext-fp8-host test-flashnext-fp8-refusal-host -j2 --summary all
zig build native compile-modelopt-startup -Doptimize=fast -Dstrip=true -j2 --summary all

# Compile the changed CUDA unit; do not execute it.
nvcc -fatbin -std=c++17 -O3 --fmad=false \
  -gencode=arch=compute_121,code=sm_121 -I zig/kernels/cuda \
  zig/kernels/cuda/flashnext_attn.cu -o flashnext_attn.fatbin
```

The local compiler is NVIDIA CUDA 13.5, nvcc `V13.5.37`, with CUDA runtime/CRT/NVVM
13.5.37 and CCCL `13.3.5.0.6`. This differs from upstream's CUDA 13.3 qualification
records. The CPU compiler is the host C++ compiler. CPU encoding tests execute
CUDA's actual header encoder/decoder without linking or loading a CUDA driver.
They cover every finite code, signed zero, NaNs, ties and neighboring values,
saturation, layout and independent row scales. The scalar Zig reference has its
own encoding and decode checks.

The fork's full offline check command is `scripts/check.sh`. Optional variables
`TF_TEST_SOURCE_ARCHIVE`, `TF_TEST_PATCHED_SOURCE_DIR` and `TF_TEST_CUDA_INCLUDE`
enable clean patch application, byte-for-byte staged-source comparison and the
C++ codec test. Existing `TF_TEST_BINARY` and `TF_TEST_ARCHIVE_AARCH64` /
`TF_TEST_ARCHIVE_X86_64` enable the original CLI, grammar and archive checks. `TF_TEST_FP8_BINARY` checks
the FP8 rank command parser against a missing model path.
No optional variable causes a model download or GPU execution.

The final offline checks passed:

- `scripts/check.sh`: 18 unittest cases with all optional source, archive, parser,
  codec and grammar checks enabled, plus the existing benchmark regressions.
- `test-flashnext-fp8-host` and `test-flashnext-fp8-refusal-host`: 62 tests passed.
- Host-only x86_64 `native` and `compile-modelopt-startup`: 10 of 10 build steps
  passed with `-Doptimize=fast -Dstrip=true`. These builds have no embedded CUDA
  images and cannot serve inference. Version and FP8 rank argument parsing passed.
  A second build with `-Doptimize=debug -Dstrip=true` and a separate prefix also
  passed all ten steps and the final binary parser check.
- The changed attention CUDA unit compiled to an SM121 fatbin with the upstream
  IEEE flags. nvcc emitted two unreachable-loop warnings in fused instantiations.
- Lean comparison: 386 inherited findings, zero new findings. Shell syntax,
  Python compilation and diff whitespace checks passed.
- Clean archive extraction plus checked patch application reproduced every changed
  source file byte for byte. All tracked and new fork text was checked for the
  prohibited project-name string; there were zero matches. No private identifiers
  were added. Legacy patches and their `3dfe459421d5` image-input hash are unchanged.

Source patch SHA-256:
`8b6724cf0a206cb525053b955aea721e69da86696c1cafb3538eb6bec5876fc3`.
Local SM121 attention fatbin SHA-256:
`4147aa7afba531aba8ac5703297e3cf4dc24c52554ec5282d4ff73356c909ae2`.
Final host-only debug executable SHA-256:
`da9ae6aecf6f75c513ced78157c2113792dfd832370408f733da548886b5a1a6`.
Its local path is `work/fp8-host-final/native/bin/tensorfold-native`; it is not an
inference artifact. Native image-input hash: `e47fbf74521c`. This identifies build inputs, not a built
or published image. These local compilation receipts do not qualify the container's
CUDA compiler or an aarch64 executable. No full GPU-enabled native executable or
container was built locally.

## GPU qualification remaining

Compilation does not test synchronization, device conversion behavior, memory
safety or the effect of FP8 error on model outputs. Before promoting the default:

1. Build the complete aarch64 image with a pinned, reviewed CUDA base digest.
   Run packaging, grammar discovery and kernel-symbol checks on that build.
2. On an explicitly available test pair, check finite, zero, extreme and distinct
   per-head cache rows, payload/scale addressing and memory bounds. Exercise all
   contiguous reader variants and media positions, with both head geometries.
3. Verify teacher-forced fidelity against BF16 with per-token margins and error
   metrics. FP8 and BF16 are different arithmetic modes; cross-precision exact
   token equality is not assumed.
4. Within FP8, require drafted/plain, concurrent/solo and resumed/fresh equality.
   Include partial speculative acceptance, cancellation, snapshots with and without
   a draft head, chunk seams and pool-free cache pressure on both ranks.
5. Check required tools, strict schema masking, streamed JSON and images through
   the actual TP=2 server. The preserved CPU grammar ABI is not this receipt.
6. Measure both ranks' peak allocations and admission at native context and the
   requested concurrency. Exercise rank disagreement and peer failure. Measure
   prefill and decode before making performance or memory-fit claims.

Paged KV, disk persistence/spill, captured attention, graphs and learning remain
explicitly unsupported in the FP8 executable. Existing broad upstream BF16/paging
GPU fixtures are not an FP8 qualification receipt. The focused host suite does not
replace the full upstream host suite or a CUDA kernel/model contract run.
