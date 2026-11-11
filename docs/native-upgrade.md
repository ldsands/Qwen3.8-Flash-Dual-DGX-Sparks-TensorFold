# Flash Next single-to-dual evaluation, 2026-11-11

This fork keeps the original dual-Spark engine and adds an opt-in native release profile.
It has not been loaded on the Sparks. Offline checks do not establish GPU correctness,
memory fit or a speed improvement. SparkFleet and its registered recipes are unchanged.

## Reviewed sources

- Dual recipe: [MiaAI-Lab dual recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold),
  `dc05c912342ec19467b3b7b686888de763f2ada1`.
- Single recipe: [MiaAI-Lab single recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold),
  `20e15e41f2f514a0d287710f4665ee1b5009ba41`, seventeen patches on TensorFold 1.1.4.
- Native candidate: [TensorFold **1.2.6**](https://github.com/ashhart/TensorFold/releases/tag/v1.2.6), `6e9af930261a88bb28ac0bfb157ce8e57e9eea07`.
  This was the latest published release with downloadable Linux archives during review.
  Main already named 1.2.7; the fork deliberately pins the available release artifacts.
- The legacy engine remains `db281878ddb836fd0df510d8771ecb7e0fe47d26` with the
  original twelve patches. No legacy engine patch was rewritten or discarded.

The new profile installs the official aarch64 release, SHA-256
`d227a6e65516329ace9d460ee8d1d695f0b66eb69005960bed538f831e1f9543`.
The Dockerfile checks the archive and its internal SHA256SUMS, retains its licenses,
and builds the single recipe's pinned JPEG and WebP decoders. Kernels and grammar
support are supplied by the native release. It does not apply the old Triton patches
to an incompatible engine, download model weights while building, or need a GPU to build.

## What transfers

The modern native engine already implements many of the single recipe's ideas.
Using those maintained implementations avoids a second copy of the patch stack.
The table distinguishes available implementations from TP=2 restrictions; it is not
an assertion that every optimized kernel runs for every checkpoint shape.

| Single-Spark work | Decision for this fork |
| --- | --- |
| 0001: quantized vision tower | Native ModelOpt vision loader; JPEG/WebP codecs carried over. Must qualify image answers on both ranks. |
| 0002: 16,384 image tokens | Not in native 1.2.6: its shared CUDA media loader caps at 4,096. Legacy retains 16,384. |
| 0003/0005: FP8 KV and its reader | Native Flash Next lacks the `kv8` option. Retain legacy FP8; native uses BF16 explicitly. |
| 0004/0006: prompt tiles and prefill work | Use native prompt kernels/chunk scheduling. Keep current engine defaults rather than copying old knobs that were removed. |
| 0007–0011: sampled batched heads, candidate sampling, batched gathers, host work | Use native lane/head and gather implementations, including deduplicated pooled PLE reads. The old optional cost-model patch is not imported. Rank coordination remains native. |
| 0012: deferred DeltaNet rollback | Present upstream but explicitly refused at TP=2. Native profile sets KEEP_LATE=0 and rejects an override to 1. |
| 0013: pread/read-ahead and prompt tails | Native Linux PLE reader defaults to pread, with pooled gathers; retain native chunk planning. No claim that all single-Spark tail heuristics transfer. |
| 0014/0015: fused prompt work, shared output and tiled projections | Use native prompt composition and eligible projection kernels. Dense TP splitting changes which shapes qualify. |
| 0016: side-stream shared expert and batched DeltaNet | Upstream implementations guard these paths to one rank. They are not enabled for TP=2 by this fork. |
| 0017: running-product draft stopping | Native MTP confidence rule already supports running-product stopping. Retain its current shared-draft defaults, not a forced old depth. |
| Locale-independent memory check | Ported to both head and worker using `/proc/meminfo`. |
| Stop health parsing | Adapted to legacy `live.connections` and native `running`/`requests_running`; invalid values become zero. |
| Benchmark stats and shared template prefix | Ported bench.py/bench_streams.py plus their fake-server regression. Prefill rate excludes cached tokens. |
| Image check | Ported; discovers the exact served model through the existing client. |
| ABLIT overlay | Not ported: it targets the separate FP4-Local checkpoint and changes model behavior. It is not a speed optimization of the original NVFP4 checkpoint. |

Native TP=2 improvements beyond the single recipe include dense-projection sharding,
shared verification dispatch and coordinated prompt-state retention. The source tests
cover rank routing, snapshot ownership and refusal paths. Those are stronger reasons
to try the new engine than copying TP=1-only kernel switches.

## Features that require the legacy profile

`ENGINE=legacy` is still the default. It keeps the million-token YaRN extension,
FP8 KV, INT4-AutoRound TP=2, videos, and the larger image-token budget. Native uses
NVIDIA ModelOpt NVFP4, at most 262,144 context, BF16 KV and at most 4,096 image tokens;
its Flash Next CUDA media path rejects video rows. `MAX_VIDEOS=0` in the native
configuration denotes that limitation; the flag is omitted because its CLI accepts
only positive video counts. Native paged KV and deferred DeltaNet rollback remain
TP=1-only, and decode graphs conflict with the dense split.

The launcher refuses known incompatible settings before Docker, SSH or preparation.
It also prevents extra CLI arguments from overriding the managed rank/world size,
endpoint or native context/cache/media checks. Ordinary supported serve flags still
pass through. `TF_REF` alone cannot select another binary in the release profile:
updating it requires reviewing and changing the archive version and checksum together.

Some upstream recipe documentation lags its source: it still describes prompt-cache
retention and shared verification as unavailable with TP=2. The 1.2.6 source and
changelog include both. This review uses `cuda_lanes_load.zig`, `cuda_rank_verify.zig`,
`cuda_snap.zig`, `cuda_dense_split.zig`, `cuda_batch_side.zig`,
`cuda_batch_gdn_lanes.zig`, `native/cuda_refusal.zig` and
`vision/qwen_cuda_media.zig` at the pinned commit as the authority.

## Validation and first hardware evaluation

See [validation.md](validation.md) for the actual offline results. No Docker image
was built on the Sparks, no checkpoint was downloaded, and no model was started or
stopped. The existing GLM deployment was left alone.

When a hardware run is explicitly requested, start with a small context and the
same original NVFP4 checkpoint/revision on both nodes. Check health, plain versus
drafted tokens, solo versus concurrent tokens, resumed versus fresh prompts,
forced tools and strict JSON, images, cancellation and memory pressure on both ranks.
Then test native context boundaries and long prompts. Use identical BF16 cache,
context, prompts, sampling and concurrency for the engine comparison; switching
from legacy FP8 to native BF16 is also a precision change, so exact tokens across
those two cache types are not a valid parity requirement.

Measure prefill, first-token latency, decode, peak memory and failures at multiple
concurrency levels. Native BF16 can fit fewer long streams despite improved weight
sharding. The legacy memory estimate is retained as a conservative preflight estimate,
not rebranded as a measurement of 1.2.6. The memory floor remains active on both ranks.
Do not promote the native profile to default based on host tests alone.
