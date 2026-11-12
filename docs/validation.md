# Offline validation, 2026-11-12

TensorFold 1.2.7 source reviewed at
`93a2e93ecebff8aa52b88e58c5dd222516e69c5f`. The official GitHub release API identifies
v1.2.7 as a published, non-prerelease release at that commit. All current checks ran
on x86_64. No Spark SSH, Docker build/lifecycle, model download or live inference
request was used.

## Current release checks

```sh
TF_TEST_BINARY=<verified-x86-release>/bin/tensorfold-native \
TF_TEST_ARCHIVE_AARCH64=<downloads>/tensorfold-1.2.7-linux-aarch64.tar.gz \
TF_TEST_ARCHIVE_X86_64=<downloads>/tensorfold-1.2.7-linux-x86_64.tar.gz \
scripts/check.sh
```

- All **10 unittest cases passed**, including eight real rank-command parses across
  both ranks, context boundaries, concurrency, drafts, thinking and vision settings.
  Each accepted command stopped at the nonexistent absolute model path with
  `HF_HUB_OFFLINE=1`, before loading a checkpoint. The binary reports
  `tensorfold-native 1.2.7`.
- Both official archives matched their companion `.sha256` files and GitHub release
  asset digests. All **23 internal manifest entries per archive** verified, including
  `libtfgrammar.so` and its license notices. ELF headers identify aarch64 and x86_64
  for the respective binary and grammar library.
- Native `capabilities --json` found the adjacent grammar library on x86_64. The
  actual library accepted typed tool-argument JSON under a strict schema and rejected
  a wrong function name, string-valued integer, string-valued array, extra property,
  incomplete reply and malformed schema. This executes the CPU grammar ABI with a
  synthetic character tokenizer, not the server or the checkpoint tokenizer.
- Native configuration refuses incompatible settings before fake Docker/SSH commands
  can execute, including the new 64-image upper bound. Legacy features, health parsing,
  memory-command quoting and the benchmark fake-server regressions passed.
- Shell syntax, Python compilation and `git diff --check` passed.
- Legacy patches and build inputs remain unchanged, with image-input hash
  `3dfe459421d5`. Native image-input hash is `c35ae0742dd8`. These are build-input
  identifiers, not Docker image digests. No native image is claimed to exist.

| Official 1.2.7 artifact | SHA-256 |
| --- | --- |
| Linux aarch64 archive | `250b7cd3d1aa2685cbf011f501c4f10bbff44c6ccd876dc63a7a4bbf337ed01b` |
| Linux x86_64 archive | `319f548ba3d8af501b30108fceccf6d12a375e2851999ae9f5cb88466406a694` |
| aarch64 native executable | `441a38475ea6eb4d4122cb9e534267c9aa598f921e6cc1bf1bb5aae0a4db3ee0` |
| aarch64 libtfgrammar.so | `abad72a824800ec45e07d2c4416bf4249266f70bf86046a0f0770653068eb09e` |
| x86_64 libtfgrammar.so | `64a30d23b890d4bfce2854cbb6659fc636cd36dacf0eb1696e2a5b2d732e2e8b` |

Sources: [official release](https://github.com/ashhart/TensorFold/releases/tag/v1.2.7)
and its downloadable assets; [pinned source](https://github.com/ashhart/TensorFold/tree/93a2e93ecebff8aa52b88e58c5dd222516e69c5f).
Release signatures were not verified. These checks establish checksum agreement
with the official release assets.

## Current limits

The 1.2.6 Zig host-test results below are historical and were **not rerun against
1.2.7**. Current validation covers release packaging, the real CLI parser, the real
CPU grammar library and recipe tests. It does not establish aarch64 execution,
Docker build success, CUDA/NCCL/CX7 operation, checkpoint loading, TP=2 token parity,
image/tool/streamed-JSON responses, context or concurrency memory fit, cancellation
or performance. Both profiles remain unlaunched. The legacy profile stays the default.

---

# Historical 1.2.6 validation, 2026-11-11

All checks ran on an x86_64 controller. No Spark was contacted for this work; no
model, container or checkpoint download was started.

## Recipe and release checks

- `TF_TEST_BINARY=<verified-1.2.6-x86_64>/bin/tensorfold-native scripts/check.sh`:
  **6 unittest cases passed**, with subcases for both ranks, both engines, nine
  incompatible configurations across start/prepare, extra-argument bypasses and
  six health-response layouts. Benchmark fake-server regressions also passed.
- The real 1.2.6 parser accepted both generated rank commands and reached the expected
  nonexistent-model error. No model/config was supplied and Hub access was disabled.
  This caught and fixed the invalid `--vision-max-videos 0` argument during development.
- Shell syntax, Python compilation and `git diff --check`: passed.
- The twelve legacy engine patches remain byte-for-byte unchanged.
- aarch64 archive SHA-256 and all **23 internal manifest entries**: verified.
- x86_64 test archive SHA-256:
  `a8abf9b8d6b1109e716996eba18d5c5ce1dbe013a7808b603f81f218e0bcff3d`.
  Its binary reports `tensorfold-native 1.2.6`.
- Both pinned codec source archives match their Dockerfile SHA-256 values, and the
  license files copied into the image exist in those archives.
- Native image-input hash: `2220068877f3`; legacy image-input hash: `3dfe459421d5`.
  These identify build inputs, not built Docker images.

## TensorFold source checks

Unmodified release source: `6e9af930261a88bb28ac0bfb157ce8e57e9eea07`.
Zig **0.17.0**, downloaded from ziglang.org and checked against its published
x86_64 SHA-256 `1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026`.

```sh
zig build test-flashnext-loader-host test-flashnext-completion-host \
  test-flashnext-compose-host test-flashnext-mtp-plan-host -j2 --summary all
```

**100/100 tests passed; 13/13 build steps succeeded.** These exercise NVFP4 loading
with recording devices, cleanup, composition and head capacity; they do not execute
GPU kernels or prove full-model parity.

The broader unmodified-source command `zig build test-cuda-host -j4 --summary all`
also succeeded: **365/365 build steps; 5,951 tests passed and 28 skipped** (5,979 total).
The log includes intentional warnings, missing optional fixture/library notices and
DeepSeek namespace-hash mismatch diagnostics from a test that deliberately corrupts
a fixture and expects rejection. These were not test failures. Checkpoint-dependent
coverage remained skipped; this result is not full-model or device qualification.

## Still requires hardware

Neither profile was launched. The new aarch64 Dockerfile was not built on a Spark.
CUDA compilation/execution, NCCL/CX7 behavior, image responses, tool/JSON output,
long-context memory fit, concurrent/resumed token parity, cancellation and performance
remain unverified for this fork. No throughput gain is claimed. The original legacy
measurements in the upstream guide are historical evidence, not tests of this fork.
