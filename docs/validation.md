# Offline validation — 2026-11-11

All checks ran on an x86_64 controller. No Spark was contacted for this work; no
model, container or checkpoint download was started. The SparkFleet checkout stayed
clean at `73aff8d47e196a8216a73375afa184f783f6e9e2`.

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
