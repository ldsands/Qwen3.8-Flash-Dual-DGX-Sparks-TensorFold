# Flash Next on two DGX Sparks — independent fork

This is [ldsands's fork](https://github.com/ldsands/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold)
of MiaAI-Lab's dual-Spark recipe. The native profile has **not been loaded or
benchmarked on the Sparks**.

The existing patched engine remains the default (`ENGINE=legacy`). An opt-in
`ENGINE=native` profile uses pinned TensorFold **1.2.7**, including its newer TP=2
implementation and compatible single-Spark optimizations. It uses the original
`nvidia/Qwen3.8-Flash-Next-NVFP4` checkpoint at its unchanged revision.

Read the [evaluation and compatibility matrix](docs/native-upgrade.md) before selecting
an engine, and the [validation record](docs/validation.md) for checks actually run.
Native currently needs BF16 KV, at most 262,144 context and 4,096 image tokens, with
no video input or INT4 TP=2. Legacy retains FP8 KV, million-token YaRN, videos and INT4.

## Configuration and operation

Run on the head Spark, with key-based SSH to the worker. Copy
`scripts/local.sh.example` to `scripts/local.sh` and set `WORKER`. Use a separate local
configuration for this fork.

For a future explicitly requested native test, put `ENGINE=native` in local.sh.
Keeping it there ensures start, prepare, stop and publish select the same profile.
Alternatively prefix **every** command with `ENGINE=native`:

```sh
ENGINE=native PULL=0 scripts/prepare.sh   # builds image and prepares both checkpoints
ENGINE=native ./start.sh                 # loads both ranks; NOT run during this evaluation
ENGINE=native ./stop.sh                  # stops only this fork's native containers
```

`ENGINE=legacy` selects the original engine. Both profiles default to `0.0.0.0:6060`,
serve the exact Hugging Face model ID, and have separate container names, images,
state, logs, JIT caches and NFS volume names. They share the user's existing Hugging
Face cache. Do not run them together or beside another large model. No new GHCR image
has been published; preparation builds locally when the fork's image is unavailable.

The [original deployment guide](docs/upstream-readme.md) retains the detailed
network/cache setup instructions and historical measurements. Its old repository,
image/container names, default port and model aliases refer to upstream; this fork's
configuration and compatibility matrix take precedence.

## Offline checks

```sh
scripts/check.sh
```

To also parse both real rank commands without loading weights, set `TF_TEST_BINARY`
to a checksum-verified Linux x86_64 TensorFold 1.2.7 binary. The test uses a nonexistent
absolute model path with network access disabled through `HF_HUB_OFFLINE=1`.
It does not connect to either Spark. The same setting enables grammar discovery and
strict-schema acceptance/rejection checks against the adjacent `lib/libtfgrammar.so`.
Set `TF_TEST_ARCHIVE_AARCH64` and `TF_TEST_ARCHIVE_X86_64` to the official local
1.2.7 `.tar.gz` files to check their outer hashes, internal manifests, grammar
licenses and ELF architectures. These checks never download archives automatically.
The check script uses `uv run python`.

The original exactness, tools, needle, long-context and prompt-reuse clients remain
in `tools/`. New `bench.py`, `bench_streams.py` and `visioncheck.py` come from the
single-Spark recipe; run them only against a server you explicitly intend to test.
Use `MODEL` for a different checkpoint and `API_KEY` when authentication is enabled.
Benchmark prefill accounting now excludes cached template tokens.

See [CREDITS.md](CREDITS.md) and [LICENSE](LICENSE) for upstream authorship and licensing.
