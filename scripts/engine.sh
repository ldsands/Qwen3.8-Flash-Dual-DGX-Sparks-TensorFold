# Engine defaults, sourced after the user's environment and local settings.
ENGINE="${ENGINE:-legacy}"
case "$ENGINE" in
  legacy) ;;
  native)
    TF_REF="${TF_REF:-6e9af930261a88bb28ac0bfb157ce8e57e9eea07}"
    TF_RELEASE_VERSION=1.2.6
    TF_RELEASE_SHA256=d227a6e65516329ace9d460ee8d1d695f0b66eb69005960bed538f831e1f9543
    CONTEXT="${CONTEXT:-262144}"
    KV_DTYPE="${KV_DTYPE:-bf16}"
    MAX_VIDEOS="${MAX_VIDEOS:-0}"
    TENSORFOLD_IMAGE_TOKENS="${TENSORFOLD_IMAGE_TOKENS:-4096}"
    export TF_FLASHNEXT_YARN="${TF_FLASHNEXT_YARN:-0}"
    export TF_FLASHNEXT_KEEP_LATE="${TF_FLASHNEXT_KEEP_LATE:-0}"
    ;;
  *) printf 'ENGINE must be legacy or native, not %s\n' "$ENGINE" >&2; exit 1 ;;
esac

# Fail before Docker, SSH, downloads or lifecycle operations. Stop remains available.
validate_engine() {
  [[ "$ENGINE" == native ]] || return 0
  [[ "$TF_REF" == 6e9af930261a88bb28ac0bfb157ce8e57e9eea07 ]] ||
    die "native uses the pinned 1.2.6 release archive; changing TF_REF alone cannot change that binary"
  [[ "$QUANT" == nvfp4 ]] || die "native TP=2 needs ModelOpt NVFP4; use ENGINE=legacy for INT4-AutoRound"
  [[ "$KV_DTYPE" == bf16 ]] || die "native Flash Next has no FP8 KV cache; use ENGINE=legacy for KV_DTYPE=fp8"
  [[ "$CONTEXT" =~ ^[1-9][0-9]*$ ]] && (( CONTEXT <= 262144 )) ||
    die "native Flash Next supports at most 262144 context; use ENGINE=legacy for YaRN"
  [[ "$MAX_VIDEOS" == 0 ]] || die "native Flash Next has no video input; use ENGINE=legacy"
  [[ "$TENSORFOLD_IMAGE_TOKENS" =~ ^[1-9][0-9]*$ ]] && (( TENSORFOLD_IMAGE_TOKENS <= 4096 )) ||
    die "native Flash Next supports at most 4096 image tokens; use ENGINE=legacy for 16384"
  [[ "$TF_REPO" == https://github.com/ashhart/TensorFold.git ]] ||
    die "native installs the official release archive, not TF_REPO"
  [[ "$TF_FLASHNEXT_YARN" == 0 ]] || die "native Flash Next has no YaRN extension"
  [[ "$TF_FLASHNEXT_KEEP_LATE" == 0 ]] || die "native KEEP_LATE is TP=1 only"
  [[ "${TF_FLASHNEXT_KV_PAGES:-0}" == 0 ]] || die "native paged KV is TP=1 only"
  [[ "${TF_FLASHNEXT_DECODE_GRAPH:-0}" == 0 ]] || die "native decode graphs are incompatible with TP=2 dense splitting"
}

kv_arguments() {
  if [[ "$ENGINE" == legacy ]]; then
    SERVE_ARGS+=(--kv-dtype "$KV_DTYPE")
  fi
  # Native Flash Next uses bf16 implicitly, and rejects --kv-dtype.
}
