# Engine defaults, sourced after the user's environment and local settings.
ENGINE="${ENGINE:-legacy}"
case "$ENGINE" in
  legacy) ;;
  native)
    TF_REF="${TF_REF:-93a2e93ecebff8aa52b88e58c5dd222516e69c5f}"
    TF_RELEASE_VERSION=1.2.7
    TF_RELEASE_SHA256=250b7cd3d1aa2685cbf011f501c4f10bbff44c6ccd876dc63a7a4bbf337ed01b
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
  [[ "$TF_REF" == 93a2e93ecebff8aa52b88e58c5dd222516e69c5f ]] ||
    die "native uses pinned 1.2.7 release and source archives; changing TF_REF alone cannot update them"
  [[ "$QUANT" == nvfp4 ]] || die "native TP=2 needs ModelOpt NVFP4; use ENGINE=legacy for INT4-AutoRound"
  [[ "$KV_DTYPE" == bf16 || "$KV_DTYPE" == fp8 ]] || die "native KV_DTYPE must be bf16 or fp8"
  if [[ "$KV_DTYPE" == fp8 && "${NATIVE_FP8_EXPERIMENTAL:-0}" != 1 ]]; then
    die "native FP8 has no GPU qualification yet; NATIVE_FP8_EXPERIMENTAL=1 enables the source-built candidate, or use KV_DTYPE=bf16"
  fi
  if [[ "$KV_DTYPE" == fp8 ]]; then
    [[ "${TF_FLASHNEXT_WINDOW_GRAPH:-0}" == 0 ]] || die "native FP8 window graphs are unqualified"
  fi
  [[ "$CONTEXT" =~ ^[1-9][0-9]*$ ]] && (( CONTEXT <= 262144 )) ||
    die "native Flash Next supports at most 262144 context; use ENGINE=legacy for YaRN"
  [[ "$MAX_VIDEOS" == 0 ]] || die "native Flash Next has no video input; use ENGINE=legacy"
  [[ "$TENSORFOLD_IMAGE_TOKENS" =~ ^[1-9][0-9]*$ ]] && (( TENSORFOLD_IMAGE_TOKENS <= 4096 )) ||
    die "native Flash Next supports at most 4096 image tokens; use ENGINE=legacy for 16384"
  [[ "$TENSORFOLD_MAX_IMAGES" =~ ^[1-9][0-9]*$ ]] && (( TENSORFOLD_MAX_IMAGES <= 64 )) ||
    die "native Flash Next supports 1 to 64 images per request"
  [[ "$TF_REPO" == https://github.com/ashhart/TensorFold.git ]] ||
    die "native builds only the pinned official release and source archives, not a TF_REPO override"
  [[ "$TF_FLASHNEXT_YARN" == 0 ]] || die "native Flash Next has no YaRN extension"
  [[ "$TF_FLASHNEXT_KEEP_LATE" == 0 ]] || die "native KEEP_LATE is TP=1 only"
  [[ "${TF_FLASHNEXT_KV_PAGE_TEST:-0}" == 0 ]] || die "native TP=2 does not support the paged KV diagnostic"
  [[ "${TF_FLASHNEXT_KV_PAGES:-0}" == 0 ]] || die "native paged KV is TP=1 only"
  [[ "${TF_FLASHNEXT_DECODE_GRAPH:-0}" == 0 ]] || die "native decode graphs are incompatible with TP=2 dense splitting"
}

kv_arguments() {
  if [[ "$ENGINE" == legacy ]]; then
    SERVE_ARGS+=(--kv-dtype "$KV_DTYPE")
  elif [[ "$KV_DTYPE" == fp8 ]]; then
    SERVE_ARGS+=(--kv-cache fp8)
  fi
}

# Precision selects a real executable with its own cache ABI, on both ranks and in the parser check.
engine_binary() {
  if [[ "$ENGINE" == native && "$KV_DTYPE" == fp8 ]]; then
    printf '%s\n' tensorfold-native-fp8
  else
    printf '%s\n' tensorfold-native
  fi
}
