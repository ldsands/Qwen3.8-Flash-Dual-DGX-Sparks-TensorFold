# Official BF16 release plus a separately source-built FP8 executable; both retain the grammar library.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE} AS codecs
ARG LIBJPEG_TURBO_VERSION=3.1.4.1
ARG LIBJPEG_TURBO_SHA256=ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022
ARG LIBWEBP_VERSION=1.6.0
ARG LIBWEBP_SHA256=e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564
RUN mkdir -p /src /opt/codecs && cd /src && \
    curl -fsSL -o jt.tar.gz "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${LIBJPEG_TURBO_VERSION}/libjpeg-turbo-${LIBJPEG_TURBO_VERSION}.tar.gz" && \
    echo "${LIBJPEG_TURBO_SHA256}  jt.tar.gz" | sha256sum -c - && \
    tar -xzf jt.tar.gz && cmake -S libjpeg-turbo-${LIBJPEG_TURBO_VERSION} -B jt-build -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_SHARED=ON -DENABLE_STATIC=OFF -DWITH_TURBOJPEG=ON -DCMAKE_INSTALL_PREFIX=/opt/codecs/jpeg -DCMAKE_INSTALL_LIBDIR=lib && \
    cmake --build jt-build -j"$(nproc)" && cmake --install jt-build && \
    test -e /opt/codecs/jpeg/lib/libturbojpeg.so.0
RUN cd /src && \
    curl -fsSL -o webp.tar.gz "https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-${LIBWEBP_VERSION}.tar.gz" && \
    echo "${LIBWEBP_SHA256}  webp.tar.gz" | sha256sum -c - && \
    tar -xzf webp.tar.gz && cd libwebp-${LIBWEBP_VERSION} && \
    ./configure --prefix=/opt/codecs/webp --enable-libwebpdecoder --disable-static --disable-gl --disable-sdl --disable-png --disable-jpeg --disable-tiff --disable-gif --disable-wic && \
    make -j"$(nproc)" && make install && \
    test -e /opt/codecs/webp/lib/libwebpdecoder.so.3

# FP8 executable: reviewed source plus a separate native patch stack. The official BF16 package stays intact.
FROM ${BASE_IMAGE} AS fp8-build
ARG TF_REF
ARG ZIG_VERSION=0.17.0
ARG ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8
ARG TF_SOURCE_SHA256=d5c11971e9f20d549ae741fecb353bb762f0a4de4d5b0f17f23722754ae5f851
RUN test "$TF_REF" = 93a2e93ecebff8aa52b88e58c5dd222516e69c5f && \
    curl -fsSL -o /tmp/source.tar.gz "https://github.com/ashhart/TensorFold/archive/${TF_REF}.tar.gz" && \
    echo "${TF_SOURCE_SHA256}  /tmp/source.tar.gz" | sha256sum -c - && \
    mkdir /tensorfold && tar -xzf /tmp/source.tar.gz -C /tensorfold --strip-components=1 && \
    curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" && \
    echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1
COPY native-patches /opt/native-patches
COPY scripts/build-native-fp8.sh /opt/build-native-fp8.sh
RUN cd /tensorfold && for p in /opt/native-patches/*.patch; do git apply --check "$p" && git apply "$p"; done && \
    ZIG_BIN=/opt/zig/zig NVCC_BIN=/usr/local/cuda/bin/nvcc bash /opt/build-native-fp8.sh /tensorfold /opt/fp8-output

FROM ${BASE_IMAGE}
ARG TF_RELEASE_VERSION
ARG TF_RELEASE_SHA256
ARG TF_REF
ARG PATCHES_HASH
RUN test "$(uname -m)" = aarch64 && \
    curl -fsSL -o /tmp/tensorfold.tar.gz \
      "https://github.com/ashhart/TensorFold/releases/download/v${TF_RELEASE_VERSION}/tensorfold-${TF_RELEASE_VERSION}-linux-aarch64.tar.gz" && \
    echo "${TF_RELEASE_SHA256}  /tmp/tensorfold.tar.gz" | sha256sum -c - && \
    mkdir -p /opt/tensorfold/native && \
    tar -xzf /tmp/tensorfold.tar.gz -C /opt/tensorfold/native --strip-components=1 && \
    cd /opt/tensorfold/native && sha256sum -c SHA256SUMS && \
    rm /tmp/tensorfold.tar.gz && \
    ln -s /opt/tensorfold/native/bin/tensorfold-native /usr/local/bin/tensorfold-native && \
    tensorfold-native --version && test -s lib/libtfgrammar.so
COPY --from=codecs /src/libjpeg-turbo-3.1.4.1/LICENSE.md /opt/tensorfold/native/LICENSES/libjpeg-turbo.md
COPY --from=codecs /src/libwebp-1.6.0/COPYING /opt/tensorfold/native/LICENSES/libwebp.txt
COPY --from=codecs /src/libwebp-1.6.0/PATENTS /opt/tensorfold/native/LICENSES/libwebp-patents.txt
COPY --from=codecs /opt/codecs/jpeg/lib/libturbojpeg.so* /opt/tensorfold/native/lib/
COPY --from=codecs /opt/codecs/webp/lib/libwebpdecoder.so* /opt/tensorfold/native/lib/
COPY --from=fp8-build /opt/fp8-output/native/bin/tensorfold-native-fp8 /opt/tensorfold/native/bin/
COPY --from=fp8-build /opt/fp8-output/nvcc-version.txt /opt/tensorfold/native/FP8-NVCC.txt
COPY --from=fp8-build /opt/fp8-output/binary.sha256 /opt/tensorfold/native/FP8-SHA256.txt
COPY native/inputs.json /opt/tensorfold/native/FP8-INPUTS.json
COPY native-patches /opt/tensorfold/native/patches
RUN cd /opt/tensorfold/native && sha256sum -c FP8-SHA256.txt && \
    ln -s /opt/tensorfold/native/bin/tensorfold-native-fp8 /usr/local/bin/tensorfold-native-fp8 && \
    tensorfold-native-fp8 --version
ENV TENSORFOLD_LIBTURBOJPEG=/opt/tensorfold/native/lib/libturbojpeg.so.0 \
    TENSORFOLD_LIBWEBP=/opt/tensorfold/native/lib/libwebpdecoder.so.3
RUN pip install --no-cache-dir "huggingface_hub>=1.0"
LABEL tf.patches=${PATCHES_HASH} tf.kernels=present tf.engine=native tf.revision=${TF_REF}
ENV HF_HOME=/root/.cache/huggingface CUDA_CACHE_PATH=/cache/cuda
WORKDIR /workspace
