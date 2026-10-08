# Build stage: toolchain + pinned deps + compile
FROM ubuntu:24.04 AS build

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY scripts/fetch_deps.sh scripts/fetch_deps.sh
RUN bash scripts/fetch_deps.sh

COPY CMakeLists.txt ./
COPY src/ src/
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j"$(nproc)"

# Runtime stage: only the binary and the ONNX Runtime shared libraries
FROM ubuntu:24.04

WORKDIR /app
COPY --from=build /src/build/mini-infer /app/mini-infer

# Copy the real shared object only — COPY dereferences the build stage's
# libonnxruntime.so.1 → … symlinks into duplicate copies. Relink the SONAME here.
COPY --from=build /src/third_party/onnxruntime/lib/libonnxruntime.so.1.* /app/lib/
RUN ln -s /app/lib/libonnxruntime.so.1.* /app/lib/libonnxruntime.so.1

# Quiet ONNX Runtime's missing-CA-bundle warning (and any future HTTPS use).
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt

# The build-stage RUNPATH points at the (absent) source tree; this takes precedence.
ENV LD_LIBRARY_PATH=/app/lib

# The ONNX model is not baked in: mount models/ at /app/models, or run
# gateway-only (/predict answers 503).
EXPOSE 8080
ENTRYPOINT ["/app/mini-infer"]
CMD ["8080"]
