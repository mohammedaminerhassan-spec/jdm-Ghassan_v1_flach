FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CXX=g++
ENV CC=gcc

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    libomp-dev \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/ghassan_ai
COPY . /workspace/ghassan_ai

RUN mkdir -p build && cd build && \
    cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGAI_ENABLE_CUDA=ON \
        -DGAI_ENABLE_OPENMP=ON \
        -DGAI_FAST_MATH=ON \
        -DGAI_BUILD_TESTS=ON && \
    ninja && \
    ctest --output-on-failure

ENV PATH="/workspace/ghassan_ai/build/bin:${PATH}"

CMD ["ghassan-ai", "--help"]
