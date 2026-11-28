FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Kaggle's Tesla T4 uses compute capability 7.5. Keep the CI image targeted
# at the actual training accelerator instead of compiling unused GPU targets.
ARG CUDA_ARCHITECTURES=75

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

RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGAI_ENABLE_CUDA=ON \
        -DGAI_ENABLE_OPENMP=ON \
        -DGAI_FAST_MATH=OFF \
        -DGAI_CUDA_WERROR=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    cmake --build build --parallel 2

ENV PATH="/workspace/ghassan_ai/build/bin:${PATH}"

CMD ["ghassan-ai", "--help"]
