# Contributing to Ghassan v1 Flash

Thank you for your interest in contributing. This project is a native C++/CUDA language model engine. All contributions must respect the core technical constraints of the codebase.

---

## Core Constraints

- **No Python in the engine.** The training, inference, and data preparation paths are 100% native C++20 and CUDA 12.4. Do not introduce Python scripts, bindings, or dependencies into these paths.
- **CUDA 12.4 minimum.** All GPU kernels must compile with `nvcc` from CUDA Toolkit 12.4 or later.
- **C++20 standard.** Use `--std=c++20`. Avoid compiler extensions unless they are NVCC-specific and documented.
- **No external ML frameworks.** Do not add PyTorch, TensorFlow, ONNX Runtime, or similar as dependencies. The engine uses cuBLAS directly.

---

## What We Accept

- Bug fixes in the C++/CUDA engine (`core/`, `cuda/`, `model/`, `tools/`)
- Performance improvements to CUDA kernels (memory bandwidth, occupancy, fusion)
- New optimizer implementations (follow the pattern in `training/`)
- Improvements to the tokenizer or data pipeline (`tokenizer/`, `training/`, `tools/data_prep_cli.cpp`)
- Documentation corrections in `.github/README.md` or recipe YAML files
- Dockerfile improvements (build reproducibility, layer caching, image size)

---

## How to Contribute

1. Fork the repository.
2. Create a branch from `main` with a descriptive name (`fix/moe-routing-oob`, `perf/mamba2-kernel-fusion`).
3. Make your changes. Ensure the project builds cleanly:
   ```bash
   mkdir build && cd build
   cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
   ninja -j$(nproc)
   ```
   Or via Docker:
   ```bash
   docker build -t ghassan-ai:cuda-latest .
   ```
4. Test your changes manually (`gai_train --help`, `ghassan-ai --help`, `gai_data --help`).
5. Open a pull request against `main` with a clear description of what changed and why.

---

## Code Style

- Follow the existing naming and formatting conventions in each file.
- Keep functions focused and avoid unnecessary abstraction.
- Prefer explicit over implicit — this is a research codebase, not a library.
- Document any non-obvious CUDA kernel behavior with inline comments.

---

## License

By submitting a pull request you agree that your contribution will be licensed under the [Apache License, Version 2.0](../LICENSE).
