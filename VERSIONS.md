# Versions & build matrix

## ggml (Git submodule)

- **Path**: `vendor/ggml` → [ggml-org/ggml](https://github.com/ggml-org/ggml).
- **Pinned release**: **v0.9.8** (gitlink `2fb6431f67dd505584a9fefe94fd2866b944c85f`).
- **Clone**: `git submodule update --init --recursive` (or clone with `git clone --recurse-submodules …`).
- **Bump**: `cd vendor/ggml && git fetch origin && git checkout <tag> && cd ../.. && git add vendor/ggml && git commit`.
- **Reported version** (upstream `CMakeLists.txt` at this pin): **0.9.8**.

## `ggml-sys` Cargo features → CMake

| Feature | CMake option |
|---------|----------------|
| `native` | `GGML_NATIVE=ON` |
| `metal` | `GGML_METAL=ON`, `GGML_METAL_EMBED_LIBRARY=ON` |
| `blas` | `GGML_BLAS=ON` (on Apple, `GGML_ACCELERATE` follows ggml defaults when BLAS on) |
| `cuda` | `GGML_CUDA=ON` |
| `vulkan` | `GGML_VULKAN=ON` |
| `hip` | `GGML_HIP=ON` |
| `musa` | `GGML_MUSA=ON` |
| `opencl` | `GGML_OPENCL=ON` |
| `rpc` | `GGML_RPC=ON` |
| `sycl` | `GGML_SYCL=ON` |
| `webgpu` | `GGML_WEBGPU=ON` |
| `openvino` | `GGML_OPENVINO=ON` |
| `hexagon` | `GGML_HEXAGON=ON` |
| `cann` | `GGML_CANN=ON` |
| `zendnn` | `GGML_ZENDNN=ON` |
| `zdnn` | `GGML_ZDNN=ON` |
| `virtgpu` | `GGML_VIRTGPU=ON` |

**Platform guards**

- `metal` is **Apple targets only** (build script will `panic!` elsewhere).
- `cuda` is **rejected on Apple** targets.
- `vulkan` requires the Vulkan toolchain to be discoverable by CMake, including `glslc`.

Default workspace build uses **CPU** only and disables Apple Metal/BLAS in CMake unless you enable the corresponding features.

## Override ggml path

Set `GGML_SRC` to point at another checkout when experimenting:

```bash
export GGML_SRC=/path/to/ggml
cargo build -p ggml-sys
```

## `qwen3-tts`

- Direct Rust implementation on top of `ggml` / `ggml-sys`.
- Current implemented pieces:
  - GGUF validation and metadata access
  - direct tokenizer loading from GGUF metadata
  - `encode_for_tts()` request formatting
- Reference only:
  - [predict-woo/qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) for module boundaries, metadata keys, and tensor naming

| Feature | Effect |
|---------|--------|
| `hf` | Hugging Face download helpers. |
| `metal` | Prefer Metal at runtime on Apple builds; fall back to CPU if Metal init fails. |
| `vulkan` | Prefer Vulkan at runtime on non-Apple builds; fall back to CPU if Vulkan init fails. |
| `native` | Enable host-CPU-specific ggml kernels for less portable but faster CPU binaries. |

## Vulkan prerequisites

- Linux: install a Vulkan loader / headers plus `glslc` before building `--features vulkan`.
- Windows: install the Vulkan SDK so CMake can find both Vulkan and `glslc`.
- Apple: the project currently keeps Metal as the preferred GPU backend; enabling `vulkan` does not change runtime selection there.
