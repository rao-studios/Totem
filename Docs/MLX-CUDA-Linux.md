# MLX on Linux with CUDA

This document covers how to build Totem with on-device MLX embeddings on a Linux machine using an NVIDIA GPU. It records the full dependency chain, the fork situation, and every source-level patch that was needed to get a clean production build on Ubuntu 24.04 + CUDA 12.9.

The reference machine is:

| Component | Version |
|---|---|
| OS | Ubuntu 24.04 LTS |
| GPU | NVIDIA RTX 3090 (sm_86) |
| CUDA Toolkit | 12.9 |
| GCC | 13 |
| Swift | 6.3+ |

---

## Why a fork?

MLX's upstream CUDA backend (`ml-explore/mlx`) was developed primarily against macOS Accelerate and CUDA 12.x on systems with CUTLASS installed. On a stock Ubuntu 24.04 setup several things break:

1. **Half-precision math ambiguity** — CUDA 12.9 + GCC 13's `<cmath>` exposes `std::tanh(float)` and `std::tanh(long double)` but no `std::tanh(__half)` or `std::tanh(__nv_bfloat16)`. Every call to `cuda::std::` math with a half type is ambiguous.
2. **CUTLASS dependency** — The `quantized/qmm/` CUDA kernels require CUTLASS headers (`cute/numeric/numeric_types.hpp`) which are not installed on this system.
3. **LAPACK header include path** — `mlx/backend/cpu/lapack.h` does `#include <lapack.h>` using angled brackets. SwiftPM's `unsafeFlags` for `-isystem` are not correctly forwarded to the C++ compiler, so the header at `~/.local/include/lapack.h` is not found.
4. **cuDNN Frontend** — The CUDA attention backend requires `cudnn_frontend.h`, which is not in the default include path.

The fixes are committed to two forks:

| Fork | Branch | Role |
|---|---|---|
| `riteshpakala/mlx-swift` | `gab/cuda1` | SwiftPM package — Package.swift, CudaBuild.json |
| `riteshpakala/mlx` | `gab/cuda1` | mlx C++ submodule — lapack.h, unary_ops.cuh, binary_ops.cuh |

---

## System dependencies

### CUDA Toolkit

Install CUDA 12.x from NVIDIA's repository. The toolkit must include `nvcc` and the runtime libraries.

```bash
# Verify installation
nvcc --version
ls /usr/local/cuda/bin/nvcc
ls /usr/local/cuda/lib64/libcudart.so
```

### LAPACK / LAPACKE

The mlx CPU backend uses LAPACK for SVD and QR. Install the C interface headers:

```bash
sudo apt install liblapack-dev liblapacke-dev
```

Or build from source and install to `~/.local`:

```bash
# If using a local install, note the path — you will need it below
ls ~/.local/include/lapack.h
```

### cuDNN Frontend (header-only)

The CUDA attention backend requires the cuDNN Frontend header-only library. Clone it to `~/.local/cudnn-frontend`:

```bash
git clone https://github.com/NVIDIA/cudnn-frontend.git ~/.local/cudnn-frontend
```

No build step is needed — it is headers only.

### nlohmann/json (skip — not needed)

The `-DCUDNN_FRONTEND_SKIP_JSON_LIB` flag in `Package.swift` tells the frontend to omit its nlohmann/json dependency, so you do not need to install it.

---

## Build

```bash
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CUDA_ARCH=sm_86   # match your GPU — sm_86 = RTX 3080/3090/A100
export SPM_CUDA=1
swift build -c release --jobs 2
```

Or use the included script:

```bash
chmod +x build-linux-cuda.sh
./build-linux-cuda.sh
```

Run with the MLX backend:

```bash
.build/release/totem --host 127.0.0.1 --port 8080 --use-mlx
```

### CUDA_ARCH

Set `CUDA_ARCH` to your GPU's compute capability:

| GPU family | CUDA_ARCH |
|---|---|
| RTX 3080 / 3090 / A100 | `sm_86` |
| RTX 4080 / 4090 | `sm_89` |
| H100 | `sm_90` |

Without this env var, nvcc targets all architectures including pre-sm_70, which causes build failures with `__grid_constant__`.

---

## Patches applied

### 1 — `unary_ops.cuh`: half-precision math ambiguity

**File:** `mlx/mlx/backend/cuda/device/unary_ops.cuh`

CUDA 12.9 + GCC 13 exposes both `std::tanh(float)` and `std::tanh(long double)` but no overload for `__half` or `__nv_bfloat16`. Calling `cuda::std::tanh(x)` with a half type triggers an ambiguous overload error.

**Fix:** Added an `is_half_v<T>` type trait and routed all 23 unary math operators (`ArcCos`, `ArcCosh`, `ArcSin`, `ArcSinh`, `ArcTan`, `ArcTanh`, `Abs`, `Ceil`, `Cos`, `Cosh`, `Exp`, `Expm1`, `Floor`, `Log`, `Log2`, `Log10`, `Log1p`, `Round`, `Sigmoid`, `Sin`, `Sinh`, `Sqrt`, `Tan`, `Tanh`) through single-precision C math functions (`::cosf`, `::tanhf`, etc.) when the input type is `__half` or `__nv_bfloat16`:

```cpp
template <typename T>
static constexpr bool is_half_v =
    cuda::std::is_same_v<T, __half> || cuda::std::is_same_v<T, __nv_bfloat16>;

struct Tanh {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (is_half_v<T>) {
      return static_cast<T>(::tanhf(static_cast<float>(x)));
    } else {
      return cuda::std::tanh(x);
    }
  }
};
```

### 2 — `binary_ops.cuh`: half-precision math ambiguity

**File:** `mlx/mlx/backend/cuda/device/binary_ops.cuh`

Same ambiguity pattern for binary math operations.

**Fix:** Applied the same `is_half_v<T>` guard to `FloorDivide`, `Remainder` (`::fmodf`), `NaNEqual` (`::isnan`), `LogAddExp`, `Maximum`, `Minimum`, and `Power` (`::powf`), `ArcTan2` (`::atan2f`).

### 3 — `lapack.h`: ensure `liblapacke-dev` is installed

**File:** `mlx/mlx/backend/cpu/lapack.h`

The file uses `#include <lapack.h>` (angled brackets). The compiler finds this automatically at `/usr/include/lapack.h` — but only if `liblapacke-dev` is installed. `liblapack-dev` alone does not provide the C interface header.

`setup-cuda-ubuntu.sh` installs `liblapacke-dev` as part of its BLAS/LAPACK step, so no patch is required on any machine that has run the setup script.

### 4 — `CudaBuild.json`: exclude CUTLASS-dependent kernels

**File:** `Source/Cmlx/CudaBuild.json` in `riteshpakala/mlx-swift`

The `mlx/mlx/backend/cuda/quantized/qmm/` directory contains CUDA kernels that depend on CUTLASS (`cute/numeric/numeric_types.hpp`). CUTLASS is not installed on this system.

**Fix:** Added the directory to the `"exclude"` list in `CudaBuild.json`. The CudaBuild SwiftPM plugin reads this file to decide which `.cu` files to compile — this is separate from the `exclude:` array in `Package.swift`.

A stub file `mlx/mlx/backend/cuda/quantized/no_qqmm_impl.cpp` provides a runtime error if the quantized matrix-multiply path is ever reached, so the binary still links cleanly.

### 5 — `Package.swift`: cuDNN Frontend + stub inclusion

**File:** `Package.swift` in `riteshpakala/mlx-swift`

Two changes:

- Added `"-I/usr/local/cudnn-frontend/include"` and `"-DCUDNN_FRONTEND_SKIP_JSON_LIB"` to `cxxSettings` so the CUDA attention backend can find `cudnn_frontend.h` without pulling in nlohmann/json. `setup-cuda-ubuntu.sh` clones cudnn-frontend v1.16.0 to `/usr/local/cudnn-frontend`, which is consistent across all user accounts on any Ubuntu machine.
- Removed `"mlx/mlx/backend/cuda/quantized/no_qqmm_impl.cpp"` from `platformExcludes` so SwiftPM compiles the stub (previously it was excluded, which meant the linker had no symbol for the quantized MM path at all).

---

## Package resolution

`Totem/Package.swift` pins `riteshpakala/mlx-swift` at branch `gab/cuda1`. SwiftPM will clone that fork and initialize its `Source/Cmlx/mlx` submodule from `riteshpakala/mlx:gab/cuda1`, which contains the patched headers.

If you run `swift package update`, the resolved commit will advance to the latest commit on `gab/cuda1` in both forks. The patches are permanent there, so this is safe.

---

## Troubleshooting

### `__grid_constant__ annotation is only allowed for architecture compute_70 or later`

You forgot to set `CUDA_ARCH`. Export it before building:

```bash
export CUDA_ARCH=sm_86
```

### `'lapack.h' file not found`

`liblapacke-dev` is not installed. It provides the C interface header (`lapack.h`) at `/usr/include/lapack.h` — `liblapack-dev` alone does not. Run the setup script, or install directly:

```bash
sudo apt install liblapacke-dev
```

### `'cudnn_frontend.h' file not found`

Clone the cuDNN Frontend library:

```bash
git clone https://github.com/NVIDIA/cudnn-frontend.git ~/.local/cudnn-frontend
```

Then verify the path in `Package.swift` matches: `-I/home/totem/.local/cudnn-frontend/include`.

### `'cute/numeric/numeric_types.hpp' file not found`

A `qmm/` CUDA file is being compiled despite the exclusion. Delete `.build/build.db` to force llbuild to re-read the build plan:

```bash
rm .build/build.db
SPM_CUDA=1 swift build -c release --jobs 2
```

### Ambiguous call to `cuda::std::tanh` / `cuda::std::cos` / etc.

The half-precision patches in `riteshpakala/mlx:gab/cuda1` are not in the checkout. Delete the checkout cache and let SwiftPM re-fetch:

```bash
rm -rf .build/checkouts/mlx-swift
swift package resolve
```
