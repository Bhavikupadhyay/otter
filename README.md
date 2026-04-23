# OTTER

Otter is a C++ autodiff library. It computes reverse-mode gradients over a computation graph built at runtime, runs on a pluggable multi-backend kernel layer, and installs as a CMake static library. CPU and CUDA backends are both supported.

```cpp
#include "otter/tensor.h"
#include "otter/backends/cpu.h"
#include "otter/optim/sgd.h"

otter::Backend& be = otter::cpu_backend();

// Dataset
otter::Tensor x = otter::Tensor::from_data<double>({0.1,0.2, 0.3,0.4, 0.5,0.6, 0.7,0.8}, {4,2}, be);
otter::Tensor y = otter::Tensor::from_data<double>({0.3, 0.7, 1.1, 1.5}, {4,1}, be);

// Weights
otter::Tensor W1 = otter::Tensor::zeros({2,4}, be, otter::DType::Float64, /*requires_grad=*/true);
otter::Tensor b1 = otter::Tensor::zeros({4},   be, otter::DType::Float64, /*requires_grad=*/true);
otter::Tensor W2 = otter::Tensor::zeros({4,1}, be, otter::DType::Float64, /*requires_grad=*/true);
otter::Tensor b2 = otter::Tensor::zeros({1},   be, otter::DType::Float64, /*requires_grad=*/true);

otter::optim::SGD sgd({W1, b1, W2, b2}, /*lr=*/0.01);

for (int step = 0; step < 100; ++step) {
    sgd.zero_grad();

    otter::Tensor h   = x.matmul(W1).add(b1.broadcast_to({4,4})).relu();
    otter::Tensor out = h.matmul(W2).add(b2.broadcast_to({4,1}));
    otter::Tensor mse = out.sub(y).mul(out.sub(y)).mean();

    mse.backward();
    sgd.step();
}
```

The forward pass traces a computation graph. `.backward()` traverses it in reverse topological order and accumulates `∂loss/∂param` into every leaf tensor marked `requires_grad=true`. The graph is freed after each pass unless `retain_graph=true` is set.

---

## Tensor ops

```
Binary:           add  sub  mul  div
Unary:            neg  exp  log  sqrt  relu
Matmul:           matmul  (batched; batch dims accept stride-0 broadcast views)
Reductions:       sum  mean
Views (diffable): reshape  transpose  slice  broadcast_to
Layout:           view  contiguous
```

All binary, unary, matmul, and reduction ops participate in the graph. View ops wire a backward node that routes gradients back through the layout change. `view`, `contiguous`, and `fill_` are not differentiable.

---

## Autograd

```cpp
otter::Tensor a = otter::Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, be,
                                                   /*requires_grad=*/true);
otter::Tensor loss = a.mul(a).sum();   // loss = Σaᵢ²
loss.backward();
// a.grad() == [2.0, 4.0, 6.0]   (∂loss/∂aᵢ = 2aᵢ)
```

- `retain_graph=true` keeps the graph intact for a second backward call
- `detach()` returns a shallow copy with no grad history; the buffer is shared
- `NoGradGuard` disables graph construction — used internally by all backward passes and optimizer steps
- Gradients accumulate across `backward()` calls; call `zero_grad()` to reset before the next pass

---

## Optimizer

SGD with optional momentum and weight decay:

```cpp
// Basic
otter::optim::SGD sgd({W, b}, /*lr=*/0.01);

// With momentum
otter::optim::SGD sgd({W, b}, /*lr=*/0.01, /*momentum=*/0.9);

// With weight decay
otter::optim::SGD sgd({W, b}, /*lr=*/0.01, /*momentum=*/0.0, /*weight_decay=*/1e-4);
```

Parameters are passed by value. The optimizer shares the same `Buffer` as the caller's tensors — `step()` updates are visible through the original handles immediately. `set_lr(double)` adjusts the learning rate between steps.

---

## CUDA backend

```cpp
#include "otter/backends/cuda.h"

otter::Backend& cuda = otter::cuda_backend();

otter::Tensor d = otter::Tensor::from_data<double>({1.0, 2.0, 3.0}, {3}, cuda);
d.fill_(5.0);

// Move between devices
otter::Tensor host = d.cpu();
otter::Tensor back = host.cuda();
```

`cuda_backend()` allocates via `cudaMallocManaged` (unified memory). `Tensor::cuda()` and `Tensor::cpu()` copy data across devices in one `cudaMemcpy` call. Non-contiguous CUDA tensors must be made contiguous before moving to CPU.

Double-precision atomic accumulations (used by `sum` and `reduce_to`) use a portable CAS loop that works on SM 2.0+. Hardware `atomicAdd(double*)` (SM 6.0+) is not required.

---

## Thread safety

`grad()`, `zero_grad()`, and `accumulate_grad()` are mutex-protected via `GradAccumulator::mtx`. Concurrent reads and writes on the same leaf's gradient are safe from any thread.

Concurrent `backward()` calls on separate computation graphs that share a leaf weight are safe. The canonical data-parallel pattern — N threads each computing loss and calling `backward()`, one shared weight tensor — works without external synchronization.

`SGD::step()` is not safe for concurrent calls on the same optimizer. `dispatch_scale` and `dispatch_axpy` write directly into parameter and velocity buffers. Call `step()` from one thread after all backward passes have joined.

---

## Architecture

`Tensor` is a value type. Copies share one `Buffer` via `shared_ptr` with independent shape, stride, and offset metadata. Views are zero-copy; `contiguous()` copies only when the strides are non-standard.

`Backend` owns a `MemoryManager` and a `KernelEngine`. `cpu_backend()` and `cuda_backend()` are program-lifetime singletons. Tensors bind to a backend at creation. Two tensors are on the same device iff their backend pointers are equal.

`KernelEngine` is a dispatcher registry. Each op family registers a typed `Dispatcher` struct in the backend constructor. Adding a kernel means adding a dispatcher and a `KernelType` entry; the registry interface does not change. The CPU and CUDA engines register independently — an op missing from CUDA throws a `std::runtime_error` with the exact dispatcher name, rather than silently falling back.

`Buffer::data()` requires `Passkey<KernelEngine>`. Only `KernelEngine` subclasses can construct the passkey. Every raw-pointer access site is auditable with `grep Passkey<KernelEngine>`.

`Operation::execute()` owns all graph wiring. `forward()` and `backward()` are pure compute. Leaf tensors get `grad_accum_` at creation; computed tensors get `grad_op_` set by `execute()` and nulled after backward unless `retain_graph=true`.

---

## Debug utilities

`include/otter/debug.h` provides header-only utilities over the public Tensor API:

```cpp
#include "otter/debug.h"

otter::has_nan(t);              // bool
otter::has_inf(t);              // bool
otter::max_abs_diff(a, b);      // double
otter::shape_str(t);            // "[2, 3]"
otter::dtype_str(t);            // "Float64"
t.print("label");               // shape, dtype, values to stdout
t.to_vector<double>();          // host-side copy in logical row-major order
```

---

## Build

### CPU only

```bash
cmake -B build/debug   -GNinja -DCMAKE_BUILD_TYPE=Debug   -DCMAKE_CXX_COMPILER=clang++
cmake -B build/release -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
cmake --build build/debug
./build/debug/tests/otter_tests
```

### With CUDA (requires NVIDIA GPU and CUDA toolkit)

```bash
cmake -B build/cuda/debug   -GNinja -DCMAKE_BUILD_TYPE=Debug   \
      -DCMAKE_CXX_COMPILER=clang++ -DOTTER_CUDA=ON
cmake -B build/cuda/release -GNinja -DCMAKE_BUILD_TYPE=Release  \
      -DCMAKE_CXX_COMPILER=clang++ -DOTTER_CUDA=ON
cmake --build build/cuda/debug
cmake --build build/cuda/release
./build/cuda/debug/tests/otter_tests
./build/cuda/release/tests/otter_tests
```

By default CMake targets the GPU in the machine. To target a specific architecture (e.g. for CI or cross-compilation), pass `-DCMAKE_CUDA_ARCHITECTURES=75` at configure time. Use `native` to auto-detect.

Debug builds enable `-fsanitize=address,undefined` (CPU only; incompatible with the CUDA runtime). Both compile with `-Wall -Wextra -Werror` scoped to C++ only — nvcc uses its own error flags.

---

## Requirements

- C++17 compiler (Clang 10+ or GCC 9+)
- CMake 3.20+
- Ninja (optional, faster builds)
- Linux or macOS for CPU builds (`mmap` and `posix_memalign` required)
- CUDA toolkit 11+ and an NVIDIA GPU (SM 2.0+) for `-DOTTER_CUDA=ON`
