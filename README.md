# OTTER

OTTER is a C++ autodiff library with an eager execution forward model and a modular multi-backend architecture. Tensor semantics, kernel dispatch, and memory management are independent layers; a new backend requires no changes to core abstractions. Backends are pre-built singletons; `cpu_backend()` returns the CPU backend and tensors bind to a backend at creation time.

```cpp
#include "otter/tensor.h"
#include "otter/backends/cpu.h"

Backend& be = otter::cpu_backend();

otter::Tensor a = otter::Tensor::from_data<double>({1,2,3,4,5,6}, {2,3}, be);
otter::Tensor b = otter::Tensor::zeros({2,3}, be);

be.kernel_engine()->dispatch_binary(otter::KernelType::Add, a, a, b);
double val = b.at({1, 2});  // 12.0

// Non-contiguous view; contiguous() materialises to a fresh buffer
otter::Tensor t = a.view({3, 2}, {1, 3});
otter::Tensor c = t.contiguous();
```


## Architecture

`Backend` represents a device instance (e.g., CPU). It owns a `MemoryManager` and a `KernelEngine`. `Tensor` stores a non-owning `Backend*`; device identity is pointer identity (two tensors are on the same device iff their backend pointers are equal). Backend singletons handle their own construction; `cpu_backend()` returns the same `Backend&` on every call.

`KernelEngine` is a registry. Each op family registers a typed `Dispatcher` struct. Adding a new kernel adds a dispatcher and a `KernelType` entry; the registry interface is unchanged. Dispatch is non-virtual at the engine level. Polymorphism lives only inside the `Dispatcher` structs.

`Tensor` is a value type. Multiple tensors share one `Buffer` via `shared_ptr` with independent shape, stride, and offset metadata. `view()` returns a new `Tensor` over the same allocation with no copy. No data moves until `contiguous()` is explicitly called.

`Buffer::data()` requires `Passkey<KernelEngine>`. Only `KernelEngine` subclasses can construct the passkey, so raw-pointer access is structurally restricted to the kernel layer. Every raw-pointer site is auditable with `grep Passkey<KernelEngine>`.


## CPU backend

`CPUKernelEngine` registers a dispatcher per op family (unary, binary, matmul, etc.). Each stores a `CPUKernelEngine*` and forwards to a `cpu_*` method on the engine, which calls the protected `raw_const`/`raw_mutable` helpers. The Passkey invariant holds without giving dispatcher structs direct buffer access.


## Tensor API

- `Tensor::zeros(shape, backend)` — allocates and zero-initialises a tensor
- `Tensor::from_data<double>(data, shape, backend)` — copies data into a new contiguous tensor
- `at({i, j, ...})` — bounds-checked scalar read; throws `std::out_of_range` on bad index in all builds
- `view(shape, stride)` — layout alias over the same buffer
- `contiguous()` — returns `*this` if already contiguous, otherwise copies to a fresh buffer
- `fill_(value)` — in-place fill on contiguous tensors


## Kernels

| Operation | Dispatch call | Notes |
|---|---|---|
| Elementwise add | `dispatch_binary(KernelType::Add, ...)` | Fast path for all-contiguous inputs |
| Elementwise mul | `dispatch_binary(KernelType::Mul, ...)` | Same |
| Elementwise neg | `dispatch_unary(KernelType::Neg, ...)` | Same |
| Matrix multiply | `dispatch_matmul(...)` | Batched; batch dims can be stride-0 |
| Reduce sum | `dispatch_unary(KernelType::ReduceSum, ...)` | Input must be contiguous |
| Reduce to shape | `dispatch_unary(KernelType::ReduceTo, ...)` | Scatter-accumulate; used in broadcast backward |
| Strided copy | `dispatch_copy(...)` | Handles stride-0 broadcast views |
| Scalar read | `dispatch_element_read(...)` | Device-safe; explicit transfer on non-CPU backends |


## Build

```bash
cmake -B build/debug   -GNinja -DCMAKE_BUILD_TYPE=Debug   -DCMAKE_CXX_COMPILER=clang++
cmake -B build/release -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
cmake --build build/debug
ctest --test-dir build/debug --output-on-failure
```

Debug builds enable `-fsanitize=address,undefined`. Both builds compile with `-Wall -Wextra -Werror`.


## Requirements

- C++17 compiler (clang++ 10+)
- CMake 3.20+
- Ninja (optional, but recommended)


## Scope

OTTER currently supports `float64` on CPU. The dtype constraint runs through the full stack — memory allocation, buffer sizing, and kernel dispatch are all typed against it.
