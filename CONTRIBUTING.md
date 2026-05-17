# Contributing to Otter

Otter is a C++ autodiff library with CPU and CUDA backends — see the [README](README.md) for an overview.

## Requirements

| Dependency | Version |
|---|---|
| C++17 compiler | GCC 9+ or Clang 10+ |
| CMake | 3.20+ |
| Ninja | any recent (recommended) |
| Test suite | No external test framework — built-in executables linked by CMake |
| CUDA toolkit + NVIDIA GPU | 11.2+ — only for `-DOTTER_CUDA=ON` |

## Build and test

**CPU build**
```bash
cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
./build/tests/otter_cpu_tests
```

**CUDA build** (requires a CUDA-capable machine)
```bash
cmake -B build/cuda -GNinja -DCMAKE_BUILD_TYPE=Debug -DOTTER_CUDA=ON
cmake --build build/cuda
./build/cuda/tests/otter_cpu_tests
./build/cuda/tests/otter_cuda_tests
./build/cuda/tests/otter_cross_tests
```

## Sanitizer requirements (CUDA builds only)

Both tools must pass clean for `otter_cuda_tests` and `otter_cross_tests` before a CUDA-touching PR can merge:

```bash
compute-sanitizer --tool memcheck  ./build/cuda/tests/otter_cuda_tests
compute-sanitizer --tool memcheck  ./build/cuda/tests/otter_cross_tests
compute-sanitizer --tool racecheck ./build/cuda/tests/otter_cuda_tests
compute-sanitizer --tool racecheck ./build/cuda/tests/otter_cross_tests
```

CPU-only contributions do not need a CUDA machine.

## Commit message format

```
<type>(<scope>): <imperative sentence, ≤72 chars>
```

Types: `feat` `fix` `perf` `refactor` `test` `docs` `build` `chore`

One logical change per commit.

## Branch and PR conventions

- Work off `main`; one logical change per PR.
- Every PR must satisfy the checklist in `.github/PULL_REQUEST_TEMPLATE.md`.

## What "done" looks like

- Both builds pass, all tests pass.
- `bytes_allocated() == 0` after the relevant test scope.
- On CUDA builds: `memcheck` and `racecheck` pass for `otter_cuda_tests` and `otter_cross_tests`.
