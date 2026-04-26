target_sources(otter_core PRIVATE
    cuda_backend.cpp
    memory/cuda_memory_manager.cpp
    engine/stream.cpp
    engine/cuda_kernel_engine.cu
    kernels/fill.cu
    kernels/binary.cu
    kernels/unary.cu
    kernels/inplace.cu
    kernels/copy.cu
    kernels/reduce.cu
    kernels/matmul.cu
)
# CUDA::cudart is linked implicitly — project() uses LANGUAGES CUDA,
# so CMake handles runtime linkage for any target containing .cu files.
