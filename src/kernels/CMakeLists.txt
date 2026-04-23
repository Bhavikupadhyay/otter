# src/kernels/CMakeLists.txt
target_sources(otter_core PRIVATE cpu_kernel_engine.cpp)

if(OTTER_CUDA AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernel_engine.cu")
    target_sources(otter_core PRIVATE
        cuda_kernel_engine.cu
        cuda_fill.cu
        cuda_binary.cu
        cuda_unary.cu
        cuda_inplace.cu
        cuda_copy.cu
        cuda_reduce.cu
        cuda_matmul.cu
    )
    # CUDA::cudart is linked implicitly — project() uses LANGUAGES CUDA,
    # so CMake handles runtime linkage for any target containing .cu files.
endif()
