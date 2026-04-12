# src/kernels/CMakeLists.txt
target_sources(otter_core PRIVATE cpu_kernel_engine.cpp)

if(OTTER_CUDA AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernel_engine.cpp")
    target_sources(otter_core PRIVATE
        cuda_kernel_engine.cpp
        cuda_fill.cu
        cuda_binary.cu
        cuda_unary.cu
        cuda_inplace.cu
        cuda_copy.cu
        cuda_reduce.cu
        cuda_matmul.cu
    )
    target_link_libraries(otter_core PRIVATE CUDA::cudart)
endif()
