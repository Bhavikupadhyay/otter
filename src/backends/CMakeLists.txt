target_sources(otter_core PRIVATE cpu_backend.cpp)

if(OTTER_CUDA AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_backend.cpp")
    target_sources(otter_core PRIVATE cuda_backend.cpp cuda_stream.cpp)
endif()
