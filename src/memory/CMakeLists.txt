target_sources(otter_core PRIVATE cpu_memory_manager.cpp)

if(OTTER_CUDA AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_memory_manager.cpp")
    target_sources(otter_core PRIVATE cuda_memory_manager.cpp)
endif()
