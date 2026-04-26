add_library(otter_core STATIC otter.cpp tensor.cpp)

target_include_directories(otter_core
    PUBLIC  ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(otter_core PRIVATE buffer.cpp)
target_sources(otter_core PRIVATE backend.cpp)

if(OTTER_CUDA)
    target_compile_definitions(otter_core PRIVATE OTTER_CUDA)
    add_library(otter_cuda ALIAS otter_core)
endif()

add_subdirectory(memory)
add_subdirectory(kernels)
add_subdirectory(backends)
add_subdirectory(ops)
add_subdirectory(optim)
