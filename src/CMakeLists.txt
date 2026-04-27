add_library(otter_core STATIC core/otter.cpp core/tensor.cpp)

target_include_directories(otter_core
    PUBLIC  ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(otter_core PRIVATE
    core/buffer.cpp
    core/backend.cpp
    core/autograd/graph_executor.cpp
)

add_subdirectory(cpu)

if(OTTER_CUDA)
    target_compile_definitions(otter_core PRIVATE OTTER_CUDA)
    add_library(otter_cuda ALIAS otter_core)
    add_subdirectory(cuda)
endif()

add_subdirectory(ops)
add_subdirectory(optim)
