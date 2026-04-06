add_library(otter_core STATIC otter.cpp)

target_include_directories(otter_core
    PUBLIC  ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(otter_core PRIVATE buffer.cpp)
target_sources(otter_core PRIVATE backend.cpp)

add_subdirectory(memory)
add_subdirectory(kernels)
