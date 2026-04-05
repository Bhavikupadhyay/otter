add_library(otter_core STATIC otter.cpp)

target_include_directories(otter_core
    PUBLIC  ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# Sources are appended here as files are added in subsequent steps.
