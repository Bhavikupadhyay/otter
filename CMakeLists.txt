cmake_minimum_required(VERSION 3.20)

# ── OTTER_CUDA must be set before project() so LANGUAGES is correct ──────────
option(OTTER_CUDA "Enable CUDA backend (requires CUDA toolkit and NVIDIA GPU)" OFF)

if(OTTER_CUDA)
    project(otter VERSION 0.1.0 LANGUAGES CXX CUDA)
else()
    project(otter VERSION 0.1.0 LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(OTTER_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Separate device code compilation — required for device-side linking.
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

add_compile_options(-Wall -Wextra -Werror -Wshadow -Wconversion -Wsign-conversion)

add_compile_options($<$<CONFIG:Debug>:-O0>)
add_compile_options($<$<CONFIG:Debug>:-g>)

# ASan/UBSan are incompatible with the CUDA runtime. Enable only for CPU builds.
if(NOT OTTER_CUDA)
    add_compile_options($<$<CONFIG:Debug>:-fsanitize=address,undefined>)
    add_link_options($<$<CONFIG:Debug>:-fsanitize=address,undefined>)
endif()

add_compile_options($<$<CONFIG:Release>:-O2>)
add_compile_options($<$<CONFIG:Release>:-DNDEBUG>)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
