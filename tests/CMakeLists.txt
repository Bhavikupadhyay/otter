find_package(Threads REQUIRED)

# ── CPU tests (always built) ──────────────────────────────────────────────────
add_executable(otter_cpu_tests
    cpu/main_cpu.cpp
    cpu/run_cpu_tests.cpp
    cpu/test_cpu_memory.cpp
    cpu/test_cpu_kernel_impl.cpp)

target_link_libraries(otter_cpu_tests PRIVATE otter_core Threads::Threads)

target_include_directories(otter_cpu_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include)

add_test(NAME otter_cpu_tests COMMAND otter_cpu_tests)

# ── CUDA tests (GPU machine only) ─────────────────────────────────────────────
if(OTTER_CUDA)
    add_executable(otter_cuda_tests
        cuda/main_cuda.cpp
        cuda/run_cuda_tests.cpp
        cuda/test_cuda_memory.cpp
        cuda/test_cuda_streams.cpp
        cuda/test_cuda_concurrency.cpp)

    target_link_libraries(otter_cuda_tests PRIVATE otter_cuda Threads::Threads)
    target_compile_definitions(otter_cuda_tests PRIVATE OTTER_CUDA)
    target_include_directories(otter_cuda_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/include)

    add_test(NAME otter_cuda_tests COMMAND otter_cuda_tests)

    # ── Cross-device tests (CPU↔CUDA transfers) ───────────────────────────────
    add_executable(otter_cross_tests
        cross/main_cross.cpp
        cross/test_device_transfer.cpp)

    target_link_libraries(otter_cross_tests PRIVATE otter_cuda Threads::Threads)
    target_compile_definitions(otter_cross_tests PRIVATE OTTER_CUDA)
    target_include_directories(otter_cross_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/include)

    add_test(NAME otter_cross_tests COMMAND otter_cross_tests)
endif()
