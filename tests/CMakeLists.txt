add_executable(otter_tests
    main.cpp
    test_memory.cpp
    test_kernel.cpp
    test_cpu_kernels.cpp
    test_tensor.cpp
)

target_link_libraries(otter_tests PRIVATE otter_core)

target_include_directories(otter_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)

add_test(NAME otter_tests COMMAND otter_tests)
