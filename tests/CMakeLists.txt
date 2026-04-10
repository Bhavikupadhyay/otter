find_package(Threads REQUIRED)

add_executable(otter_tests
    main.cpp
    test_memory.cpp
    test_kernel.cpp
    test_cpu_kernels.cpp
    test_tensor.cpp
    test_autograd.cpp
    test_broadcast.cpp
    test_matmul.cpp
    test_views.cpp
    test_math_ops.cpp
    test_factories_debug.cpp
    test_mean.cpp
    test_optim.cpp
    test_thread_safety.cpp
)

target_link_libraries(otter_tests PRIVATE otter_core Threads::Threads)

target_include_directories(otter_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)

add_test(NAME otter_tests COMMAND otter_tests)
