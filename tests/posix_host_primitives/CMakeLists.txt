cmake_minimum_required(VERSION 3.20)
project(posix_host_primitives_test C)
# The Win32 primitives win32_compat.c implements on POSIX: vectored exception
# handlers on signals, SuspendThread/GetThreadContext, SRW condition variables
# and WaitOnAddress. On Windows they are the OS's own.
if(WIN32)
    message(FATAL_ERROR "This regression covers the POSIX host layer")
endif()
set(CMAKE_C_STANDARD 11)
add_subdirectory(../../src/platform platform)
add_executable(posix_host_primitives_test test_main.c)
target_link_libraries(posix_host_primitives_test PRIVATE platform)
enable_testing()
add_test(NAME posix_host_primitives_test COMMAND posix_host_primitives_test)
