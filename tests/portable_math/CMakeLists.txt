cmake_minimum_required(VERSION 3.20)
project(portable_math_test C)
# RECOMP_PORTABLE_MATH must give the same bits under every compiler: build
# this with MSVC on Windows and with gcc/clang on Linux, and the printed hash
# must be the same (pass -DEXPECTED_HASH=0x... to make it a hard check).
set(CMAKE_C_STANDARD 11)
add_executable(portable_math_test test_main.c ../../src/kernel/recomp_math.c)
if(NOT MSVC)
    # No fused multiply-add anywhere: the harness's own input generator must
    # round the same as MSVC's too (gcc contracts by default when the target
    # has FMA, e.g. -march=native).
    target_compile_options(portable_math_test PRIVATE -ffp-contract=off)
    target_link_libraries(portable_math_test PRIVATE m)
endif()
set(EXPECTED_HASH "" CACHE STRING "the hash every build must print, e.g. 0x0123456789abcdef")
if(EXPECTED_HASH)
    target_compile_definitions(portable_math_test PRIVATE EXPECTED_HASH=${EXPECTED_HASH}ull)
endif()
enable_testing()
add_test(NAME portable_math_test COMMAND portable_math_test)
