cmake_minimum_required(VERSION 3.23 FATAL_ERROR)


project(StdExecTest LANGUAGES CXX CUDA)
# project(StdExecTest)



# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------
find_package(OpenMP)

if (NOT CMAKE_CXX_COMPILER_ID MATCHES NVHPC)
    # find_package(TBB REQUIRED)
    # find_package(OpenMP REQUIRED)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/CMake/CPM.cmake)

CPMAddPackage("gh:fmtlib/fmt#9.1.0")
CPMAddPackage("gh:catchorg/Catch2@2.13.6")
# CPMAddPackage("gh:NVIDIA/stdexec#main")
# CPMAddPackage("gh:catchorg/Catch2@3.2.1")
# CPMAddPackage("gh:kokkos/mdspan#stable")

add_subdirectory(stdexec)

set(STDEXEC_BUILD_EXAMPLES OFF)
set(STDEXEC_BUILD_TESTS OFF)


# -----------------------------------------------------------------------------
if (NOT CMAKE_CXX_COMPILER_ID MATCHES NVHPC)
    file(GLOB_RECURSE cuda_sources *.cu*)
    file(GLOB_RECURSE gpu_sources *_gpu.*)
    set_source_files_properties(${cuda_sources} PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(${gpu_sources} PROPERTIES LANGUAGE CUDA)  
endif()


# -----------------------------------------------------------------------------
# Default CPU Compilation Configuration
# -----------------------------------------------------------------------------
add_library(default_cpu_config INTERFACE)

# target_include_directories( default_cpu_config INTERFACE stdexec/include)

target_compile_features(default_cpu_config INTERFACE cxx_std_20)

target_compile_options(default_cpu_config INTERFACE
    "-Wall"
    "-Wextra"
    "-Wshadow"
    $<$<CXX_COMPILER_ID:GNU>:
        -Wpedantic
        -Wconversion
        -Werror
    >
    # $<$<CXX_COMPILER_ID:NVHPC>:-mp>
    $<$<CXX_COMPILER_ID:NVHPC>:-stdpar=multicore>
    $<$<CXX_COMPILER_ID:NVHPC>:--diag_suppress177,550,111,497,554>
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
        -ferror-limit=0
        -fmacro-backtrace-limit=0
        -ftemplate-backtrace-limit=0
        -stdlib=libstdc++
        -Wno-deprecated-builtins
    >
)

target_link_options(default_cpu_config INTERFACE
    # $<$<LINK_LANG_AND_ID:CXX,NVHPC>:-mp>
    $<$<CXX_COMPILER_ID:GNU>:
        -fsanitize=address
        -fsanitize=undefined
        # -fsanitize=memory
    >
)

target_link_libraries(default_cpu_config INTERFACE
    pthread
    fmt::fmt
    STDEXEC::stdexec
)

# if(OpenMP_CXX_FOUND)
#     target_link_libraries(default_cpu_config INTERFACE OpenMP::OpenMP_CXX)
# endif()



# -----------------------------------------------------------------------------
# Default GPU Compilation Configuration
# -----------------------------------------------------------------------------
add_library(default_gpu_config INTERFACE)

# target_include_directories( default_cpu_config INTERFACE
#     stdexec/include
#    # $<$<COMPILE_LANG_AND_ID:CXX,Clang>:${CUDAToolkit_INCLUDE_DIRS}>
# )

target_compile_features(default_gpu_config INTERFACE cxx_std_20)
# target_compile_features(default_gpu_config INTERFACE cuda_std_20)

target_compile_definitions(default_gpu_config INTERFACE USE_GPU)

target_compile_options(default_gpu_config INTERFACE
    "-Wall"
    "-Wextra"
    "-Wshadow"
    $<$<CXX_COMPILER_ID:GNU>:
        -Wpedantic
        -Wconversion
        -Werror
    >
    # $<$<AND:$<CXX_COMPILER_ID:NVHPC>,$<COMPILE_LANGUAGE:CXX>>:-stdpar -gpu=nomanaged -gpu=cc${CMAKE_CUDA_ARCHITECTURES}>
    # $<$<CXX_COMPILER_ID:Clang>:-stdlib=libstdc++ -Wno-deprecated-builtins>
    $<$<CXX_COMPILER_ID:NVHPC>:
        -stdpar=gpu
        --diag_suppress177,550,111,497,554
    >
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
        -ferror-limit=0
        -fmacro-backtrace-limit=0
        -ftemplate-backtrace-limit=0
        -stdlib=libstdc++
        -Wno-deprecated-builtins
    >
    $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:
        -Wno-unknown-cuda-version
        -Xclang=-fcuda-allow-variadic-functions
        -D_GLIBCXX_USE_TBB_PAR_BACKEND=0
        # -include stdexec/__detail/__force_include.hpp>  
    >
)


target_link_options(default_gpu_config INTERFACE
    $<$<LINK_LANG_AND_ID:CXX,NVHPC>:-cuda>
    # $<$<AND:$<CXX_COMPILER_ID:NVHPC>,$<COMPILE_LANGUAGE:CXX>>:-stdpar -gpu=nomanaged -gpu=cc${CMAKE_CUDA_ARCHITECTURES}>
)

target_link_libraries(default_gpu_config INTERFACE
    pthread
    fmt::fmt
    STDEXEC::stdexec
)



# -----------------------------------------------------------------------------
# Hello World
# -----------------------------------------------------------------------------
add_executable(hello_cpu hello_cpu.cpp)
target_link_libraries(hello_cpu PRIVATE default_cpu_config)


add_executable(hello_gpu hello_gpu.cpp)
target_link_libraries(hello_gpu PRIVATE default_gpu_config)



# -----------------------------------------------------------------------------
# Demos
# -----------------------------------------------------------------------------
add_executable(algo_cpu demo_algorithms.cpp)
target_link_libraries(algo_cpu PRIVATE default_cpu_config)


add_executable(algo_sync_cpu demo_algorithms_sync.cpp)
target_link_libraries(algo_sync_cpu PRIVATE default_cpu_config)


add_executable(matrix_sweep_cpu demo_matrix_sweep.cpp)
target_link_libraries(matrix_sweep_cpu PRIVATE default_cpu_config)


add_executable(algo_gpu demo_algorithms_gpu.cpp)
target_link_libraries(algo_gpu PRIVATE default_gpu_config)


add_executable(ifthen if_then_else.cpp)
target_link_libraries(ifthen PRIVATE default_cpu_config)



# -----------------------------------------------------------------------------
# Playground
# -----------------------------------------------------------------------------
add_executable(play_cpu playground.cpp)
target_link_libraries(play_cpu PRIVATE default_cpu_config)


add_executable(play_gpu playground_gpu.cpp)
target_link_libraries(play_gpu PRIVATE default_gpu_config)



# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
enable_testing()


add_executable(test_cpu test_algorithms.cpp)
target_link_libraries(test_cpu PRIVATE default_cpu_config Catch2::Catch2)
add_test(test test_cpu COMMAND $<TARGET_FILE:test_cpu>)


add_executable(test_gpu test_algorithms_gpu.cpp)
target_link_libraries(test_gpu PRIVATE default_gpu_config Catch2::Catch2)
add_test(test test_gpu COMMAND $<TARGET_FILE:test_gpu>)

add_test(test test-gpu COMMAND $<TARGET_FILE:test_cpu>)

