cmake_minimum_required(VERSION 3.23 FATAL_ERROR)


# set(_lang_cxx CXX)
# set(_lang_gpu CXX)
#
# if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC"))
#     set(_lang_gpu CUDA)
# endif()


# project(StdExecTest LANGUAGES ${_lang_cxx} ${_lang_gpu})
project(StdExecTest)

# enable_language(CUDA)
#
# if(NOT DEFINED CMAKE_CUDA_STANDARD)
#     set(CMAKE_CUDA_STANDARD 20)
#     set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# endif()


# DEPENDENCIES
find_package(OpenMP)

if (NOT CMAKE_CXX_COMPILER_ID MATCHES NVHPC)
    find_package(TBB REQUIRED)
    # find_package(OpenMP REQUIRED)
endif()


include(CMake/CPM.cmake)

CPMAddPackage("gh:fmtlib/fmt#9.1.0")
CPMAddPackage("gh:NVIDIA/stdexec#main")
CPMAddPackage("gh:kokkos/mdspan#stable")
# CPMAddPackage("gh:catchorg/Catch2@3.2.1")


# add_executable(nvc++test nvc++test.cpp)
# target_compile_options(nvc++test PRIVATE "-U__PGI")
# target_compile_features(nvc++test PRIVATE cxx_std_20)




# CPU Test
add_executable(cpu_test cpu_test.cpp)

target_compile_features(cpu_test PRIVATE cxx_std_20)

target_compile_options(cpu_test PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,NVHPC>:-stdpar=multicore>
    # $<$<COMPILE_LANG_AND_ID:CXX,NVHPC>:-mp>
    # $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-stdlib=libstdc++ -Wno-deprecated-builtins>
    $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wno-deprecated-builtins>
)

# target_link_options(cpu_test PRIVATE
#     # $<$<LINK_LANG_AND_ID:CXX,NVHPC>:-mp>
#     $<$<LINK_LANG_AND_ID:CXX,NVHPC>:-stdpar=multicore>
#     $<$<LINK_LANG_AND_ID:CXX,Clang>:-ltbb>
# )

target_link_libraries(cpu_test PRIVATE
    pthread
    fmt::fmt
    STDEXEC::stdexec
    mdspan
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(cpu_test PRIVATE OpenMP::OpenMP_CXX)
endif()

if (NOT CMAKE_CXX_COMPILER_ID MATCHES NVHPC)
    target_link_libraries(cpu_test PRIVATE TBB::tbb)
    # target_link_libraries(cpu_test PRIVATE OpenMP::OpenMP_CXX)
endif()




# GPU Test
add_executable(gpu_test gpu_test.cpp)

# set_source_files_properties(gpu_test.cpp PROPERTIES LANGUAGE ${_lang_gpu})

target_compile_definitions(gpu_test PRIVATE USE_GPU)

target_compile_features(gpu_test PRIVATE cxx_std_20)
# target_compile_features(gpu_test PRIVATE cuda_std_20)

target_compile_options(gpu_test PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,NVHPC>:-stdpar=gpu>
    # $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-stdlib=libstdc++ -Wno-deprecated-builtins>
    "-Wall"
    "-Wextra"
    "-Wshadow"
)

target_link_options(gpu_test PRIVATE
    $<$<LINK_LANG_AND_ID:CXX,NVHPC>:-cuda>
)

target_link_libraries(gpu_test PRIVATE
    pthread
    fmt::fmt
    STDEXEC::stdexec
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(gpu_test PUBLIC OpenMP::OpenMP_CXX)
endif()

