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




# CPU Test
add_executable(cpu main.cpp)

target_compile_features(cpu PRIVATE cxx_std_20)

target_compile_options(cpu PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,NVHPC>:-stdpar=multicore>
    # $<$<COMPILE_LANG_AND_ID:CXX,NVHPC>:-mp>
    # $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-stdlib=libstdc++ -Wno-deprecated-builtins>
    $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wno-deprecated-builtins>
)

# target_link_options(cpu PRIVATE
#     # $<$<LINK_LANG_AND_ID:CXX,NVHPC>:-mp>
#     $<$<LINK_LANG_AND_ID:CXX,NVHPC>:-stdpar=multicore>
#     $<$<LINK_LANG_AND_ID:CXX,Clang>:-ltbb>
# )

target_link_libraries(cpu PRIVATE
    pthread
    fmt::fmt
    STDEXEC::stdexec
    mdspan
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(cpu PRIVATE OpenMP::OpenMP_CXX)
endif()

if (NOT CMAKE_CXX_COMPILER_ID MATCHES NVHPC)
    target_link_libraries(cpu PRIVATE TBB::tbb)
    # target_link_libraries(cpu PRIVATE OpenMP::OpenMP_CXX)
endif()




# Single GPU Test
add_executable(gpu main.cpp)

# set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${_lang_gpu})

target_compile_definitions(gpu PRIVATE USE_GPU)

target_compile_features(gpu PRIVATE cxx_std_20)
# target_compile_features(gpu PRIVATE cuda_std_20)

target_compile_options(gpu PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,NVHPC>:-stdpar=gpu>
    # $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-stdlib=libstdc++ -Wno-deprecated-builtins>
    "-Wall"
    "-Wextra"
    "-Wshadow"
)

target_link_options(gpu PRIVATE
    $<$<LINK_LANG_AND_ID:CXX,NVHPC>:-cuda>
)

target_link_libraries(gpu PRIVATE
    pthread
    fmt::fmt
    STDEXEC::stdexec
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(gpu PUBLIC OpenMP::OpenMP_CXX)
endif()




# Multi GPU Test
add_executable(mgpu main.cpp)

# set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${_lang_gpu})

target_compile_definitions(mgpu PRIVATE USE_GPU USE_MULTI_GPU)

target_compile_features(mgpu PRIVATE cxx_std_20)
# target_compile_features(mgpu PRIVATE cuda_std_20)

target_compile_options(mgpu PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,NVHPC>:-stdpar=gpu>
    # $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-stdlib=libstdc++ -Wno-deprecated-builtins>
    "-Wall"
    "-Wextra"
    "-Wshadow"
)

target_link_options(mgpu PRIVATE
    $<$<LINK_LANG_AND_ID:CXX,NVHPC>:-cuda>
)

target_link_libraries(mgpu PRIVATE
    pthread
    fmt::fmt
    STDEXEC::stdexec
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(mgpu PUBLIC OpenMP::OpenMP_CXX)
endif()

