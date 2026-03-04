cmake_minimum_required(VERSION 3.22)

project(
    "CppAI"
    VERSION 1.0.0
    LANGUAGES CXX)

include_directories(${CMAKE_SOURCE_DIR}/inc)

option(ENABLE_TESTING "Enable unit test building with Catch2" ON)
option(ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" ON)
option(ENABLE_CUDA "Enable NVIDIA CUDA support" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

add_compile_options($<$<CONFIG:Debug>:-g3>
                    $<$<CONFIG:Debug>:-O0>
                    $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
                    $<$<CONFIG:Release>:-O3>)

if(ENABLE_CUDA)
    add_compile_definitions(USE_CUDA)  # Find a proper way
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_ARCHITECTURES native)
        find_package(CUDAToolkit REQUIRED)

        add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
                            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-g>
                            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>)
    else()
        message(WARNING "CUDA compiler not found. Disabling CUDA support.")
        set(ENABLE_CUDA OFF)
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------------- DEPENDENCIES --------------------------------

# TESTING (Catch2) 
if(ENABLE_TESTING)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.12.0
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(Catch2)
    enable_testing()
endif()

# -------------------------------- LINTING --------------------------------

if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES "clang-tidy")

    if(CLANG_TIDY_EXE)
        set(CMAKE_CXX_CLANG_TIDY
            ${CLANG_TIDY_EXE}
            --extra-arg=-std=c++17
            --quiet
        )
        message(STATUS "clang-tidy enabled using: " ${CLANG_TIDY_EXE})
    else()
        message(WARNING "clang-tidy requested but executable not found.")
    endif()
endif()

if (ENABLE_CLANG_TIDY AND CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_COMMAND})
endif()

# -------------------------------- SUBDIRECTORIES --------------------------------

add_subdirectory(src)
add_subdirectory(app)

set(CMAKE_CXX_CLANG_TIDY "")

if(ENABLE_TESTING)
    add_subdirectory(tests)
endif()