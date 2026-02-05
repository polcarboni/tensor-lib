cmake_minimum_required(VERSION 3.22)

project(
    "CppAI"
    VERSION 1.0.0
    LANGUAGES CXX)

option(ENABLE_TESTING "Enable unit test building with Catch2" ON)
option(ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" ON)
option(ENABLE_CUDA "Enable NVIDIA CUDA support" ON)

if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_ARCHITECTURE native)
    else()
        messsage(WARNING "CUDA compiler not found. Disabling CUDA support.")
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

# set(EXECUTABLE_NAME_OLD "main_old")
set(EXECUTABLE_NAME "main")

add_subdirectory(src)
add_subdirectory(app)

set(CMAKE_CXX_CLANG_TIDY "")

if(ENABLE_TESTING)
    add_subdirectory(tests)
endif()

