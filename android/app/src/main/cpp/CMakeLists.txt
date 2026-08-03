cmake_minimum_required(VERSION 3.22.1)
project(ledgerlens C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# llama.cpp is expected as a git submodule at <android root>/third_party/llama.cpp.
# See android/README.md — it must be checked out at the tag this code was written against.
set(LLAMA_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/llama.cpp)

if(NOT EXISTS ${LLAMA_CPP_DIR}/CMakeLists.txt)
    message(FATAL_ERROR
        "llama.cpp not found at ${LLAMA_CPP_DIR}.\n"
        "Run (from the android/ directory):\n"
        "  git submodule add https://github.com/ggml-org/llama.cpp third_party/llama.cpp\n"
        "  git -C third_party/llama.cpp checkout <PINNED_TAG>   # see README.md\n")
endif()

# Statically link llama/ggml/mtmd into our single libledgerlens.so.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# The mtmd library lives under tools/mtmd and is only configured when tools are enabled.
# EXCLUDE_FROM_ALL below ensures only the targets we actually link against get built,
# so the various CLI tools are configured but not compiled.
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)

# Keep the ggml build simple and portable for Android/NDK.
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_CPP_DIR} ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)

add_library(ledgerlens SHARED
    ledgerlens_jni.cpp)

target_include_directories(ledgerlens PRIVATE
    ${LLAMA_CPP_DIR}/include        # llama.h
    ${LLAMA_CPP_DIR}/ggml/include   # ggml.h
    ${LLAMA_CPP_DIR}/common         # common helpers (linked per project convention)
    ${LLAMA_CPP_DIR}/tools/mtmd)    # mtmd.h, mtmd-helper.h

target_link_libraries(ledgerlens PRIVATE
    llama
    mtmd
    common
    android
    log)
