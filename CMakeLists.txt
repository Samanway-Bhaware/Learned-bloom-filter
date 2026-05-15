cmake_minimum_required(VERSION 3.20)
project(lbf VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# C++ standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(LBF_STRICT_WARNINGS "Treat warnings as errors (-Werror)" ON)
option(LBF_ENABLE_ONNX     "Enable ONNX Runtime model backend"  OFF)
option(LBF_BUILD_BENCHMARKS "Build Google Benchmark targets"    ON)

# ---------------------------------------------------------------------------
# Compiler warnings
# ---------------------------------------------------------------------------
add_library(lbf_warnings INTERFACE)
if(MSVC)
  target_compile_options(lbf_warnings INTERFACE /W4 $<$<BOOL:${LBF_STRICT_WARNINGS}>:/WX>)
else()
  target_compile_options(lbf_warnings INTERFACE
    -Wall -Wextra -Wpedantic
    $<$<BOOL:${LBF_STRICT_WARNINGS}>:-Werror>)
endif()

# ---------------------------------------------------------------------------
# Interface library (header-only core)
# ---------------------------------------------------------------------------
add_library(lbf INTERFACE)
target_include_directories(lbf INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(lbf INTERFACE lbf_warnings)
target_compile_features(lbf INTERFACE cxx_std_20)

# ---------------------------------------------------------------------------
# Dependencies via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)

# GoogleTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0)
# Prevent overriding the parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# xxHash — header-only via XXH_INLINE_ALL; we only need xxhash.h so we skip
# xxHash's own build system entirely with FetchContent_Populate.
FetchContent_Declare(
  xxhash
  GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
  GIT_TAG        v0.8.2)
FetchContent_GetProperties(xxhash)
if(NOT xxhash_POPULATED)
  FetchContent_Populate(xxhash)
endif()
# SYSTEM suppresses warnings emitted by xxHash headers under /W4 (MSVC) and
# -Wall (GCC/Clang) — particularly C4505 (unreferenced static function).
target_include_directories(lbf SYSTEM INTERFACE ${xxhash_SOURCE_DIR})

# cxxopts — header-only argument parser (used by lbf-train).
# Use FetchContent_Populate so the header can be added as SYSTEM to suppress
# any third-party warnings, mirroring the xxHash pattern above.
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG        v3.1.1)
FetchContent_GetProperties(cxxopts)
if(NOT cxxopts_POPULATED)
  FetchContent_Populate(cxxopts)
endif()

# Google Benchmark
if(LBF_BUILD_BENCHMARKS)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3)
  FetchContent_MakeAvailable(benchmark)
endif()

# ---------------------------------------------------------------------------
# Enable CTest — must come before add_subdirectory so gtest_discover_tests
# registers tests in the parent scope correctly.
# ---------------------------------------------------------------------------
enable_testing()

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)
add_subdirectory(training)
