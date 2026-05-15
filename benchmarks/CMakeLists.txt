# Benchmark targets.
# Google Benchmark is fetched by the root CMakeLists.txt when
# LBF_BUILD_BENCHMARKS=ON (default ON locally, OFF in CI for speed).

if(NOT LBF_BUILD_BENCHMARKS)
  return()
endif()

# Model-layer micro-benchmarks (NGramHasher, LogisticRegressionModel::predict).
add_executable(bench_predict bench_predict.cpp)
target_link_libraries(bench_predict PRIVATE lbf_model benchmark::benchmark_main)
target_include_directories(bench_predict PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Comprehensive classical BF benchmarks (all datasets).
add_executable(bench_classical_bf bench_classical_bf.cpp)
target_link_libraries(bench_classical_bf PRIVATE lbf benchmark::benchmark_main)
target_include_directories(bench_classical_bf PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/benchmarks)
target_compile_definitions(bench_classical_bf PRIVATE
  "LBF_BENCH_DATASETS_DIR_DEFAULT=\"${CMAKE_SOURCE_DIR}/benchmarks/datasets\""
  "LBF_BENCH_FIXTURES_DIR_DEFAULT=\"${CMAKE_SOURCE_DIR}/benchmarks/fixtures\"")

# Phase 5 — comprehensive LearnedBloomFilter benchmarks (all datasets).
add_executable(bench_learned_bf bench_learned_bf.cpp)
target_link_libraries(bench_learned_bf PRIVATE lbf_model benchmark::benchmark_main)
target_include_directories(bench_learned_bf PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/benchmarks)
target_compile_definitions(bench_learned_bf PRIVATE
  "LBF_BENCH_DATASETS_DIR_DEFAULT=\"${CMAKE_SOURCE_DIR}/benchmarks/datasets\""
  "LBF_BENCH_FIXTURES_DIR_DEFAULT=\"${CMAKE_SOURCE_DIR}/benchmarks/fixtures\"")

# Phase 5 — head-to-head comparison: ClassicalBF vs LearnedBloomFilter.
add_executable(bench_compare bench_compare.cpp)
target_link_libraries(bench_compare PRIVATE lbf_model benchmark::benchmark_main)
target_include_directories(bench_compare PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/benchmarks)
target_compile_definitions(bench_compare PRIVATE
  "LBF_BENCH_DATASETS_DIR_DEFAULT=\"${CMAKE_SOURCE_DIR}/benchmarks/datasets\""
  "LBF_BENCH_FIXTURES_DIR_DEFAULT=\"${CMAKE_SOURCE_DIR}/benchmarks/fixtures\"")
