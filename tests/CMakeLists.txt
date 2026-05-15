include(GoogleTest)

add_executable(lbf_tests
  test_placeholder.cpp
  test_bloom_filter.cpp
  test_model.cpp
  test_learned_bloom_filter.cpp)
target_link_libraries(lbf_tests PRIVATE lbf lbf_model GTest::gtest_main)
target_include_directories(lbf_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)

gtest_discover_tests(lbf_tests)

# std::getenv triggers C4996 ("may be unsafe") on MSVC under /W4.
# Suppress it for the test binary only — the library itself is unaffected.
if(MSVC)
  target_compile_definitions(lbf_tests PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
