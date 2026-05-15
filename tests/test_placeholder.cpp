#include "lbf/bloom_filter.hpp"
#include "lbf/hashing.hpp"
#include "lbf/learned_bloom_filter.hpp"
#include "lbf/model.hpp"

#include <gtest/gtest.h>

// Phase 1 placeholder: confirms the test infrastructure compiles and runs.
// Real unit tests are added in Phase 2 (hashing + classical BF).

TEST(Phase1, BuildSystemSanity) {
  EXPECT_EQ(1 + 1, 2);
}

// If the four lbf headers compile and all their namespace declarations are
// reachable, this translation unit will compile — which is the assertion.
TEST(Phase1, HeadersCompileClean) {
  SUCCEED();
}
