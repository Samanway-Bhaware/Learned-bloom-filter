#include "lbf/bloom_filter.hpp"
#include "lbf/detail/crc32.hpp"
#include "lbf/hashing.hpp"

#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

// ===========================================================================
// CRC-32
// ===========================================================================

TEST(Crc32, EmptyInput) {
  // CRC-32 of empty data is defined as 0x00000000 with the standard XOR-out.
  EXPECT_EQ(lbf::detail::crc32({}), 0x0000'0000U);
}

TEST(Crc32, KnownVector) {
  // Standard test vector: CRC-32("123456789") == 0xCBF43926
  const std::string_view s = "123456789";
  const uint32_t result = lbf::detail::crc32(std::as_bytes(std::span{s.data(), s.size()}));
  EXPECT_EQ(result, 0xCBF4'3926U);
}

TEST(Crc32, Deterministic) {
  const std::string_view s = "hello lbf";
  const auto a = lbf::detail::crc32(std::as_bytes(std::span{s.data(), s.size()}));
  const auto b = lbf::detail::crc32(std::as_bytes(std::span{s.data(), s.size()}));
  EXPECT_EQ(a, b);
}

// ===========================================================================
// Hash128 / Hasher concept
// ===========================================================================

TEST(Hash128, FieldAccess) {
  const lbf::Hash128 h{0xAB'CDU, 0xEF'01U};
  EXPECT_EQ(h.low_, 0xAB'CDU);
  EXPECT_EQ(h.high_, 0xEF'01U);
}

TEST(XXH3Hasher, SatisfiesConcept) {
  static_assert(lbf::Hasher<lbf::XXH3Hasher>, "XXH3Hasher must satisfy the Hasher concept");
  SUCCEED();
}

TEST(XXH3Hasher, Deterministic) {
  const std::string_view key = "determinism check";
  const auto span = std::as_bytes(std::span{key.data(), key.size()});
  const auto h1 = lbf::XXH3Hasher::hash128(span);
  const auto h2 = lbf::XXH3Hasher::hash128(span);
  EXPECT_EQ(h1.low_, h2.low_);
  EXPECT_EQ(h1.high_, h2.high_);
}

TEST(XXH3Hasher, SeedChangesOutput) {
  const std::string_view key = "seed test";
  const auto span = std::as_bytes(std::span{key.data(), key.size()});
  const auto h0 = lbf::XXH3Hasher::hash128(span, 0);
  const auto h1 = lbf::XXH3Hasher::hash128(span, 42);
  EXPECT_TRUE(h0.low_ != h1.low_ || h0.high_ != h1.high_)
      << "Different seeds should produce different hashes";
}

TEST(XXH3Hasher, DifferentInputsDifferentHashes) {
  const auto ha =
      lbf::XXH3Hasher::hash128(std::as_bytes(std::span{std::string_view{"abc"}.data(), 3}));
  const auto hb =
      lbf::XXH3Hasher::hash128(std::as_bytes(std::span{std::string_view{"abd"}.data(), 3}));
  EXPECT_TRUE(ha.low_ != hb.low_ || ha.high_ != hb.high_);
}

TEST(XXH3Hasher, EmptyInputValid) {
  // An empty span must produce a valid (non-crashing) hash.
  const auto h = lbf::XXH3Hasher::hash128({});
  (void)h; // result is implementation-defined but must not be UB
  SUCCEED();
}

// ===========================================================================
// derive_hash
// ===========================================================================

TEST(DeriveHash, LinearCombination) {
  const lbf::Hash128 h{10, 3};
  EXPECT_EQ(lbf::derive_hash(h, 0), 10U);  // 10 + 0*3
  EXPECT_EQ(lbf::derive_hash(h, 1), 13U);  // 10 + 1*3
  EXPECT_EQ(lbf::derive_hash(h, 2), 16U);  // 10 + 2*3
  EXPECT_EQ(lbf::derive_hash(h, 10), 40U); // 10 + 10*3
}

TEST(DeriveHash, WrapsModulo64) {
  // Unsigned overflow is intentional — wraps mod 2^64.
  const lbf::Hash128 h{UINT64_MAX, 1};
  EXPECT_EQ(lbf::derive_hash(h, 1), uint64_t{0}); // UINT64_MAX + 1 == 0
}

// ===========================================================================
// BloomFilterParams
// ===========================================================================

TEST(BloomFilterParams, BitCountKnownValue) {
  // n=1000, p=0.01 → m = ceil(-1000 * ln(0.01) / (ln2)^2) = 9586
  const lbf::BloomFilterParams p{1000, 0.01};
  EXPECT_EQ(p.bit_count(), 9586U);
}

TEST(BloomFilterParams, HashCountKnownValue) {
  // n=1000, p=0.01 → k = round((9586/1000) * ln2) = round(6.644) = 7
  const lbf::BloomFilterParams p{1000, 0.01};
  EXPECT_EQ(p.hash_count(), 7U);
}

TEST(BloomFilterParams, HashCountAtLeastOne) {
  // Extreme FPR near 1 should still give k >= 1.
  const lbf::BloomFilterParams p{1, 0.9999};
  EXPECT_GE(p.hash_count(), 1U);
}

// ===========================================================================
// BloomFilter — construction
// ===========================================================================

TEST(BloomFilter, ConstructAndIntrospect) {
  const lbf::BloomFilter<> bf{{.expected_items_ = 1000, .target_fpr_ = 0.01}};
  EXPECT_EQ(bf.bit_count(), 9586U);
  EXPECT_EQ(bf.hash_count(), 7U);
  EXPECT_EQ(bf.inserted_count(), 0U);
  EXPECT_NEAR(bf.load_factor(), 0.0, 1e-12);
  EXPECT_NEAR(bf.estimated_fpr(), 0.0, 1e-12);
}

TEST(BloomFilter, InvalidExpectedItemsThrows) {
  EXPECT_THROW((lbf::BloomFilter<>{{0, 0.01}}), std::invalid_argument);
}

TEST(BloomFilter, InvalidFprZeroThrows) {
  EXPECT_THROW((lbf::BloomFilter<>{{1000, 0.0}}), std::invalid_argument);
}

TEST(BloomFilter, InvalidFprOneThrows) {
  EXPECT_THROW((lbf::BloomFilter<>{{1000, 1.0}}), std::invalid_argument);
}

TEST(BloomFilter, InvalidFprNegativeThrows) {
  EXPECT_THROW((lbf::BloomFilter<>{{1000, -0.5}}), std::invalid_argument);
}

// ===========================================================================
// BloomFilter — insert / contains correctness
// ===========================================================================

TEST(BloomFilter, InsertAndContainsBytes) {
  lbf::BloomFilter<> bf{{.expected_items_ = 100, .target_fpr_ = 0.01}};
  const std::string key = "hello world";
  const auto span = std::as_bytes(std::span{key.data(), key.size()});

  EXPECT_FALSE(bf.contains(span));
  bf.insert(span);
  EXPECT_TRUE(bf.contains(span));
  EXPECT_EQ(bf.inserted_count(), 1U);
}

TEST(BloomFilter, InsertStringView) {
  lbf::BloomFilter<> bf{{.expected_items_ = 100, .target_fpr_ = 0.01}};
  bf.insert(std::string_view{"alpha"});
  EXPECT_TRUE(bf.contains(std::string_view{"alpha"}));
  EXPECT_FALSE(bf.contains(std::string_view{"beta"}));
}

TEST(BloomFilter, InsertTrivialType) {
  lbf::BloomFilter<> bf{{.expected_items_ = 100, .target_fpr_ = 0.01}};
  const uint64_t key = 0xDEAD'BEEF'CAFE'BABEULL;
  bf.insert(key);
  EXPECT_TRUE(bf.contains(key));
  EXPECT_FALSE(bf.contains(uint64_t{0}));
}

TEST(BloomFilter, InsertedCountTracksAllOverloads) {
  lbf::BloomFilter<> bf{{.expected_items_ = 100, .target_fpr_ = 0.01}};
  const std::string k1 = "one";
  const auto span = std::as_bytes(std::span{k1.data(), k1.size()});
  bf.insert(span);                    // primary
  bf.insert(std::string_view{"two"}); // string_view overload
  bf.insert(uint64_t{3});             // trivial overload
  EXPECT_EQ(bf.inserted_count(), 3U);
}

TEST(BloomFilter, ZeroFalseNegatives) {
  // Exhaustively verify: every key inserted must be found.
  constexpr int n_items = 2000;
  lbf::BloomFilter<> bf{{.expected_items_ = n_items, .target_fpr_ = 0.01}};

  std::vector<std::string> keys;
  keys.reserve(n_items);
  for (int i = 0; i < n_items; ++i) {
    keys.push_back("item_" + std::to_string(i));
  }
  for (const auto &k : keys) {
    bf.insert(std::string_view{k});
  }
  for (const auto &k : keys) {
    EXPECT_TRUE(bf.contains(std::string_view{k})) << "False negative for key: " << k;
  }
  EXPECT_EQ(bf.inserted_count(), static_cast<size_t>(n_items));
}

TEST(BloomFilter, LoadFactorIncreasesWithInsertions) {
  lbf::BloomFilter<> bf{{.expected_items_ = 1000, .target_fpr_ = 0.01}};
  EXPECT_NEAR(bf.load_factor(), 0.0, 1e-12);
  for (int i = 0; i < 500; ++i) {
    bf.insert(static_cast<uint64_t>(i));
  }
  EXPECT_GT(bf.load_factor(), 0.0);
  EXPECT_LT(bf.load_factor(), 1.0);
}

TEST(BloomFilter, EstimatedFprAtCapacity) {
  // At n == expected_items, analytical FPR should be in the right ballpark.
  constexpr size_t n_cap = 1000;
  lbf::BloomFilter<> bf{{.expected_items_ = n_cap, .target_fpr_ = 0.01}};
  for (size_t i = 0; i < n_cap; ++i) {
    bf.insert(static_cast<uint64_t>(i));
  }
  const double fpr = bf.estimated_fpr();
  EXPECT_GT(fpr, 0.001); // not near zero
  EXPECT_LT(fpr, 0.02);  // not catastrophically high (2x target is generous)
}

// ===========================================================================
// BloomFilter — serialization
// ===========================================================================

TEST(BloomFilter, SerializationRoundTrip) {
  lbf::BloomFilter<> original{{.expected_items_ = 500, .target_fpr_ = 0.01}};
  for (int i = 0; i < 200; ++i) {
    const std::string key = "roundtrip_" + std::to_string(i);
    original.insert(std::string_view{key});
  }

  std::stringstream ss;
  ss.exceptions(std::ios::failbit | std::ios::badbit);
  original.save(ss);

  ss.seekg(0);
  const auto loaded = lbf::BloomFilter<>::load(ss);

  EXPECT_EQ(original, loaded);
}

TEST(BloomFilter, RoundTripPreservesMembership) {
  lbf::BloomFilter<> original{{.expected_items_ = 200, .target_fpr_ = 0.05}};
  const std::vector<std::string> keys = {"alpha", "beta", "gamma", "delta"};
  for (const auto &k : keys) {
    original.insert(std::string_view{k});
  }

  std::stringstream ss;
  original.save(ss);
  ss.seekg(0);
  const auto loaded = lbf::BloomFilter<>::load(ss);

  for (const auto &k : keys) {
    EXPECT_TRUE(loaded.contains(std::string_view{k})) << "After load, key '" << k << "' not found";
  }
}

TEST(BloomFilter, LoadBadMagicThrows) {
  lbf::BloomFilter<> bf{{.expected_items_ = 100, .target_fpr_ = 0.01}};
  std::stringstream ss;
  bf.save(ss);

  std::string data = ss.str();
  data[0] = static_cast<char>(0xFF); // corrupt first magic byte
  std::stringstream bad{data};
  // void-cast suppresses -Wunused-result from [[nodiscard]] on load().
  EXPECT_THROW({ (void)lbf::BloomFilter<>::load(bad); }, std::runtime_error);
}

TEST(BloomFilter, LoadBadCrcThrows) {
  lbf::BloomFilter<> bf{{.expected_items_ = 100, .target_fpr_ = 0.01}};
  std::stringstream ss;
  bf.save(ss);

  std::string data = ss.str();
  // Flip a byte well inside the bit array (offset 60, past 54-byte header).
  data[60] = static_cast<char>(data[60] ^ static_cast<char>(0xFF));
  std::stringstream bad{data};
  EXPECT_THROW({ (void)lbf::BloomFilter<>::load(bad); }, std::runtime_error);
}

TEST(BloomFilter, EqualityOperator) {
  const lbf::BloomFilterParams params{200, 0.02};
  lbf::BloomFilter<> a{params};
  lbf::BloomFilter<> b{params};
  EXPECT_EQ(a, b);

  a.insert(std::string_view{"x"});
  EXPECT_NE(a, b);

  b.insert(std::string_view{"x"});
  EXPECT_EQ(a, b);
}

// ===========================================================================
// Statistical FPR test (SLOW) — gated by LBF_SLOW_TESTS=1 env var
//
// Inserts 1 000 000 integer keys, then queries 100 000 held-out negatives.
// Verifies that the Wilson 99.9%-confidence upper bound on the observed FPR
// does not exceed 1.5 × target_fpr.  The slack covers:
//   (a) rounding of m and k can push the actual FPR slightly above target, and
//   (b) statistical sampling variance in the negative queries.
// ===========================================================================

TEST(BloomFilter, FprWithinWilsonBoundsAtScale) {
  if (std::getenv("LBF_SLOW_TESTS") == nullptr) {
    GTEST_SKIP() << "Skipped: set LBF_SLOW_TESTS=1 to run statistical FPR test";
  }

  constexpr size_t n_insert = 1'000'000;
  constexpr size_t n_queries = 100'000;
  constexpr double target_fpr = 0.01;
  // z = 3.291 gives a one-sided 99.9%-confidence upper bound.
  constexpr double z_score = 3.291;

  lbf::BloomFilter<> bf{{.expected_items_ = n_insert, .target_fpr_ = target_fpr}};
  for (size_t i = 0; i < n_insert; ++i) {
    bf.insert(static_cast<uint64_t>(i));
  }
  ASSERT_EQ(bf.inserted_count(), n_insert);

  // Query n_insert … n_insert+n_queries-1 (guaranteed non-inserted).
  size_t false_positives = 0;
  for (size_t i = n_insert; i < n_insert + n_queries; ++i) {
    if (bf.contains(static_cast<uint64_t>(i))) {
      ++false_positives;
    }
  }

  const double p_hat = static_cast<double>(false_positives) / n_queries;
  const auto n_q = static_cast<double>(n_queries);
  const double z_sq = z_score * z_score;

  // Wilson score upper confidence bound.
  const double center = (p_hat + z_sq / (2.0 * n_q)) / (1.0 + z_sq / n_q);
  const double variance_term = p_hat * (1.0 - p_hat) / n_q + z_sq / (4.0 * n_q * n_q);
  const double margin = (z_score / (1.0 + z_sq / n_q)) * std::sqrt(variance_term);
  const double upper_bound = center + margin;

  EXPECT_LE(upper_bound, target_fpr * 1.5)
      << "Observed FPR: " << p_hat << " (" << false_positives << "/" << n_queries
      << "), Wilson 99.9% upper bound: " << upper_bound << ", limit: " << target_fpr * 1.5;
}
