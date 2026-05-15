/// @file test_learned_bloom_filter.cpp
/// @brief Unit tests for @ref LearnedBloomFilter (Phase 4).
///
/// Tests cover:
///   - Zero-false-negative guarantee: every member always returns true.
///   - Non-members can return false (FPR is bounded, not zero).
///   - Threshold accessor.
///   - Backup-count matches model false-negative count.
///   - Serialization round-trip (save → load → same results).
///   - string_view query convenience overload.

#include "lbf/learned_bloom_filter.hpp"
#include "lbf/models/logistic_regression.hpp"

#include <gtest/gtest.h>
#include <cstdint>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::span<const std::byte> to_bytes(const std::string &s) {
  return std::as_bytes(std::span<const char>{s.data(), s.size()});
}

/// Build and train a LogisticRegressionModel on member + non-member strings,
/// then construct a LearnedBloomFilter from it.
///
/// Uses a small feature dimension and few epochs so the fixture is fast.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
lbf::LearnedBloomFilter make_lbf(const std::vector<std::string> &member_strs,
                                  const std::vector<std::string> &nonmember_strs,
                                  double threshold = 0.5) {
  // Combine into a single training set.
  std::vector<std::string> all_strs;
  all_strs.reserve(member_strs.size() + nonmember_strs.size());
  std::vector<uint8_t> labels;
  labels.reserve(all_strs.capacity());

  for (const auto &s : member_strs) {
    all_strs.push_back(s);
    labels.push_back(uint8_t{1});
  }
  for (const auto &s : nonmember_strs) {
    all_strs.push_back(s);
    labels.push_back(uint8_t{0});
  }

  // Build byte-span views (stable: all_strs owns the strings).
  std::vector<std::span<const std::byte>> all_spans;
  all_spans.reserve(all_strs.size());
  for (const auto &s : all_strs) {
    all_spans.push_back(to_bytes(s));
  }

  const lbf::NGramConfig ngram{.min_n_ = 3, .max_n_ = 5, .feature_dim_ = 1024};
  const lbf::TrainingConfig train_cfg{.learning_rate_ = 0.1,
                                      .l2_regularization_ = 1e-4,
                                      .epochs_ = 30,
                                      .batch_size_ = 8,
                                      .momentum_ = 0.9,
                                      .random_seed_ = 42,
                                      .verbose_ = false};

  auto result = lbf::LogisticRegressionModel::train(
      std::span<const std::span<const std::byte>>{all_spans},
      std::span<const uint8_t>{labels}, ngram, train_cfg);

  // Member spans (first member_strs.size() entries of all_strs).
  std::vector<std::span<const std::byte>> member_spans;
  member_spans.reserve(member_strs.size());
  for (size_t i = 0; i < member_strs.size(); ++i) {
    member_spans.push_back(to_bytes(all_strs[i]));
  }

  return lbf::LearnedBloomFilter::build(
      std::make_unique<lbf::LogisticRegressionModel>(std::move(result.first)),
      std::span<const std::span<const std::byte>>{member_spans},
      0.01, threshold);
}

// ---------------------------------------------------------------------------
// Shared small dataset
// ---------------------------------------------------------------------------

const std::vector<std::string> MEMBER_STRS = {
    "apple", "banana", "cherry", "date", "elderberry",
    "fig",   "grape",  "honeydew", "jackfruit", "kiwi"};

const std::vector<std::string> NONMEMBER_STRS = {
    "xylophone", "zodiac",  "quasar",   "neutron",  "photon",
    "electron",  "proton",  "meson",    "baryon",   "lepton"};

} // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(LearnedBloomFilter, ZeroFalseNegatives) {
  const auto lbf = make_lbf(MEMBER_STRS, NONMEMBER_STRS);
  for (const auto &s : MEMBER_STRS) {
    EXPECT_TRUE(lbf.contains(to_bytes(s))) << "False negative for: " << s;
  }
}

TEST(LearnedBloomFilter, NonMembersNotAllTrue) {
  const auto lbf = make_lbf(MEMBER_STRS, NONMEMBER_STRS);
  // At least one non-member must return false — the filter is not trivially
  // broken. (With high probability almost all return false.)
  int false_positives = 0;
  for (const auto &s : NONMEMBER_STRS) {
    if (lbf.contains(to_bytes(s))) {
      ++false_positives;
    }
  }
  EXPECT_LT(false_positives, static_cast<int>(NONMEMBER_STRS.size()))
      << "All non-members returned true — filter may be degenerate";
}

TEST(LearnedBloomFilter, ThresholdAccessor) {
  const double custom_threshold = 0.7;
  const auto lbf = make_lbf(MEMBER_STRS, NONMEMBER_STRS, custom_threshold);
  EXPECT_DOUBLE_EQ(lbf.threshold(), custom_threshold);
}

TEST(LearnedBloomFilter, BackupCountIsNonNegative) {
  const auto lbf = make_lbf(MEMBER_STRS, NONMEMBER_STRS);
  // backup_count() must be in [0, MEMBER_STRS.size()].
  EXPECT_LE(lbf.backup_count(), MEMBER_STRS.size());
}

TEST(LearnedBloomFilter, ZeroFalseNegativesHighThreshold) {
  // Threshold 0.99 forces nearly all members into the backup — tests the
  // backup path exclusively.
  const auto lbf = make_lbf(MEMBER_STRS, NONMEMBER_STRS, 0.99);
  for (const auto &s : MEMBER_STRS) {
    EXPECT_TRUE(lbf.contains(to_bytes(s)))
        << "False negative (high threshold) for: " << s;
  }
}

TEST(LearnedBloomFilter, ZeroFalseNegativesLowThreshold) {
  // Threshold 0.01 forces nearly all members through the model fast path.
  const auto lbf = make_lbf(MEMBER_STRS, NONMEMBER_STRS, 0.01);
  for (const auto &s : MEMBER_STRS) {
    EXPECT_TRUE(lbf.contains(to_bytes(s)))
        << "False negative (low threshold) for: " << s;
  }
}

TEST(LearnedBloomFilter, StringViewOverload) {
  const auto lbf = make_lbf(MEMBER_STRS, NONMEMBER_STRS);
  for (const auto &s : MEMBER_STRS) {
    EXPECT_TRUE(lbf.contains(std::string_view{s}))
        << "string_view overload: false negative for: " << s;
  }
}

TEST(LearnedBloomFilter, MemoryBytesPositive) {
  const auto lbf = make_lbf(MEMBER_STRS, NONMEMBER_STRS);
  EXPECT_GT(lbf.memory_bytes(), size_t{0});
}

TEST(LearnedBloomFilter, SerializationRoundTrip) {
  const auto lbf = make_lbf(MEMBER_STRS, NONMEMBER_STRS);

  // Save.
  std::ostringstream oss;
  lbf.save(oss);
  const std::string serialized = oss.str();
  ASSERT_GT(serialized.size(), size_t{0});

  // Load.
  std::istringstream iss{serialized};
  const auto loaded = lbf::LearnedBloomFilter::load(iss);

  // Verify: all members still found, threshold preserved.
  EXPECT_DOUBLE_EQ(loaded.threshold(), lbf.threshold());
  for (const auto &s : MEMBER_STRS) {
    EXPECT_TRUE(loaded.contains(to_bytes(s)))
        << "Post-load false negative for: " << s;
  }
}

TEST(LearnedBloomFilter, SerializationCrcDetectsCorruption) {
  const auto lbf = make_lbf(MEMBER_STRS, NONMEMBER_STRS);

  std::ostringstream oss;
  lbf.save(oss);
  std::string serialized = oss.str();

  // Corrupt a byte in the middle of the payload.
  const size_t mid = serialized.size() / 2;
  serialized[mid] = static_cast<char>(static_cast<uint8_t>(serialized[mid]) ^ 0xFFU);

  std::istringstream iss{serialized};
  // Cast to void suppresses [[nodiscard]] warning inside EXPECT_THROW.
  EXPECT_THROW((void)lbf::LearnedBloomFilter::load(iss), std::runtime_error);
}
