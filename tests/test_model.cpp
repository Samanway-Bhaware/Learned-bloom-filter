#include "lbf/features/ngram_hasher.hpp"
#include "lbf/model.hpp"
#include "lbf/models/logistic_regression.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a std::string reference to a std::span<const std::byte>.
std::span<const std::byte> str_bytes(const std::string &s) {
  return std::as_bytes(std::span<const char>{s.data(), s.size()});
}

/// Build (key_spans, labels) vectors for a simple separable training set.
/// Positive examples: "pos_0" … "pos_(n-1)".
/// Negative examples: "neg_0" … "neg_(n-1)".
/// Both vectors remain alive for the duration of the calling test.
struct TrainingData {
  std::vector<std::string> strings_;
  std::vector<std::span<const std::byte>> key_spans_;
  std::vector<uint8_t> labels_;
};

TrainingData make_training_data(size_t n_each) {
  TrainingData td;
  td.strings_.reserve(2U * n_each);
  td.labels_.reserve(2U * n_each);

  for (size_t i = 0; i < n_each; ++i) {
    td.strings_.push_back("positive_key_" + std::to_string(i));
    td.labels_.push_back(1U);
  }
  for (size_t i = 0; i < n_each; ++i) {
    td.strings_.push_back("negative_sample_" + std::to_string(i));
    td.labels_.push_back(0U);
  }

  td.key_spans_.reserve(td.strings_.size());
  for (const auto &s : td.strings_) {
    td.key_spans_.push_back(str_bytes(s));
  }
  return td;
}

/// Small NGramConfig useful in many tests (feature_dim=1024, n=[3,5]).
lbf::NGramConfig small_config() {
  return lbf::NGramConfig{.min_n_ = 3, .max_n_ = 5, .feature_dim_ = 1024};
}

} // namespace

// ===========================================================================
// NGramHasher — construction / validation
// ===========================================================================

TEST(NGramHasher, ThrowsOnZeroFeatureDim) {
  lbf::NGramConfig cfg{.min_n_ = 2, .max_n_ = 3, .feature_dim_ = 0};
  EXPECT_THROW(lbf::NGramHasher{cfg}, std::invalid_argument);
}

TEST(NGramHasher, ThrowsOnNonPowerOfTwoFeatureDim) {
  lbf::NGramConfig cfg{.min_n_ = 2, .max_n_ = 3, .feature_dim_ = 1000};
  EXPECT_THROW(lbf::NGramHasher{cfg}, std::invalid_argument);
}

TEST(NGramHasher, ThrowsOnZeroMinN) {
  lbf::NGramConfig cfg{.min_n_ = 0, .max_n_ = 3, .feature_dim_ = 1024};
  EXPECT_THROW(lbf::NGramHasher{cfg}, std::invalid_argument);
}

TEST(NGramHasher, ThrowsOnMinNGreaterThanMaxN) {
  lbf::NGramConfig cfg{.min_n_ = 5, .max_n_ = 2, .feature_dim_ = 1024};
  EXPECT_THROW(lbf::NGramHasher{cfg}, std::invalid_argument);
}

// ===========================================================================
// NGramHasher — feature extraction
// ===========================================================================

TEST(NGramHasher, EmptyKeyProducesNoFeatures) {
  lbf::NGramHasher h{small_config()};
  std::vector<std::pair<uint32_t, float>> out;
  h.hash({}, out);
  EXPECT_TRUE(out.empty());
}

TEST(NGramHasher, ShortKeyBelowMinNProducesNoFeatures) {
  // key is 1 byte; min_n_=3, so no windows are produced.
  lbf::NGramHasher h{small_config()};
  const std::string s = "a";
  std::vector<std::pair<uint32_t, float>> out;
  h.hash(str_bytes(s), out);
  EXPECT_TRUE(out.empty());
}

TEST(NGramHasher, Deterministic) {
  lbf::NGramHasher h{small_config()};
  const std::string s = "hello world";
  std::vector<std::pair<uint32_t, float>> out1;
  std::vector<std::pair<uint32_t, float>> out2;
  h.hash(str_bytes(s), out1);
  h.hash(str_bytes(s), out2);
  ASSERT_EQ(out1.size(), out2.size());
  for (size_t i = 0; i < out1.size(); ++i) {
    EXPECT_EQ(out1[i].first, out2[i].first);
    EXPECT_EQ(out1[i].second, out2[i].second);
  }
}

TEST(NGramHasher, IndicesInRange) {
  lbf::NGramHasher h{small_config()};
  const std::string s = "the quick brown fox jumps over the lazy dog";
  std::vector<std::pair<uint32_t, float>> out;
  h.hash(str_bytes(s), out);
  EXPECT_FALSE(out.empty());
  for (const auto &[idx, cnt] : out) {
    EXPECT_LT(idx, small_config().feature_dim_);
  }
}

TEST(NGramHasher, NoDuplicateIndices) {
  lbf::NGramHasher h{small_config()};
  const std::string s = "the quick brown fox jumps over the lazy dog";
  std::vector<std::pair<uint32_t, float>> out;
  h.hash(str_bytes(s), out);
  // Output must be sorted with no consecutive duplicate indices.
  for (size_t i = 1; i < out.size(); ++i) {
    EXPECT_GT(out[i].first, out[i - 1U].first) << "duplicate index at position " << i;
  }
}

TEST(NGramHasher, CountsArePositive) {
  lbf::NGramHasher h{small_config()};
  const std::string s = "aabbccddee";
  std::vector<std::pair<uint32_t, float>> out;
  h.hash(str_bytes(s), out);
  for (const auto &[idx, cnt] : out) {
    EXPECT_GT(cnt, 0.0F);
  }
}

TEST(NGramHasher, WiderRangeProducesMoreOrEqualFeatures) {
  // [1,5] can only produce >= features than [3,5] for the same key.
  const std::string s = "abcdefghij";
  lbf::NGramConfig cfg_wide{.min_n_ = 1, .max_n_ = 5, .feature_dim_ = 1024};
  lbf::NGramConfig cfg_narrow{.min_n_ = 3, .max_n_ = 5, .feature_dim_ = 1024};
  lbf::NGramHasher h_wide{cfg_wide};
  lbf::NGramHasher h_narrow{cfg_narrow};
  std::vector<std::pair<uint32_t, float>> out_wide;
  std::vector<std::pair<uint32_t, float>> out_narrow;
  h_wide.hash(str_bytes(s), out_wide);
  h_narrow.hash(str_bytes(s), out_narrow);
  // Total window count for wide > narrow, so at least as many unique buckets.
  EXPECT_GE(out_wide.size(), out_narrow.size());
}

// ===========================================================================
// NGramHasher — serialization
// ===========================================================================

TEST(NGramHasher, SerializationRoundTrip) {
  lbf::NGramConfig cfg{.min_n_ = 2, .max_n_ = 6, .feature_dim_ = 2048};
  lbf::NGramHasher original{cfg};

  std::ostringstream oss;
  original.save(oss);
  const std::string blob = oss.str();

  std::istringstream iss{blob};
  const lbf::NGramHasher restored = lbf::NGramHasher::load(iss);

  EXPECT_EQ(restored.config().min_n_, cfg.min_n_);
  EXPECT_EQ(restored.config().max_n_, cfg.max_n_);
  EXPECT_EQ(restored.config().feature_dim_, cfg.feature_dim_);

  // Features must be identical after round-trip.
  const std::string s = "serialization test key";
  std::vector<std::pair<uint32_t, float>> out_orig;
  std::vector<std::pair<uint32_t, float>> out_rest;
  original.hash(str_bytes(s), out_orig);
  restored.hash(str_bytes(s), out_rest);
  ASSERT_EQ(out_orig.size(), out_rest.size());
  for (size_t i = 0; i < out_orig.size(); ++i) {
    EXPECT_EQ(out_orig[i].first, out_rest[i].first);
    EXPECT_FLOAT_EQ(out_orig[i].second, out_rest[i].second);
  }
}

TEST(NGramHasher, LoadThrowsOnBadMagic) {
  // Four bytes that are not "LNG\0".
  const std::string bad = "XXXX"
                          "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                          "\x00\x00\x00\x00";
  std::istringstream iss{bad};
  // NOLINT: lambda discards [[nodiscard]] return intentionally inside EXPECT_THROW.
  EXPECT_THROW([&] { (void)lbf::NGramHasher::load(iss); }(), std::runtime_error);
}

TEST(NGramHasher, LoadThrowsOnCrcMismatch) {
  lbf::NGramHasher original{small_config()};
  std::ostringstream oss;
  original.save(oss);
  std::string blob = oss.str();
  // Corrupt the last byte of the CRC.
  blob.back() = static_cast<char>(static_cast<unsigned char>(blob.back()) ^ 0xFFU);
  std::istringstream iss{blob};
  EXPECT_THROW([&] { (void)lbf::NGramHasher::load(iss); }(), std::runtime_error);
}

// ===========================================================================
// LogisticRegressionModel — construction / prediction
// ===========================================================================

TEST(LogisticRegressionModel, ThrowsOnWeightSizeMismatch) {
  lbf::NGramHasher h{small_config()};
  std::vector<float> wrong_weights(512, 0.0F); // feature_dim is 1024, not 512
  EXPECT_THROW(lbf::LogisticRegressionModel(h, std::move(wrong_weights), 0.0F),
               std::invalid_argument);
}

TEST(LogisticRegressionModel, PredictRangeAllZeroWeights) {
  lbf::NGramHasher h{small_config()};
  const size_t dim = small_config().feature_dim_;
  std::vector<float> weights(dim, 0.0F);
  lbf::LogisticRegressionModel m{h, std::move(weights), 0.0F};
  const std::string s = "any query key";
  const double score = m.predict(str_bytes(s));
  EXPECT_GE(score, 0.0);
  EXPECT_LE(score, 1.0);
  // Zero weights + zero bias → sigmoid(0) = 0.5.
  EXPECT_DOUBLE_EQ(score, 0.5);
}

TEST(LogisticRegressionModel, EmptyKeyPredictsStable) {
  lbf::NGramHasher h{small_config()};
  const size_t dim = small_config().feature_dim_;
  std::vector<float> weights(dim, 0.0F);
  lbf::LogisticRegressionModel m{h, std::move(weights), 0.0F};
  const double score = m.predict({});
  EXPECT_GE(score, 0.0);
  EXPECT_LE(score, 1.0);
  EXPECT_TRUE(std::isfinite(score));
}

TEST(LogisticRegressionModel, NumericalStabilityLargePositiveBias) {
  lbf::NGramHasher h{small_config()};
  const size_t dim = small_config().feature_dim_;
  std::vector<float> weights(dim, 0.0F);
  // Extremely large positive bias → sigmoid should saturate to 1.0, not NaN.
  lbf::LogisticRegressionModel m{h, std::move(weights), 1e30F};
  const std::string s = "stability test";
  const double score = m.predict(str_bytes(s));
  EXPECT_TRUE(std::isfinite(score));
  EXPECT_GE(score, 0.0);
  EXPECT_LE(score, 1.0);
  EXPECT_GT(score, 0.9); // Should be close to 1.
}

TEST(LogisticRegressionModel, NumericalStabilityLargeNegativeBias) {
  lbf::NGramHasher h{small_config()};
  const size_t dim = small_config().feature_dim_;
  std::vector<float> weights(dim, 0.0F);
  // Extremely large negative bias → sigmoid should saturate to 0.0, not NaN.
  lbf::LogisticRegressionModel m{h, std::move(weights), -1e30F};
  const std::string s = "stability test";
  const double score = m.predict(str_bytes(s));
  EXPECT_TRUE(std::isfinite(score));
  EXPECT_GE(score, 0.0);
  EXPECT_LE(score, 1.0);
  EXPECT_LT(score, 0.1); // Should be close to 0.
}

TEST(LogisticRegressionModel, WeightCountMatchesFeatureDim) {
  const size_t dim = small_config().feature_dim_;
  lbf::NGramHasher h{small_config()};
  std::vector<float> weights(dim, 0.0F);
  lbf::LogisticRegressionModel m{h, std::move(weights), 0.0F};
  EXPECT_EQ(m.weight_count(), dim);
}

TEST(LogisticRegressionModel, MemoryBytesAtLeastWeightBytes) {
  const size_t dim = small_config().feature_dim_;
  lbf::NGramHasher h{small_config()};
  std::vector<float> weights(dim, 1.0F);
  lbf::LogisticRegressionModel m{h, std::move(weights), 0.0F};
  EXPECT_GE(m.memory_bytes(), dim * sizeof(float));
}

// ===========================================================================
// LogisticRegressionModel — serialization
// ===========================================================================

TEST(LogisticRegressionModel, SerializationRoundTrip) {
  const size_t dim = small_config().feature_dim_;
  lbf::NGramHasher h{small_config()};
  // Give some non-trivial weights.
  std::vector<float> weights(dim);
  for (size_t i = 0; i < dim; ++i) {
    weights[i] = static_cast<float>(i % 7U) * 0.01F - 0.03F;
  }
  const float bias = 0.42F;
  lbf::LogisticRegressionModel original{h, std::move(weights), bias};

  std::ostringstream oss;
  original.save(oss);
  const std::string blob = oss.str();

  std::istringstream iss{blob};
  const lbf::LogisticRegressionModel restored = lbf::LogisticRegressionModel::load_from(iss);

  // Predictions must be identical after round-trip.
  for (const std::string s : {"hello", "world", "foo bar baz", ""}) {
    const double p_orig = original.predict(str_bytes(s));
    const double p_rest = restored.predict(str_bytes(s));
    EXPECT_DOUBLE_EQ(p_orig, p_rest) << "mismatch for key: " << s;
  }
}

TEST(LogisticRegressionModel, LoadThrowsOnBadMagic) {
  // 34-byte header with wrong magic.
  const std::string bad(34, '\x00');
  std::istringstream iss{bad};
  EXPECT_THROW([&] { (void)lbf::LogisticRegressionModel::load_from(iss); }(), std::runtime_error);
}

// ===========================================================================
// LogisticRegressionModel — training
// ===========================================================================

TEST(LogisticRegressionModel, TrainingThrowsOnEmptyInput) {
  std::vector<std::span<const std::byte>> empty_keys;
  std::vector<uint8_t> empty_labels;
  EXPECT_THROW(
      [&] {
        (void)lbf::LogisticRegressionModel::train(
            std::span<const std::span<const std::byte>>{empty_keys},
            std::span<const uint8_t>{empty_labels}, small_config(), lbf::TrainingConfig{});
      }(),
      std::invalid_argument);
}

TEST(LogisticRegressionModel, TrainingThrowsOnSizeMismatch) {
  auto td = make_training_data(5U);
  // Give one fewer label than keys.
  td.labels_.pop_back();
  EXPECT_THROW(
      [&] {
        (void)lbf::LogisticRegressionModel::train(
            std::span<const std::span<const std::byte>>{td.key_spans_},
            std::span<const uint8_t>{td.labels_}, small_config(), lbf::TrainingConfig{});
      }(),
      std::invalid_argument);
}

TEST(LogisticRegressionModel, TrainingProducesMetrics) {
  auto td = make_training_data(50U);
  lbf::TrainingConfig tcfg{
      .learning_rate_ = 0.1,
      .l2_regularization_ = 1e-4,
      .epochs_ = 3,
      .batch_size_ = 32,
      .momentum_ = 0.9,
      .random_seed_ = 42,
      .verbose_ = false,
  };
  auto [model, metrics] = lbf::LogisticRegressionModel::train(
      std::span<const std::span<const std::byte>>{td.key_spans_},
      std::span<const uint8_t>{td.labels_}, small_config(), tcfg);

  EXPECT_EQ(metrics.train_loss_per_epoch_.size(), tcfg.epochs_);
  EXPECT_EQ(metrics.train_accuracy_per_epoch_.size(), tcfg.epochs_);
  EXPECT_GE(metrics.final_auc_, 0.0);
  EXPECT_LE(metrics.final_auc_, 1.0);
}

TEST(LogisticRegressionModel, TrainingLossDecreases) {
  auto td = make_training_data(80U);
  lbf::TrainingConfig tcfg{
      .learning_rate_ = 0.5,
      .l2_regularization_ = 1e-5,
      .epochs_ = 15,
      .batch_size_ = 32,
      .momentum_ = 0.9,
      .random_seed_ = 7,
      .verbose_ = false,
  };
  auto [model, metrics] = lbf::LogisticRegressionModel::train(
      std::span<const std::span<const std::byte>>{td.key_spans_},
      std::span<const uint8_t>{td.labels_}, small_config(), tcfg);

  // Loss should not have increased from epoch 1 to the final epoch overall.
  EXPECT_GT(metrics.train_loss_per_epoch_.front(), 0.0);
  EXPECT_GT(metrics.train_loss_per_epoch_.back(), 0.0);
  // Final loss should be ≤ initial loss (we're training on separable data).
  EXPECT_LE(metrics.train_loss_per_epoch_.back(), metrics.train_loss_per_epoch_.front());
}

TEST(LogisticRegressionModel, TrainingAUCAboveChance) {
  auto td = make_training_data(100U);
  lbf::TrainingConfig tcfg{
      .learning_rate_ = 0.5,
      .l2_regularization_ = 1e-5,
      .epochs_ = 20,
      .batch_size_ = 32,
      .momentum_ = 0.9,
      .random_seed_ = 0,
      .verbose_ = false,
  };
  auto [model, metrics] = lbf::LogisticRegressionModel::train(
      std::span<const std::span<const std::byte>>{td.key_spans_},
      std::span<const uint8_t>{td.labels_}, small_config(), tcfg);

  // On clearly separable data, AUC should be well above chance (0.5).
  EXPECT_GT(metrics.final_auc_, 0.6)
      << "AUC=" << metrics.final_auc_ << " expected > 0.6 on separable data";
}

TEST(LogisticRegressionModel, TrainingReproducible) {
  auto td = make_training_data(50U);
  lbf::TrainingConfig tcfg{
      .learning_rate_ = 0.1,
      .l2_regularization_ = 1e-4,
      .epochs_ = 5,
      .batch_size_ = 32,
      .momentum_ = 0.9,
      .random_seed_ = 123,
      .verbose_ = false,
  };

  auto [m1, metrics1] = lbf::LogisticRegressionModel::train(
      std::span<const std::span<const std::byte>>{td.key_spans_},
      std::span<const uint8_t>{td.labels_}, small_config(), tcfg);
  auto [m2, metrics2] = lbf::LogisticRegressionModel::train(
      std::span<const std::span<const std::byte>>{td.key_spans_},
      std::span<const uint8_t>{td.labels_}, small_config(), tcfg);

  // Same seed → identical AUC and predictions.
  EXPECT_DOUBLE_EQ(metrics1.final_auc_, metrics2.final_auc_);
  for (const std::string s : {"positive_key_0", "negative_sample_0", "unseen_key"}) {
    EXPECT_DOUBLE_EQ(m1.predict(str_bytes(s)), m2.predict(str_bytes(s)));
  }
}

// ===========================================================================
// MembershipModel — polymorphic factory
// ===========================================================================

TEST(MembershipModel, LoadDispatchesToLogisticRegression) {
  auto td = make_training_data(20U);
  lbf::TrainingConfig tcfg{
      .learning_rate_ = 0.1,
      .l2_regularization_ = 1e-4,
      .epochs_ = 2,
      .batch_size_ = 16,
      .momentum_ = 0.9,
      .random_seed_ = 1,
      .verbose_ = false,
  };
  auto [model, metrics] = lbf::LogisticRegressionModel::train(
      std::span<const std::span<const std::byte>>{td.key_spans_},
      std::span<const uint8_t>{td.labels_}, small_config(), tcfg);

  // Save via the concrete type.
  std::ostringstream oss;
  model.save(oss);
  const std::string blob = oss.str();

  // Load via the polymorphic factory.
  std::istringstream iss{blob};
  const auto loaded = lbf::MembershipModel::load(iss);
  ASSERT_NE(loaded, nullptr);

  // Predictions must match the original.
  for (const std::string s : {"positive_key_0", "negative_sample_0", "unseen"}) {
    EXPECT_DOUBLE_EQ(model.predict(str_bytes(s)), loaded->predict(str_bytes(s)));
  }
}

TEST(MembershipModel, LoadThrowsOnUnknownMagic) {
  const std::string bad(64, '\x00');
  std::istringstream iss{bad};
  EXPECT_THROW([&] { (void)lbf::MembershipModel::load(iss); }(), std::runtime_error);
}
