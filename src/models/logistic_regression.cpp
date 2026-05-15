#include "lbf/models/logistic_regression.hpp"

#include "lbf/detail/crc32.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace lbf {

// ---------------------------------------------------------------------------
// Serialization constants
// ---------------------------------------------------------------------------

// Magic bytes "LLR\0".
constexpr std::array<uint8_t, 4> LR_MAGIC = {0x4CU, 0x4CU, 0x52U, 0x00U};
constexpr uint16_t LR_VERSION = 1U;

// Fixed header: 4 magic + 2 version + 8 min_n + 8 max_n + 8 feature_dim + 4 weight_count = 34.
constexpr size_t LR_HEADER_BYTES = 34;

// ---------------------------------------------------------------------------
// Internal helpers (anonymous namespace)
// ---------------------------------------------------------------------------

namespace {

/// Numerically stable sigmoid.
/// For z ≥ 0: σ(z) = 1 / (1 + exp(-z)) — exp(-z) → 0, result → 1.
/// For z < 0: σ(z) = exp(z) / (1 + exp(z)) — avoids exp overflow.
[[nodiscard]] double stable_sigmoid(double z) noexcept {
  if (z >= 0.0) {
    return 1.0 / (1.0 + std::exp(-z));
  }
  const double e = std::exp(z);
  return e / (1.0 + e);
}

/// BCE loss, clipped to avoid log(0).
[[nodiscard]] double bce_loss(double p, double y) noexcept {
  constexpr double eps = 1e-7;
  const double p_clipped = std::clamp(p, eps, 1.0 - eps);
  return -y * std::log(p_clipped) - (1.0 - y) * std::log(1.0 - p_clipped);
}

/// Mann-Whitney U rank-based AUC (exact, handles ties by average rank).
[[nodiscard]] double compute_auc(std::span<const double> scores, std::span<const uint8_t> labels) {
  const size_t n = scores.size();
  if (n == 0) {
    return 0.5;
  }

  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), size_t{0});
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return scores[a] < scores[b]; });

  size_t total_pos = 0;
  double rank_sum_pos = 0.0;

  size_t i = 0;
  while (i < n) {
    // Find the extent of the current tie group.
    size_t j = i;
    while (j < n && scores[order[j]] == scores[order[i]]) {
      ++j;
    }
    // Average rank for this group (1-based).
    const double rank = (static_cast<double>(i) + static_cast<double>(j) + 1.0) / 2.0;
    for (size_t k = i; k < j; ++k) {
      if (labels[order[k]] != 0U) {
        ++total_pos;
        rank_sum_pos += rank;
      }
    }
    i = j;
  }

  const size_t total_neg = n - total_pos;
  if (total_pos == 0 || total_neg == 0) {
    return 0.5;
  }

  const auto np = static_cast<double>(total_pos);
  const auto nn = static_cast<double>(total_neg);
  return (rank_sum_pos - np * (np + 1.0) / 2.0) / (np * nn);
}

} // namespace

// ---------------------------------------------------------------------------
// LogisticRegressionModel — construction
// ---------------------------------------------------------------------------

LogisticRegressionModel::LogisticRegressionModel(NGramHasher hasher, std::vector<float> weights,
                                                 float bias)
    : hasher_(std::move(hasher)), weights_(std::move(weights)), bias_(bias) {
  if (weights_.size() != hasher_.config().feature_dim_) {
    throw std::invalid_argument(
        "LogisticRegressionModel: weights.size() must equal hasher.config().feature_dim_");
  }
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

double LogisticRegressionModel::predict(std::span<const std::byte> key) const noexcept {
  try {
    // thread_local avoids heap allocation per call in predict loops.
    thread_local std::vector<std::pair<uint32_t, float>> features;
    hasher_.hash(key, features);

    double logit = static_cast<double>(bias_);
    for (const auto &[idx, cnt] : features) {
      logit += static_cast<double>(weights_[idx]) * static_cast<double>(cnt);
    }
    return stable_sigmoid(logit);
  } catch (...) {
    return 0.5; // fallback on allocation failure — neutral score
  }
}

void LogisticRegressionModel::predict_batch(std::span<const std::span<const std::byte>> keys,
                                            std::span<double> out_scores) const {
  for (size_t i = 0; i < keys.size(); ++i) {
    out_scores[i] = predict(keys[i]);
  }
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

size_t LogisticRegressionModel::weight_count() const noexcept {
  return weights_.size();
}

size_t LogisticRegressionModel::memory_bytes() const noexcept {
  return weights_.size() * sizeof(float);
}

// ---------------------------------------------------------------------------
// Serialization — save
// ---------------------------------------------------------------------------

void LogisticRegressionModel::save(std::ostream &os) const {
  const auto w_count = static_cast<uint32_t>(weights_.size());

  std::vector<uint8_t> buf;
  buf.reserve(LR_HEADER_BYTES + static_cast<size_t>(w_count) * sizeof(float) + sizeof(float));

  const auto push_u16 = [&](uint16_t v) {
    buf.push_back(static_cast<uint8_t>(v));
    buf.push_back(static_cast<uint8_t>(v >> 8U));
  };
  const auto push_u32 = [&](uint32_t v) {
    for (unsigned i = 0; i < 4U; ++i) {
      buf.push_back(static_cast<uint8_t>(v >> (8U * i)));
    }
  };
  const auto push_u64 = [&](uint64_t v) {
    for (unsigned i = 0; i < 8U; ++i) {
      buf.push_back(static_cast<uint8_t>(v >> (8U * i)));
    }
  };
  const auto push_f32 = [&](float v) { push_u32(std::bit_cast<uint32_t>(v)); };

  // Header.
  for (const auto byte : LR_MAGIC) {
    buf.push_back(byte);
  }
  push_u16(LR_VERSION);
  push_u64(static_cast<uint64_t>(hasher_.config().min_n_));
  push_u64(static_cast<uint64_t>(hasher_.config().max_n_));
  push_u64(static_cast<uint64_t>(hasher_.config().feature_dim_));
  push_u32(w_count);

  // Weights and bias.
  for (const float w : weights_) {
    push_f32(w);
  }
  push_f32(bias_);

  // CRC-32 trailer.
  const uint32_t checksum = detail::crc32(std::as_bytes(std::span{buf}));
  push_u32(checksum);

  os.write(reinterpret_cast<const char *>(buf.data()), static_cast<std::streamsize>(buf.size()));
}

// ---------------------------------------------------------------------------
// Serialization — load
// ---------------------------------------------------------------------------

LogisticRegressionModel LogisticRegressionModel::load_from(std::istream &is) {
  // Read fixed header (34 bytes).
  std::vector<uint8_t> buf(LR_HEADER_BYTES);
  if (!is.read(reinterpret_cast<char *>(buf.data()),
               static_cast<std::streamsize>(LR_HEADER_BYTES))) {
    throw std::runtime_error("LogisticRegressionModel::load_from: truncated header");
  }

  // Validate magic.
  if (buf[0] != LR_MAGIC[0] || buf[1] != LR_MAGIC[1] || buf[2] != LR_MAGIC[2] ||
      buf[3] != LR_MAGIC[3]) {
    throw std::runtime_error("LogisticRegressionModel::load_from: invalid magic bytes");
  }

  // Validate version.
  const auto version = static_cast<uint16_t>(buf[4]) | (static_cast<uint16_t>(buf[5]) << 8U);
  if (version != 1U) {
    throw std::runtime_error("LogisticRegressionModel::load_from: unsupported version " +
                             std::to_string(version));
  }

  // Little-endian uint64 reader from buf.
  const auto read_u64 = [&](size_t offset) -> uint64_t {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
      v |= static_cast<uint64_t>(buf[offset + static_cast<size_t>(i)])
           << (8U * static_cast<unsigned>(i));
    }
    return v;
  };

  NGramConfig cfg{};
  cfg.min_n_ = static_cast<size_t>(read_u64(6));
  cfg.max_n_ = static_cast<size_t>(read_u64(14));
  cfg.feature_dim_ = static_cast<size_t>(read_u64(22));

  // weight_count is the last 4 bytes of the fixed header.
  const uint32_t w_count = static_cast<uint32_t>(buf[30]) | (static_cast<uint32_t>(buf[31]) << 8U) |
                           (static_cast<uint32_t>(buf[32]) << 16U) |
                           (static_cast<uint32_t>(buf[33]) << 24U);

  // Read weights + bias + CRC into the growing buffer.
  const size_t weights_bytes = static_cast<size_t>(w_count) * sizeof(float);
  const size_t bias_bytes = sizeof(float);
  const size_t crc_bytes = sizeof(uint32_t);
  const size_t tail_bytes = weights_bytes + bias_bytes + crc_bytes;

  buf.resize(LR_HEADER_BYTES + tail_bytes);
  if (!is.read(reinterpret_cast<char *>(buf.data() + LR_HEADER_BYTES),
               static_cast<std::streamsize>(tail_bytes))) {
    throw std::runtime_error("LogisticRegressionModel::load_from: truncated payload");
  }

  // Validate CRC (covers everything except the last 4 bytes).
  const size_t payload_end = LR_HEADER_BYTES + weights_bytes + bias_bytes;
  const uint32_t computed = detail::crc32(std::as_bytes(std::span{buf}.subspan(0, payload_end)));
  const uint32_t stored = static_cast<uint32_t>(buf[payload_end]) |
                          (static_cast<uint32_t>(buf[payload_end + 1]) << 8U) |
                          (static_cast<uint32_t>(buf[payload_end + 2]) << 16U) |
                          (static_cast<uint32_t>(buf[payload_end + 3]) << 24U);
  if (computed != stored) {
    throw std::runtime_error(
        "LogisticRegressionModel::load_from: CRC-32 mismatch — data may be corrupted");
  }

  // Deserialise weights.
  std::vector<float> weights(w_count);
  for (uint32_t wi = 0; wi < w_count; ++wi) {
    const size_t off = LR_HEADER_BYTES + static_cast<size_t>(wi) * sizeof(float);
    const uint32_t bits =
        static_cast<uint32_t>(buf[off]) | (static_cast<uint32_t>(buf[off + 1]) << 8U) |
        (static_cast<uint32_t>(buf[off + 2]) << 16U) | (static_cast<uint32_t>(buf[off + 3]) << 24U);
    weights[wi] = std::bit_cast<float>(bits);
  }

  // Deserialise bias.
  const size_t bias_off = LR_HEADER_BYTES + weights_bytes;
  const uint32_t bias_bits = static_cast<uint32_t>(buf[bias_off]) |
                             (static_cast<uint32_t>(buf[bias_off + 1]) << 8U) |
                             (static_cast<uint32_t>(buf[bias_off + 2]) << 16U) |
                             (static_cast<uint32_t>(buf[bias_off + 3]) << 24U);
  const float bias = std::bit_cast<float>(bias_bits);

  return LogisticRegressionModel{NGramHasher{cfg}, std::move(weights), bias};
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

std::pair<LogisticRegressionModel, TrainingMetrics>
LogisticRegressionModel::train(std::span<const std::span<const std::byte>> keys,
                               std::span<const uint8_t> labels, NGramConfig ngram_config,
                               TrainingConfig train_config) {
  if (keys.empty() || keys.size() != labels.size()) {
    throw std::invalid_argument(
        "LogisticRegressionModel::train: keys and labels must be non-empty and equal size");
  }

  NGramHasher hasher{ngram_config};
  const size_t feature_dim = ngram_config.feature_dim_;
  const size_t n_examples = keys.size();

  // Initialise weights to zero.
  std::vector<float> weights(feature_dim, 0.0F);

  // Initialise bias to log-odds of the positive label frequency.
  const auto n_pos = static_cast<double>(std::count(labels.begin(), labels.end(), uint8_t{1}));
  const double pos_rate = std::clamp(n_pos / static_cast<double>(n_examples), 1e-6, 1.0 - 1e-6);
  auto bias = static_cast<float>(std::log(pos_rate / (1.0 - pos_rate)));

  // Momentum velocity buffers.
  std::vector<float> vel_w(feature_dim, 0.0F);
  float vel_b = 0.0F;

  // Dense gradient accumulator (reused across batches; reset after each).
  std::vector<float> grad_w(feature_dim, 0.0F);

  // Pre-compute scalar hyper-params as float to avoid repeated casting.
  const auto lr_f = static_cast<float>(train_config.learning_rate_);
  const auto mom_f = static_cast<float>(train_config.momentum_);
  const auto l2_f = static_cast<float>(train_config.l2_regularization_);

  // Shuffled index order; seeded once, re-shuffled each epoch.
  std::vector<size_t> order(n_examples);
  std::iota(order.begin(), order.end(), size_t{0});
  std::mt19937_64 rng(train_config.random_seed_);

  // Reusable sparse feature buffer.
  std::vector<std::pair<uint32_t, float>> features;

  TrainingMetrics metrics;
  metrics.train_loss_per_epoch_.reserve(train_config.epochs_);
  metrics.train_accuracy_per_epoch_.reserve(train_config.epochs_);

  for (size_t epoch = 0; epoch < train_config.epochs_; ++epoch) {
    std::shuffle(order.begin(), order.end(), rng);

    double epoch_loss = 0.0;
    size_t epoch_correct = 0;
    size_t batch_start = 0;

    while (batch_start < n_examples) {
      const size_t batch_end = std::min(batch_start + train_config.batch_size_, n_examples);
      const size_t actual_batch = batch_end - batch_start;

      float grad_b = 0.0F;

      // Accumulate sparse data gradients into the dense grad_w buffer.
      for (size_t bi = batch_start; bi < batch_end; ++bi) {
        const size_t ex = order[bi];
        hasher.hash(keys[ex], features);

        double logit = static_cast<double>(bias);
        for (const auto &[fi, cnt] : features) {
          logit += static_cast<double>(weights[fi]) * static_cast<double>(cnt);
        }

        const double prob = stable_sigmoid(logit);
        const double y = static_cast<double>(labels[ex]);
        const auto residual = static_cast<float>(prob - y);

        epoch_loss += bce_loss(prob, y);
        if ((prob >= 0.5) == (labels[ex] != 0U)) {
          ++epoch_correct;
        }

        for (const auto &[fi, cnt] : features) {
          grad_w[fi] += residual * cnt;
        }
        grad_b += residual;
      }

      // Dense momentum + L2 update for all weights.
      const auto batch_scale = 1.0F / static_cast<float>(actual_batch);
      for (size_t i = 0; i < feature_dim; ++i) {
        // Full gradient = data gradient / batch + L2 penalty.
        const float full_grad = grad_w[i] * batch_scale + l2_f * weights[i];
        vel_w[i] = mom_f * vel_w[i] - lr_f * full_grad;
        weights[i] += vel_w[i];
        grad_w[i] = 0.0F; // reset for next batch
      }

      // Bias update (no L2 penalty on bias — standard practice).
      vel_b = mom_f * vel_b - lr_f * (grad_b * batch_scale);
      bias += vel_b;

      batch_start = batch_end;
    }

    const double epoch_loss_avg = epoch_loss / static_cast<double>(n_examples);
    const double epoch_acc = static_cast<double>(epoch_correct) / static_cast<double>(n_examples);
    metrics.train_loss_per_epoch_.push_back(epoch_loss_avg);
    metrics.train_accuracy_per_epoch_.push_back(epoch_acc);

    if (train_config.verbose_) {
      std::cerr << "[lbf train] epoch " << (epoch + 1) << "/" << train_config.epochs_
                << "  loss=" << epoch_loss_avg << "  acc=" << epoch_acc << '\n';
    }
  }

  // Compute final training-set AUC.
  std::vector<double> train_scores(n_examples);
  for (size_t i = 0; i < n_examples; ++i) {
    hasher.hash(keys[i], features);
    double logit = static_cast<double>(bias);
    for (const auto &[fi, cnt] : features) {
      logit += static_cast<double>(weights[fi]) * static_cast<double>(cnt);
    }
    train_scores[i] = stable_sigmoid(logit);
  }
  metrics.final_auc_ = compute_auc(train_scores, labels);

  return {LogisticRegressionModel{std::move(hasher), std::move(weights), bias}, metrics};
}

} // namespace lbf
