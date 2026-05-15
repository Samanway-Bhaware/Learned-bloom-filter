#pragma once

/// @file models/logistic_regression.hpp
/// @brief Sparse logistic regression trained on character n-gram hash features.
///
/// Implements @ref MembershipModel with a linear classifier whose features are
/// produced by @ref NGramHasher. Training uses mini-batch SGD with momentum and
/// L2 regularization. Prediction is a single sparse dot product followed by a
/// numerically stable sigmoid.

#include "lbf/features/ngram_hasher.hpp"
#include "lbf/model.hpp"

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <span>
#include <utility>
#include <vector>

namespace lbf {

// ---------------------------------------------------------------------------
// TrainingConfig
// ---------------------------------------------------------------------------

/// @brief Hyper-parameters for the mini-batch SGD training loop.
///
/// All fields have production-ready defaults; only @c epochs_ and
/// @c learning_rate_ typically need tuning.
struct TrainingConfig {
  double learning_rate_ = 0.1;      ///< Initial step size η (gradient scale).
  double l2_regularization_ = 1e-4; ///< L2 penalty coefficient λ (weight decay).
  size_t epochs_ = 10;              ///< Number of full passes over the training set.
  size_t batch_size_ = 256;         ///< Examples per mini-batch gradient update.
  double momentum_ = 0.9;           ///< SGD momentum coefficient β ∈ [0, 1).
  uint64_t random_seed_ = 42;       ///< Seed for epoch-level batch shuffling.
  bool verbose_ = false;            ///< If true, print per-epoch metrics to stderr.
};

// ---------------------------------------------------------------------------
// TrainingMetrics
// ---------------------------------------------------------------------------

/// @brief Diagnostics collected at the end of each training epoch.
struct TrainingMetrics {
  std::vector<double> train_loss_per_epoch_;     ///< BCE loss averaged over the epoch.
  std::vector<double> train_accuracy_per_epoch_; ///< Fraction correctly classified.
  double final_auc_ = 0.0;                       ///< Area under ROC (trapezoid rule).
};

// ---------------------------------------------------------------------------
// LogisticRegressionModel
// ---------------------------------------------------------------------------

/// @brief Learned membership model: sparse logistic regression on n-gram features.
///
/// Given a query key @c x the score is:
/// @f[ s(x) = \sigma\!\left(\sum_{i \in \phi(x)} w_i \cdot c_i + b\right) @f]
/// where @f$\phi(x)@f$ is the sparse n-gram feature vector from @ref NGramHasher,
/// @f$w@f$ is the learned weight vector, @f$b@f$ is the bias, and
/// @f$\sigma@f$ is the numerically stable sigmoid.
///
/// @par Numerical stability
/// Sigmoid uses the branch-free stable form: for @f$z \ge 0@f$,
/// @f$\sigma(z) = 1 / (1 + e^{-z})@f$; for @f$z < 0@f$,
/// @f$\sigma(z) = e^z / (1 + e^z)@f$.  BCE loss is clipped to @c [eps, 1-eps].
///
/// @par Serialization format
/// Binary layout (little-endian integers):
/// @code
///   Offset  Size   Field
///   ------  -----  -----
///      0      4    Magic: 0x4C 0x4C 0x52 0x00 ("LLR\0")
///      4      2    Version: uint16_t = 1
///      6      ?    NGramHasher sub-record (variable, self-described)
///      ?      4    weight_count: uint32_t
///      ?    4*W    weights: W × float32 (little-endian)
///      ?      4    bias: float32
///      ?      4    CRC-32 over all preceding bytes
/// @endcode
class LogisticRegressionModel : public MembershipModel {
public:
  // -------------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------------

  /// @brief Construct a fully-trained model from its components.
  ///
  /// Intended for deserialization and testing. Prefer @c train() in normal use.
  ///
  /// @param hasher   Feature extractor; its @c feature_dim_ must equal
  ///                 @p weights.size().
  /// @param weights  Learned weight vector (one float per feature bucket).
  /// @param bias     Learned bias (intercept) term.
  /// @throws std::invalid_argument  If @p weights.size() != hasher.config().feature_dim_.
  LogisticRegressionModel(NGramHasher hasher, std::vector<float> weights, float bias);

  // -------------------------------------------------------------------------
  // Prediction
  // -------------------------------------------------------------------------

  /// @brief Compute the membership probability of @p key.
  ///
  /// Extracts n-gram features, computes the sparse dot product @f$w \cdot \phi + b@f$,
  /// applies the stable sigmoid, and returns the result.
  ///
  /// @return Score in @c [0, 1].
  [[nodiscard]] double predict(std::span<const std::byte> key) const noexcept override;

  /// @brief Batch prediction; calls @c predict() per key sequentially.
  ///
  /// @pre @p out_scores.size() == @p keys.size().
  void predict_batch(std::span<const std::span<const std::byte>> keys,
                     std::span<double> out_scores) const override;

  // -------------------------------------------------------------------------
  // Serialization
  // -------------------------------------------------------------------------

  /// @brief Serialize to @p os in the lbf binary format.
  /// @throws std::ios_base::failure  On write error (if exceptions enabled).
  void save(std::ostream &os) const override;

  /// @brief Deserialize from @p is. Invoked by @ref MembershipModel::load().
  ///
  /// @throws std::runtime_error      On magic/version mismatch or CRC-32 failure.
  /// @throws std::ios_base::failure  On read error.
  [[nodiscard]] static LogisticRegressionModel load_from(std::istream &is);

  // -------------------------------------------------------------------------
  // Training
  // -------------------------------------------------------------------------

  /// @brief Train a logistic regression model on labelled byte-key data.
  ///
  /// Algorithm:
  ///  1. Build an @ref NGramHasher from @p ngram_config.
  ///  2. Initialise weights to zero; bias to the log-odds of positive rate.
  ///  3. Run @c epochs_ passes of mini-batch SGD with momentum and L2 decay.
  ///  4. Compute final AUC on the training set (for diagnostic purposes only).
  ///
  /// @param keys         Training keys (one span per example).
  /// @param labels       Binary labels parallel to @p keys (0 = negative, 1 = positive).
  /// @param ngram_config Configuration for the n-gram feature hasher.
  /// @param train_config SGD hyper-parameters.
  /// @pre @p keys.size() == @p labels.size() and both are non-empty.
  /// @throws std::invalid_argument  If the precondition is violated.
  /// @return Trained model paired with per-epoch metrics.
  [[nodiscard]] static std::pair<LogisticRegressionModel, TrainingMetrics>
  train(std::span<const std::span<const std::byte>> keys, std::span<const uint8_t> labels,
        NGramConfig ngram_config, TrainingConfig train_config);

  // -------------------------------------------------------------------------
  // Introspection
  // -------------------------------------------------------------------------

  /// @brief Number of scalar weights (equals @c hasher.config().feature_dim_).
  [[nodiscard]] size_t weight_count() const noexcept;

  /// @brief Approximate heap memory consumed by the weight vector, in bytes.
  [[nodiscard]] size_t memory_bytes() const noexcept;

private:
  NGramHasher hasher_;
  std::vector<float> weights_;
  float bias_;
};

} // namespace lbf
