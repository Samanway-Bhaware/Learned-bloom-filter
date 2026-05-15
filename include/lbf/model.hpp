#pragma once

/// @file model.hpp
/// @brief Abstract MembershipModel interface and polymorphic factory loader.
///
/// Every learned model in lbf derives from @ref MembershipModel. Callers
/// that only hold a @c std::unique_ptr<MembershipModel> can predict and
/// serialise without knowing the concrete type.

#include <istream>
#include <memory>
#include <ostream>
#include <span>

namespace lbf {

/// @brief Abstract base for all learned membership models.
///
/// Maps a raw-byte key to a probability score in @c [0, 1], where @c 1 means
/// high confidence the key is a set member. Concrete implementations:
///
///  - @ref LogisticRegressionModel — native C++, no runtime dependencies.
///  - @ref OnnxModel — optional; requires @c LBF_ENABLE_ONNX at build time.
///
/// @par Thread safety
/// All @c predict / @c predict_batch overloads are @c const and may be called
/// concurrently from multiple threads. @c save is @c const but callers must
/// externally synchronise on the output stream.
///
/// @par Polymorphic ownership
/// Always heap-allocate and hold via @c std::unique_ptr<MembershipModel>.
/// Copying polymorphic objects is disabled to prevent object slicing.
class MembershipModel {
public:
  virtual ~MembershipModel() = default;

  // Non-copyable (prevents object slicing through base-class copies).
  MembershipModel(const MembershipModel &) = delete;
  MembershipModel &operator=(const MembershipModel &) = delete;

  // -------------------------------------------------------------------------
  // Prediction
  // -------------------------------------------------------------------------

  /// @brief Predict membership probability for a single @p key.
  ///
  /// @param key  Raw byte representation of the query key. Empty spans are valid.
  /// @return     Score in @c [0, 1]. Higher values indicate stronger membership
  ///             evidence. A threshold of @c 0.5 gives the MAP decision.
  [[nodiscard]] virtual double predict(std::span<const std::byte> key) const noexcept = 0;

  /// @brief Predict membership probabilities for a batch of keys.
  ///
  /// The default implementation calls @c predict() in a loop. Concrete backends
  /// may override for SIMD / batched-inference acceleration.
  ///
  /// @pre @p out_scores.size() == @p keys.size().
  /// @param keys        One byte-span view per query key.
  /// @param out_scores  Caller-provided output buffer; overwritten in key order.
  virtual void predict_batch(std::span<const std::span<const std::byte>> keys,
                             std::span<double> out_scores) const;

  // -------------------------------------------------------------------------
  // Serialization
  // -------------------------------------------------------------------------

  /// @brief Serialize this model to @p os in the lbf binary format.
  ///
  /// The stream should be open in binary mode. The byte layout begins with a
  /// 4-byte magic that identifies the concrete subclass, followed by
  /// subclass-specific fields and a CRC-32 integrity trailer.
  ///
  /// @throws std::ios_base::failure  On write error (if exceptions enabled).
  virtual void save(std::ostream &os) const = 0;

  /// @brief Deserialize a model from @p is, dispatching on magic bytes.
  ///
  /// Reads the 4-byte magic prefix, seeks back, then delegates to the matching
  /// subclass @c load_from(). Throws on any unrecognised magic or CRC mismatch.
  ///
  /// Recognised magic values:
  ///  - @c 0x4C'4C'52'00 (@c "LLR\\0") → @ref LogisticRegressionModel
  ///  - @c 0x4C'4F'58'00 (@c "LOX\\0") → @ref OnnxModel (requires LBF_ENABLE_ONNX)
  ///
  /// @throws std::runtime_error      On unknown magic or CRC-32 mismatch.
  /// @throws std::ios_base::failure  On read error.
  [[nodiscard]] static std::unique_ptr<MembershipModel> load(std::istream &is);

protected:
  // Protected move allows derived classes to be returned by value (NRVO/move)
  // without allowing external code to slice-move through the base.
  MembershipModel() = default;
  MembershipModel(MembershipModel &&) = default;
  MembershipModel &operator=(MembershipModel &&) = default;
};

} // namespace lbf
