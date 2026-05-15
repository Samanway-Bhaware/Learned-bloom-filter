#pragma once

/// @file features/ngram_hasher.hpp
/// @brief Sparse character n-gram feature hasher backed by xxHash3.
///
/// Extracts all byte n-grams of configurable lengths from a key, hashes each
/// n-gram with XXH3 (seeded by n-gram length to decorrelate across lengths),
/// and maps the result into a power-of-2 feature space via bitwise AND.
/// Output is a sparse count vector — the natural input to a linear classifier.

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <span>
#include <utility>
#include <vector>

namespace lbf {

// ---------------------------------------------------------------------------
// NGramConfig
// ---------------------------------------------------------------------------

/// @brief Configuration for the character n-gram feature extractor.
///
/// Both @c min_n_ and @c max_n_ are inclusive. All n in
/// @c [min_n_, max_n_] contribute features independently.
///
/// @c feature_dim_ controls the hash table width and must be a power of 2
/// so that index computation reduces to a bitwise AND.
struct NGramConfig {
  size_t min_n_ = 3;           ///< Minimum n-gram length (inclusive). Must be ≥ 1.
  size_t max_n_ = 5;           ///< Maximum n-gram length (inclusive). Must be ≥ min_n_.
  size_t feature_dim_ = 65536; ///< Feature space size; must be a power of 2.
};

// ---------------------------------------------------------------------------
// NGramHasher
// ---------------------------------------------------------------------------

/// @brief Sparse character n-gram feature extractor.
///
/// For each query key the hasher:
///  1. Slides a window of width @c n over the raw bytes for all
///     @c n in [@c min_n_, @c max_n_].
///  2. Hashes each n-gram window with @c XXH3_64bits_withSeed, seeding
///     with @c n to decorrelate features across different gram lengths.
///  3. Reduces the hash to a feature index via @c hash & (feature_dim_ - 1).
///  4. Tallies hit counts per index; merges duplicates additively.
///
/// Output is a @c std::vector of @c (feature_index, count) pairs in ascending
/// index order with no duplicate indices.
///
/// @par Serialization format
/// Binary layout (little-endian integers):
/// @code
///   Offset  Size  Field
///   ------  ----  -----
///      0      4   Magic: 0x4C 0x4E 0x47 0x00 ("LNG\0")
///      4      2   Version: uint16_t = 1
///      6      8   min_n:   uint64_t
///     14      8   max_n:   uint64_t
///     22      8   feature_dim: uint64_t
///     30      4   CRC-32 over bytes [0, 30)
/// @endcode
class NGramHasher {
public:
  // -------------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------------

  /// @brief Construct a hasher with the given configuration.
  ///
  /// @pre @p config.feature_dim_ > 0 and is a power of 2.
  /// @pre @p config.min_n_ >= 1.
  /// @pre @p config.min_n_ <= @p config.max_n_.
  /// @throws std::invalid_argument  If any precondition is violated.
  explicit NGramHasher(NGramConfig config);

  // -------------------------------------------------------------------------
  // Feature extraction
  // -------------------------------------------------------------------------

  /// @brief Extract sparse n-gram features from @p key into @p out.
  ///
  /// @p out is cleared before filling. Each element is a
  /// @c (feature_index, count) pair where:
  ///  - @c feature_index is in @c [0, config().feature_dim_).
  ///  - @c count is the number of n-gram windows (across all lengths)
  ///    that mapped to that index.
  ///
  /// An empty key produces zero features. The output vector is reused
  /// across calls to amortise allocation cost — pass the same vector
  /// repeatedly in a tight loop.
  ///
  /// @param key  Byte span to featurise.
  /// @param out  Output buffer; cleared and filled by this call.
  void hash(std::span<const std::byte> key, std::vector<std::pair<uint32_t, float>> &out) const;

  // -------------------------------------------------------------------------
  // Introspection
  // -------------------------------------------------------------------------

  /// @brief Return the configuration this hasher was constructed with.
  [[nodiscard]] const NGramConfig &config() const noexcept;

  // -------------------------------------------------------------------------
  // Serialization
  // -------------------------------------------------------------------------

  /// @brief Serialize to @p os (binary, versioned, CRC-32 protected).
  /// @throws std::ios_base::failure  On write error (if exceptions enabled).
  void save(std::ostream &os) const;

  /// @brief Deserialize from @p is.
  /// @throws std::runtime_error      On magic/version mismatch or CRC-32 failure.
  /// @throws std::ios_base::failure  On read error.
  [[nodiscard]] static NGramHasher load(std::istream &is);

private:
  NGramConfig config_;
  uint32_t mask_ = 0; ///< Cached (feature_dim_ - 1) for fast modular index reduction.
};

} // namespace lbf
