#pragma once

/// @file learned_bloom_filter.hpp
/// @brief Learned Bloom Filter: MembershipModel fast path + classical backup.
///
/// Implements the Kraska et al. (2018) Learned Bloom Filter design.  A learned
/// model scores each query key; keys scoring at or above @c threshold_ are
/// accepted immediately (predicted members).  Keys below the threshold are
/// tested against a classical Bloom filter that was pre-populated with every
/// member that the model under-scores — guaranteeing zero false negatives.
///
/// @par Construction
/// Use the static @ref build() factory.  It trains nothing — the caller
/// supplies a pre-trained @ref MembershipModel.
///
/// @par Zero-false-negative guarantee
/// Every true member is either scored ≥ threshold_ by the model (fast path)
/// or was inserted into the backup filter during construction.  Hence
/// @c contains(member) always returns @c true.
///
/// @par Thread safety
/// @c contains() is fully thread-safe.  @c save() and @c load() must be
/// synchronised externally.

#include "lbf/bloom_filter.hpp"
#include "lbf/model.hpp"

#include <cstddef>
#include <istream>
#include <memory>
#include <ostream>
#include <span>
#include <string_view>

namespace lbf {

/// @brief Learned Bloom Filter combining a @ref MembershipModel with a backup
///        classical @ref BloomFilter for a zero-false-negative guarantee.
class LearnedBloomFilter {
public:
  // -------------------------------------------------------------------------
  // Rule of Five — explicitly declared because copy is deleted (unique_ptr
  // member disables implicit copy; move and destructor are explicitly
  // defaulted to satisfy cppcoreguidelines-special-member-functions).
  // -------------------------------------------------------------------------

  LearnedBloomFilter(const LearnedBloomFilter &) = delete;
  LearnedBloomFilter &operator=(const LearnedBloomFilter &) = delete;
  LearnedBloomFilter(LearnedBloomFilter &&) = default;
  LearnedBloomFilter &operator=(LearnedBloomFilter &&) = default;
  ~LearnedBloomFilter() = default;

  // -------------------------------------------------------------------------
  // Factory
  // -------------------------------------------------------------------------

  /// @brief Build a LearnedBloomFilter from a pre-trained model.
  ///
  /// Scores every member key with the model.  Members scoring below
  /// @p threshold are inserted into the backup Bloom filter so that every
  /// true positive is always returned by @ref contains().
  ///
  /// @param model       Heap-allocated, pre-trained membership scorer.
  /// @param members     All true-positive keys (raw-byte spans).
  /// @param backup_fpr  Target false-positive rate for the backup filter.
  /// @param threshold   Decision threshold in (0, 1].  Keys scoring ≥ this
  ///                    are accepted without consulting the backup filter.
  ///
  /// @pre @p model != nullptr, @p members non-empty, @p backup_fpr ∈ (0, 1),
  ///      @p threshold ∈ (0, 1].
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  [[nodiscard]] static LearnedBloomFilter
  build(std::unique_ptr<MembershipModel> model,
        std::span<const std::span<const std::byte>> members,
        double backup_fpr = 0.01, double threshold = 0.5);

  // -------------------------------------------------------------------------
  // Query
  // -------------------------------------------------------------------------

  /// @brief Test membership of @p key.
  ///
  /// Returns @c true if the model scores the key at or above the threshold,
  /// OR if the backup filter says the key was inserted.  Never returns
  /// @c false for a key that was in the @p members list passed to @ref build.
  [[nodiscard]] bool contains(std::span<const std::byte> key) const noexcept;

  /// @brief String-view convenience overload; forwards to the byte-span primary.
  [[nodiscard]] bool contains(std::string_view key) const noexcept;

  // -------------------------------------------------------------------------
  // Introspection
  // -------------------------------------------------------------------------

  /// @brief Decision threshold used by @ref contains().
  [[nodiscard]] double threshold() const noexcept;

  /// @brief Number of members that were inserted into the backup filter.
  ///
  /// Equals the number of members whose model score was below the threshold
  /// at construction time.  Zero indicates the model handles all members.
  [[nodiscard]] size_t backup_count() const noexcept;

  /// @brief Approximate heap memory used by the backup filter, in bytes.
  ///
  /// Returns @c bit_count() / 8 for the backup @ref BloomFilter.
  /// Does not include the model's memory (accessible via the concrete model
  /// type before transferring ownership to @ref build()).
  [[nodiscard]] size_t memory_bytes() const noexcept;

  // -------------------------------------------------------------------------
  // Serialization
  // -------------------------------------------------------------------------

  /// @brief Serialize to @p os in the lbf binary format.
  ///
  /// Binary layout (all integers little-endian):
  /// @code
  ///   Offset  Size   Field
  ///   ------  -----  -----
  ///      0      4    Magic: 0x4C 0x4C 0x46 0x00 ("LLF\0")
  ///      4      2    Version: uint16_t = 1
  ///      6      8    threshold: double (IEEE 754 as uint64_t)
  ///     14      8    model_byte_count: uint64_t
  ///     22      N    model bytes (self-describing; any MembershipModel)
  ///   22+N     8    backup_byte_count: uint64_t
  ///   30+N     M    backup BloomFilter bytes (self-describing)
  ///  30+N+M    4    CRC-32 over all preceding bytes
  /// @endcode
  void save(std::ostream &os) const;

  /// @brief Deserialize from @p is.
  ///
  /// @throws std::runtime_error  On magic/version mismatch or CRC-32 failure.
  /// @throws std::ios_base::failure  On read error.
  [[nodiscard]] static LearnedBloomFilter load(std::istream &is);

private:
  LearnedBloomFilter(std::unique_ptr<MembershipModel> model, double threshold,
                     BloomFilter<> backup);

  std::unique_ptr<MembershipModel> model_;
  double threshold_;
  BloomFilter<> backup_;
};

} // namespace lbf
