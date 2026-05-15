#pragma once

/// @file bloom_filter.hpp
/// @brief Classical Bloom filter with zero false-negative guarantee.

#include "lbf/detail/crc32.hpp"
#include "lbf/hashing.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <numbers>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace lbf {

// ---------------------------------------------------------------------------
// BloomFilterParams
// ---------------------------------------------------------------------------

/// @brief Construction parameters for a @ref BloomFilter.
///
/// Given expected item count @c n and target false-positive rate @c p, the
/// optimal bit-array size and hash count are derived as:
/// @f[ m = \left\lceil -\frac{n \ln p}{(\ln 2)^2} \right\rceil \qquad
///     k = \left\lfloor \frac{m}{n} \ln 2 + 0.5 \right\rfloor @f]
///
/// @c bit_count() and @c hash_count() compute these on demand.
/// @ref BloomFilter caches the results on construction.
struct BloomFilterParams {
  size_t expected_items_; ///< Expected distinct items to insert (n). Must be > 0.
  double target_fpr_;     ///< Target false-positive rate p, strictly in (0, 1).

  /// @brief Compute the optimal bit-array size m.
  /// @pre @c expected_items > 0, @c target_fpr in (0, 1).
  /// @return Optimal m >= 1.
  [[nodiscard]] size_t bit_count() const noexcept;

  /// @brief Compute the optimal hash function count k.
  /// @pre @c expected_items > 0.
  /// @return Optimal k >= 1.
  [[nodiscard]] size_t hash_count() const noexcept;
};

// ---------------------------------------------------------------------------
// Serialization format (binary, all integers little-endian)
// ---------------------------------------------------------------------------
//
//   Offset   Size   Field
//   ------   ----   -----
//      0       4    Magic:          0x4C 0x42 0x46 0x00  ("LBF\0")
//      4       2    Version:        uint16_t = 1
//      6       8    expected_items: uint64_t
//     14       8    target_fpr:     double (IEEE 754, as uint64_t via bit_cast)
//     22       8    m (bit_count):  uint64_t
//     30       8    k (hash_count): uint64_t
//     38       8    inserted_count: uint64_t
//     46       8    words:          uint64_t  (= ceil(m / 64))
//     54    8*W     bit array:      W x uint64_t (little-endian each)
//   54+8W     4    CRC-32:         uint32_t over all preceding bytes

// ---------------------------------------------------------------------------
// BloomFilter
// ---------------------------------------------------------------------------

/// @brief Space-efficient probabilistic set-membership filter.
///
/// Answers "is @p key in the set?" with zero false-negative rate and a
/// configurable false-positive rate bounded by @c target_fpr when
/// @c inserted_count() <= @c expected_items.
///
/// The bit array is backed by @c std::vector<uint64_t> with word-level
/// bit operations. Keys are consumed as raw bytes via
/// @c std::span<const std::byte>; typed convenience overloads handle
/// trivially-copyable values and @c std::string_view without allocation.
///
/// @tparam H  Hash function type satisfying the @ref Hasher concept.
///            Defaults to @ref XXH3Hasher (xxHash3, 128-bit output).
///
/// @par Thread safety
/// Concurrent @c contains calls are safe. Concurrent @c insert, or a mix
/// of @c insert and @c contains, requires external synchronisation.
///
/// @par Example
/// @code
/// lbf::BloomFilter<> bf{{.expected_items_ = 1'000'000, .target_fpr_ = 0.01}};
/// bf.insert(std::string_view{"https://example.com"});
/// assert(bf.contains(std::string_view{"https://example.com"}));
/// @endcode
template <Hasher H = XXH3Hasher>
class BloomFilter {
public:
  // -------------------------------------------------------------------------
  // Construction / rule of five
  // -------------------------------------------------------------------------

  /// @brief Construct a BloomFilter sized for @p params.
  ///
  /// Computes and caches m and k. Allocates a zero-initialised bit array.
  ///
  /// @throws std::invalid_argument if @c params.expected_items == 0 or
  ///         @c params.target_fpr is not strictly in (0, 1).
  explicit BloomFilter(BloomFilterParams params);

  BloomFilter(const BloomFilter &) = default;
  BloomFilter(BloomFilter &&) noexcept = default;
  BloomFilter &operator=(const BloomFilter &) = default;
  BloomFilter &operator=(BloomFilter &&) noexcept = default;
  ~BloomFilter() = default;

  // -------------------------------------------------------------------------
  // Core API — byte-oriented (primary)
  // -------------------------------------------------------------------------

  /// @brief Insert @p key. After this call @c contains(key) is guaranteed true.
  /// @param key  Raw byte representation of the key. Empty spans are valid.
  void insert(std::span<const std::byte> key) noexcept;

  /// @brief Test membership of @p key.
  /// @return @c true if probably inserted; @c false if definitely not inserted.
  [[nodiscard]] bool contains(std::span<const std::byte> key) const noexcept;

  // -------------------------------------------------------------------------
  // Convenience overloads (all forward to the byte-span primary)
  // -------------------------------------------------------------------------

  /// @brief Insert any trivially-copyable, non-pointer value via its object bytes.
  ///
  /// Equivalent to inserting @c std::as_bytes(std::span{&key, 1}).
  /// Do not use for pointer types or types with unspecified padding.
  ///
  /// @tparam T  Must be trivially copyable and not a pointer type.
  template <typename T>
    requires std::is_trivially_copyable_v<T> && (!std::is_pointer_v<T>)
  void insert(const T &key) noexcept;

  /// @brief Insert a string key by its UTF-8 byte content.
  void insert(std::string_view key) noexcept;

  /// @brief Test membership of any trivially-copyable, non-pointer value.
  template <typename T>
    requires std::is_trivially_copyable_v<T> && (!std::is_pointer_v<T>)
  [[nodiscard]] bool contains(const T &key) const noexcept;

  /// @brief Test membership of a string key.
  [[nodiscard]] bool contains(std::string_view key) const noexcept;

  // -------------------------------------------------------------------------
  // Introspection
  // -------------------------------------------------------------------------

  /// @brief Total bits in the bit array (m), as computed at construction.
  [[nodiscard]] size_t bit_count() const noexcept;

  /// @brief Hash functions applied per key (k), as computed at construction.
  [[nodiscard]] size_t hash_count() const noexcept;

  /// @brief Number of @c insert calls made since construction (or last load).
  [[nodiscard]] size_t inserted_count() const noexcept;

  /// @brief Analytical estimate of the current false-positive rate.
  ///
  /// Formula: @f$(1 - e^{-k \cdot n / m})^k@f$ where n = @c inserted_count().
  /// Returns 0 when no items have been inserted, and approaches 1 as the
  /// filter fills well beyond @c expected_items.
  [[nodiscard]] double estimated_fpr() const noexcept;

  /// @brief Fraction of bits currently set to 1.
  ///
  /// Runs in O(m/64) time via @c std::popcount. Useful for diagnosing
  /// over-insertion or checking filter saturation.
  [[nodiscard]] double load_factor() const noexcept;

  // -------------------------------------------------------------------------
  // Serialization
  // -------------------------------------------------------------------------

  /// @brief Serialize to @p os in the lbf binary format (versioned + CRC-32).
  ///
  /// The stream must be open in binary mode (@c std::ios::binary).
  /// @throws std::ios_base::failure on write error (if @p os has exceptions set).
  void save(std::ostream &os) const;

  /// @brief Deserialize from @p is.
  ///
  /// Validates magic bytes, format version, and CRC-32 before reconstructing.
  ///
  /// @throws std::runtime_error  On magic/version mismatch or CRC-32 failure.
  /// @throws std::ios_base::failure  On read error (if @p is has exceptions set).
  [[nodiscard]] static BloomFilter load(std::istream &is);

  // -------------------------------------------------------------------------
  // Equality
  // -------------------------------------------------------------------------

  /// @brief Structural equality: same params, same bits, same insert count.
  ///
  /// Primarily used to verify serialization round-trips.
  [[nodiscard]] bool operator==(const BloomFilter &other) const noexcept;

private:
  BloomFilterParams params_;
  size_t m_;                     ///< Cached bit_count — avoids recomputation in hot path.
  size_t k_;                     ///< Cached hash_count.
  std::vector<uint64_t> bits_{}; ///< Bit array; size = ceil(m_ / 64) words.
  size_t inserted_count_ = 0;

  /// @brief Set bit at position @p idx (unchecked; idx must be < m_).
  void set_bit(size_t idx) noexcept;

  /// @brief Read bit at position @p idx (unchecked; idx must be < m_).
  [[nodiscard]] bool get_bit(size_t idx) const noexcept;
};

// ===========================================================================
// BloomFilterParams — inline definitions
// ===========================================================================

inline size_t BloomFilterParams::bit_count() const noexcept {
  // m = ceil(-n * ln(p) / (ln 2)^2)
  const double ln2 = std::numbers::ln2;
  const double m = -static_cast<double>(expected_items_) * std::log(target_fpr_) / (ln2 * ln2);
  return static_cast<size_t>(std::ceil(m));
}

inline size_t BloomFilterParams::hash_count() const noexcept {
  // k = round((m / n) * ln 2), minimum 1
  const double k =
      (static_cast<double>(bit_count()) / static_cast<double>(expected_items_)) * std::numbers::ln2;
  return std::max(size_t{1}, static_cast<size_t>(std::round(k)));
}

// ===========================================================================
// BloomFilter<H> — method definitions
// ===========================================================================

template <Hasher H>
BloomFilter<H>::BloomFilter(BloomFilterParams params) : params_(params) {
  if (params_.expected_items_ == 0) {
    throw std::invalid_argument("lbf::BloomFilter: expected_items must be > 0");
  }
  if (params_.target_fpr_ <= 0.0 || params_.target_fpr_ >= 1.0) {
    throw std::invalid_argument("lbf::BloomFilter: target_fpr must be strictly in (0, 1)");
  }
  m_ = params_.bit_count();
  k_ = params_.hash_count();
  bits_.assign((m_ + 63) / 64, uint64_t{0});
}

// --- Core API ---------------------------------------------------------------

template <Hasher H>
void BloomFilter<H>::insert(std::span<const std::byte> key) noexcept {
  const Hash128 h = H::hash128(key);
  for (size_t i = 0; i < k_; ++i) {
    set_bit(derive_hash(h, i) % m_);
  }
  ++inserted_count_;
}

template <Hasher H>
bool BloomFilter<H>::contains(std::span<const std::byte> key) const noexcept {
  const Hash128 h = H::hash128(key);
  for (size_t i = 0; i < k_; ++i) {
    if (!get_bit(derive_hash(h, i) % m_)) {
      return false;
    }
  }
  return true;
}

// --- Convenience overloads --------------------------------------------------

template <Hasher H>
template <typename T>
  requires std::is_trivially_copyable_v<T> && (!std::is_pointer_v<T>)
void BloomFilter<H>::insert(const T &key) noexcept {
  insert(std::as_bytes(std::span{&key, 1}));
}

template <Hasher H>
void BloomFilter<H>::insert(std::string_view key) noexcept {
  insert(std::as_bytes(std::span{key.data(), key.size()}));
}

template <Hasher H>
template <typename T>
  requires std::is_trivially_copyable_v<T> && (!std::is_pointer_v<T>)
bool BloomFilter<H>::contains(const T &key) const noexcept {
  return contains(std::as_bytes(std::span{&key, 1}));
}

template <Hasher H>
bool BloomFilter<H>::contains(std::string_view key) const noexcept {
  return contains(std::as_bytes(std::span{key.data(), key.size()}));
}

// --- Introspection ----------------------------------------------------------

template <Hasher H>
size_t BloomFilter<H>::bit_count() const noexcept {
  return m_;
}

template <Hasher H>
size_t BloomFilter<H>::hash_count() const noexcept {
  return k_;
}

template <Hasher H>
size_t BloomFilter<H>::inserted_count() const noexcept {
  return inserted_count_;
}

template <Hasher H>
double BloomFilter<H>::estimated_fpr() const noexcept {
  if (inserted_count_ == 0) {
    return 0.0;
  }
  const double exponent =
      -static_cast<double>(k_) * static_cast<double>(inserted_count_) / static_cast<double>(m_);
  return std::pow(1.0 - std::exp(exponent), static_cast<double>(k_));
}

template <Hasher H>
double BloomFilter<H>::load_factor() const noexcept {
  size_t set_bits = 0;
  for (const uint64_t word : bits_) {
    set_bits += static_cast<size_t>(std::popcount(word));
  }
  return static_cast<double>(set_bits) / static_cast<double>(m_);
}

// --- Serialization ----------------------------------------------------------

template <Hasher H>
void BloomFilter<H>::save(std::ostream &os) const {
  const size_t words = bits_.size();

  // Build the full payload in memory so CRC covers exactly the bytes written.
  std::vector<uint8_t> buf;
  buf.reserve(54 + words * sizeof(uint64_t));

  // Typed little-endian helpers — avoids a C++20 template lambda whose
  // `requires` clause clang-format wraps differently across versions.
  const auto push_u16 = [&](uint16_t val) {
    buf.push_back(static_cast<uint8_t>(val));
    buf.push_back(static_cast<uint8_t>(val >> 8U));
  };
  const auto push_u32 = [&](uint32_t val) {
    for (unsigned i = 0; i < 4U; ++i) {
      buf.push_back(static_cast<uint8_t>(val >> (8U * i)));
    }
  };
  const auto push_u64 = [&](uint64_t val) {
    for (unsigned i = 0; i < 8U; ++i) {
      buf.push_back(static_cast<uint8_t>(val >> (8U * i)));
    }
  };

  // Magic ("LBF\0") + version 1.
  buf.push_back(uint8_t{0x4C});
  buf.push_back(uint8_t{0x42});
  buf.push_back(uint8_t{0x46});
  buf.push_back(uint8_t{0x00});
  push_u16(uint16_t{1});

  // Header fields.
  push_u64(static_cast<uint64_t>(params_.expected_items_));
  push_u64(std::bit_cast<uint64_t>(params_.target_fpr_));
  push_u64(static_cast<uint64_t>(m_));
  push_u64(static_cast<uint64_t>(k_));
  push_u64(static_cast<uint64_t>(inserted_count_));
  push_u64(static_cast<uint64_t>(words));

  // Bit array (each word little-endian).
  for (const uint64_t word : bits_) {
    push_u64(word);
  }

  // CRC-32 over everything written so far.
  const uint32_t checksum = detail::crc32(std::as_bytes(std::span{buf}));
  push_u32(checksum);

  os.write(reinterpret_cast<const char *>(buf.data()), static_cast<std::streamsize>(buf.size()));
}

template <Hasher H>
BloomFilter<H> BloomFilter<H>::load(std::istream &is) {
  // Read fixed-size header (54 bytes).
  constexpr size_t header_bytes = 54;
  std::vector<uint8_t> buf(header_bytes);
  if (!is.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(header_bytes))) {
    throw std::runtime_error("lbf::BloomFilter::load: truncated header");
  }

  // Validate magic bytes.
  if (buf[0] != 0x4CU || buf[1] != 0x42U || buf[2] != 0x46U || buf[3] != 0x00U) {
    throw std::runtime_error("lbf::BloomFilter::load: invalid magic bytes");
  }

  // Validate format version.
  const auto version = static_cast<uint16_t>(buf[4]) | (static_cast<uint16_t>(buf[5]) << 8U);
  if (version != 1) {
    throw std::runtime_error("lbf::BloomFilter::load: unsupported format version " +
                             std::to_string(version));
  }

  // Parse a little-endian uint64_t at @p offset inside buf.
  const auto read_u64le = [&](size_t offset) -> uint64_t {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
      v |= static_cast<uint64_t>(buf[offset + static_cast<size_t>(i)])
           << (8U * static_cast<unsigned>(i));
    }
    return v;
  };

  const auto stored_n = static_cast<size_t>(read_u64le(6));
  const auto stored_p = std::bit_cast<double>(read_u64le(14));
  const auto stored_m = static_cast<size_t>(read_u64le(22));
  const auto stored_k = static_cast<size_t>(read_u64le(30));
  const auto stored_inserted = static_cast<size_t>(read_u64le(38));
  const auto words = static_cast<size_t>(read_u64le(46));

  // Read bit array into buf (grow it to hold the full payload).
  const size_t bit_array_bytes = words * sizeof(uint64_t);
  buf.resize(header_bytes + bit_array_bytes);
  if (bit_array_bytes > 0 && !is.read(reinterpret_cast<char *>(buf.data() + header_bytes),
                                      static_cast<std::streamsize>(bit_array_bytes))) {
    throw std::runtime_error("lbf::BloomFilter::load: truncated bit array");
  }

  // Read stored CRC-32 (4 bytes after the payload).
  std::array<uint8_t, 4> crc_bytes{};
  if (!is.read(reinterpret_cast<char *>(crc_bytes.data()), 4)) {
    throw std::runtime_error("lbf::BloomFilter::load: missing CRC-32");
  }
  const uint32_t stored_crc =
      static_cast<uint32_t>(crc_bytes[0]) | (static_cast<uint32_t>(crc_bytes[1]) << 8U) |
      (static_cast<uint32_t>(crc_bytes[2]) << 16U) | (static_cast<uint32_t>(crc_bytes[3]) << 24U);

  // Verify integrity.
  const uint32_t computed_crc = detail::crc32(std::as_bytes(std::span{buf}));
  if (computed_crc != stored_crc) {
    throw std::runtime_error("lbf::BloomFilter::load: CRC-32 mismatch — data may be corrupted");
  }

  // Reconstruct the filter from params.
  BloomFilterParams params{stored_n, stored_p};
  BloomFilter<H> filter(params);

  // Sanity-check that recomputed m/k match what was stored.
  if (filter.m_ != stored_m || filter.k_ != stored_k) {
    throw std::runtime_error(
        "lbf::BloomFilter::load: stored m/k inconsistent with recomputed params");
  }

  filter.inserted_count_ = stored_inserted;

  // Parse bit array words (little-endian).
  for (size_t w = 0; w < words; ++w) {
    const size_t off = header_bytes + w * sizeof(uint64_t);
    uint64_t word = 0;
    for (int b = 0; b < 8; ++b) {
      word |= static_cast<uint64_t>(buf[off + static_cast<size_t>(b)])
              << (8U * static_cast<unsigned>(b));
    }
    filter.bits_[w] = word;
  }

  return filter;
}

// --- Equality ---------------------------------------------------------------

template <Hasher H>
bool BloomFilter<H>::operator==(const BloomFilter &other) const noexcept {
  return params_.expected_items_ == other.params_.expected_items_ &&
         params_.target_fpr_ == other.params_.target_fpr_ && m_ == other.m_ && k_ == other.k_ &&
         inserted_count_ == other.inserted_count_ && bits_ == other.bits_;
}

// --- Private helpers --------------------------------------------------------

template <Hasher H>
void BloomFilter<H>::set_bit(size_t idx) noexcept {
  bits_[idx >> 6U] |= (uint64_t{1} << (idx & 63U));
}

template <Hasher H>
bool BloomFilter<H>::get_bit(size_t idx) const noexcept {
  return ((bits_[idx >> 6U] >> (idx & 63U)) & uint64_t{1}) != 0U;
}

} // namespace lbf
