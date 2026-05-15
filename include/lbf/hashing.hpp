#pragma once

/// @file hashing.hpp
/// @brief Hashing primitives for lbf: Hash128, the Hasher concept,
///        the default XXH3Hasher, and Kirsch-Mitzenmacher derivation.

// XXH_INLINE_ALL: makes the entire xxHash implementation static-inline —
// no separate translation unit required. Recommended by the xxHash author
// for library embedding. Each TU gets its own copy (~2 KB text), which is
// negligible for typical project sizes.
#define XXH_INLINE_ALL
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <xxhash.h>

namespace lbf {

/// @brief 128-bit hash result used for Kirsch-Mitzenmacher double hashing.
///
/// The two halves are treated as two independent hash functions h1 and h2.
/// Both are produced by a single call to the underlying hash primitive,
/// avoiding a second full hash evaluation per probe.
struct Hash128 {
  uint64_t low_;  ///< h1: first independent 64-bit value.
  uint64_t high_; ///< h2: second independent 64-bit value.
};

/// @brief Concept satisfied by any type usable as a hash function in lbf.
///
/// A conforming type must expose a static @c hash128 method that is
/// @c noexcept, accepts a byte span and an optional uint64_t seed, and
/// returns a @ref Hash128.
///
/// @par Implementing a custom hasher
/// @code
/// struct MyHasher {
///   static lbf::Hash128
///   hash128(std::span<const std::byte> data, uint64_t seed = 0) noexcept;
/// };
/// static_assert(lbf::Hasher<MyHasher>);
/// @endcode
template <typename H>
concept Hasher = requires(std::span<const std::byte> data, uint64_t seed) {
  { H::hash128(data, seed) } -> std::same_as<Hash128>;
};

/// @brief Default hasher backed by XXH3 (128-bit variant).
///
/// Calls @c XXH3_128bits_withSeed internally. The canonical XXH128_hash_t
/// result is split: @c low64 → @ref Hash128::low_ (h1),
///                            @c high64 → @ref Hash128::high_ (h2).
/// These two values are suitable as independent hash functions for the
/// Kirsch-Mitzenmacher double-hashing scheme.
///
/// @note The @p seed shifts the hash family. It is NOT a keyed MAC;
///       do not use it for security-sensitive applications.
struct XXH3Hasher {
  /// @brief Hash @p data into a @ref Hash128 using XXH3_128bits_withSeed.
  /// @param data  Byte span to hash. An empty span is valid and returns a
  ///              well-defined, non-zero hash.
  /// @param seed  Optional seed (default 0). A non-zero seed produces an
  ///              independent hash family — useful for stacking filters.
  /// @return      128-bit hash split into two independent 64-bit halves.
  // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
  [[nodiscard]] static Hash128 hash128(std::span<const std::byte> bytes,
                                       uint64_t seed = 0) noexcept {
    const XXH128_hash_t h = XXH3_128bits_withSeed(bytes.data(), bytes.size_bytes(), seed);
    return {h.low64, h.high64}; // XXH128_hash_t fields, not Hash128 members
  }
};

/// @brief Derive the @p i-th bit-index candidate (Kirsch-Mitzenmacher).
///
/// Given two independent hash values h1 = @p h.low_ and h2 = @p h.high_, the
/// i-th derived hash is:
/// @f[ g_i(x) = h_1(x) + i \cdot h_2(x) \pmod{m} @f]
///
/// The result is unreduced. Callers must apply @c % @c m before using it
/// as a bit index:
/// @code
/// size_t bit = derive_hash(h, i) % m;
/// @endcode
///
/// Overflow wraps modulo 2^64 by design — unsigned arithmetic is intentional.
///
/// Reference: Kirsch & Mitzenmacher, "Less Hashing, Same Performance", ESA 2006.
///
/// @param h  Hash pair from a single @ref Hasher::hash128 call.
/// @param i  Hash index in [0, k).
/// @return   Unreduced 64-bit derived hash.
[[nodiscard]] constexpr uint64_t derive_hash(Hash128 h, size_t i) noexcept {
  return h.low_ + static_cast<uint64_t>(i) * h.high_;
}

} // namespace lbf
