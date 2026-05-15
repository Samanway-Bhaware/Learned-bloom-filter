#pragma once

/// @file detail/crc32.hpp
/// @brief Self-contained CRC-32 (ISO 3309 / PKZIP polynomial 0xEDB88320).
/// @note Internal implementation detail — do not include directly from user code.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace lbf::detail {

/// @brief Generate the 256-entry CRC-32 lookup table at compile time.
constexpr auto make_crc32_table() noexcept -> std::array<uint32_t, 256> {
  std::array<uint32_t, 256> tbl{};
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t crc = i;
    for (int j = 0; j < 8; ++j) {
      crc = (crc & 1U) != 0U ? (0xEDB8'8320U ^ (crc >> 1)) : (crc >> 1);
    }
    tbl[i] = crc;
  }
  return tbl;
}

/// @brief Precomputed CRC-32 lookup table (ISO 3309 reflected polynomial).
// NOLINTNEXTLINE(readability-identifier-naming) — UPPER_CASE required by tidy
inline constexpr std::array<uint32_t, 256> CRC32_TABLE = make_crc32_table();

/// @brief Compute CRC-32 over @p data.
///
/// Uses the ISO 3309 reflected polynomial (0xEDB88320), compatible with zlib,
/// gzip, and PKZIP. Initial value and final XOR mask are both 0xFFFFFFFF.
///
/// Standard test vector: crc32("123456789") == 0xCBF43926.
///
/// @param data  Byte span to checksum. Empty span returns 0x00000000.
/// @return      32-bit CRC checksum.
[[nodiscard]] inline uint32_t crc32(std::span<const std::byte> data) noexcept {
  uint32_t crc = 0xFFFF'FFFFU;
  for (const auto b : data) {
    const uint8_t idx = static_cast<uint8_t>(crc) ^ static_cast<uint8_t>(b);
    crc = CRC32_TABLE[idx] ^ (crc >> 8);
  }
  return crc ^ 0xFFFF'FFFFU;
}

} // namespace lbf::detail
