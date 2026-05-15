// XXH_INLINE_ALL: header-only xxHash embedding (same pattern as hashing.hpp).
#define XXH_INLINE_ALL
#include "lbf/features/ngram_hasher.hpp"

#include "lbf/detail/crc32.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <xxhash.h>

namespace lbf {

// ---------------------------------------------------------------------------
// Serialization constants
// ---------------------------------------------------------------------------

// Magic bytes "LNG\0".
constexpr std::array<uint8_t, 4> NGRAM_MAGIC = {0x4CU, 0x4EU, 0x47U, 0x00U};
constexpr uint16_t NGRAM_VERSION = 1U;
// Fixed record size: 4 magic + 2 version + 3×8 fields + 4 CRC = 34 bytes.
constexpr size_t NGRAM_RECORD_BYTES  = 34;

// ---------------------------------------------------------------------------
// Constructor helpers
// ---------------------------------------------------------------------------

namespace {

/// Validate the NGramConfig and return the computed mask. Throws on error.
uint32_t validated_mask(const NGramConfig &cfg) {
  if (cfg.feature_dim_ == 0 || (cfg.feature_dim_ & (cfg.feature_dim_ - 1)) != 0) {
    throw std::invalid_argument("NGramHasher: feature_dim_ must be a non-zero power of 2");
  }
  if (cfg.min_n_ == 0) {
    throw std::invalid_argument("NGramHasher: min_n_ must be >= 1");
  }
  if (cfg.min_n_ > cfg.max_n_) {
    throw std::invalid_argument("NGramHasher: min_n_ must be <= max_n_");
  }
  return static_cast<uint32_t>(cfg.feature_dim_ - 1U);
}

} // namespace

// ---------------------------------------------------------------------------
// NGramHasher — public API
// ---------------------------------------------------------------------------

NGramHasher::NGramHasher(NGramConfig config) : config_(config), mask_(validated_mask(config)) {}

void NGramHasher::hash(std::span<const std::byte> key,
                       std::vector<std::pair<uint32_t, float>> &out) const {
  out.clear();
  if (key.empty()) {
    return;
  }

  const auto *data = reinterpret_cast<const uint8_t *>(key.data());
  const size_t len = key.size();

  // Estimate upper bound on indices to pre-size the buffer.
  // For each n in [min_n, max_n]: max len - n + 1 windows.
  std::vector<uint32_t> indices;
  for (size_t n = config_.min_n_; n <= config_.max_n_; ++n) {
    if (len < n) {
      break;
    }
    const size_t n_windows = len - n + 1;
    indices.reserve(indices.size() + n_windows);
    for (size_t i = 0; i < n_windows; ++i) {
      // Seed each gram-length independently to decorrelate features.
      const auto h = XXH3_64bits_withSeed(data + i, n, static_cast<XXH64_hash_t>(n));
      indices.push_back(static_cast<uint32_t>(h) & mask_);
    }
  }

  if (indices.empty()) {
    return;
  }

  // Sort, then run-length encode → sparse (index, count) pairs.
  std::sort(indices.begin(), indices.end());

  uint32_t cur = indices[0];
  float cnt = 1.0F;
  for (size_t i = 1; i < indices.size(); ++i) {
    if (indices[i] == cur) {
      cnt += 1.0F;
    } else {
      out.emplace_back(cur, cnt);
      cur = indices[i];
      cnt = 1.0F;
    }
  }
  out.emplace_back(cur, cnt);
}

const NGramConfig &NGramHasher::config() const noexcept {
  return config_;
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

void NGramHasher::save(std::ostream &os) const {
  // Build payload into a flat buffer so CRC covers exactly the bytes written.
  std::vector<uint8_t> buf;
  buf.reserve(NGRAM_RECORD_BYTES );

  const auto push_u16 = [&](uint16_t v) {
    buf.push_back(static_cast<uint8_t>(v));
    buf.push_back(static_cast<uint8_t>(v >> 8U));
  };
  const auto push_u64 = [&](uint64_t v) {
    for (unsigned i = 0; i < 8U; ++i) {
      buf.push_back(static_cast<uint8_t>(v >> (8U * i)));
    }
  };
  const auto push_u32 = [&](uint32_t v) {
    for (unsigned i = 0; i < 4U; ++i) {
      buf.push_back(static_cast<uint8_t>(v >> (8U * i)));
    }
  };

  for (const auto byte : NGRAM_MAGIC) {
    buf.push_back(byte);
  }
  push_u16(NGRAM_VERSION);
  push_u64(static_cast<uint64_t>(config_.min_n_));
  push_u64(static_cast<uint64_t>(config_.max_n_));
  push_u64(static_cast<uint64_t>(config_.feature_dim_));

  const uint32_t checksum = detail::crc32(std::as_bytes(std::span{buf}));
  push_u32(checksum);

  os.write(reinterpret_cast<const char *>(buf.data()), static_cast<std::streamsize>(buf.size()));
}

NGramHasher NGramHasher::load(std::istream &is) {
  std::vector<uint8_t> buf(NGRAM_RECORD_BYTES );
  if (!is.read(reinterpret_cast<char *>(buf.data()),
               static_cast<std::streamsize>(NGRAM_RECORD_BYTES ))) {
    throw std::runtime_error("NGramHasher::load: truncated stream");
  }

  // Validate magic.
  if (buf[0] != NGRAM_MAGIC[0] || buf[1] != NGRAM_MAGIC[1] || buf[2] != NGRAM_MAGIC[2] ||
      buf[3] != NGRAM_MAGIC[3]) {
    throw std::runtime_error("NGramHasher::load: invalid magic bytes");
  }

  // Validate version.
  const auto version = static_cast<uint16_t>(buf[4]) | (static_cast<uint16_t>(buf[5]) << 8U);
  if (version != 1U) {
    throw std::runtime_error("NGramHasher::load: unsupported version " + std::to_string(version));
  }

  // Validate CRC (covers bytes [0, 30); stored CRC at [30, 34)).
  constexpr size_t payload_bytes = 30;
  const uint32_t computed = detail::crc32(std::as_bytes(std::span{buf}.subspan(0, payload_bytes)));
  const uint32_t stored = static_cast<uint32_t>(buf[30]) | (static_cast<uint32_t>(buf[31]) << 8U) |
                          (static_cast<uint32_t>(buf[32]) << 16U) |
                          (static_cast<uint32_t>(buf[33]) << 24U);
  if (computed != stored) {
    throw std::runtime_error("NGramHasher::load: CRC-32 mismatch");
  }

  // Parse fields.
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

  return NGramHasher{cfg};
}

} // namespace lbf
