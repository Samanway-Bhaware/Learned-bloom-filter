/// @file learned_bloom_filter.cpp
/// @brief Implementation of @ref LearnedBloomFilter.

#include "lbf/learned_bloom_filter.hpp"

#include "lbf/detail/crc32.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lbf {

// ---------------------------------------------------------------------------
// Serialization constants
// ---------------------------------------------------------------------------

constexpr std::array<uint8_t, 4> LLF_MAGIC = {0x4CU, 0x4CU, 0x46U, 0x00U};
constexpr uint16_t LLF_VERSION = 1U;

// ---------------------------------------------------------------------------
// Private constructor
// ---------------------------------------------------------------------------

LearnedBloomFilter::LearnedBloomFilter(std::unique_ptr<MembershipModel> model,
                                       double threshold, BloomFilter<> backup)
    : model_(std::move(model)), threshold_(threshold),
      backup_(std::move(backup)) {}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

LearnedBloomFilter
LearnedBloomFilter::build(std::unique_ptr<MembershipModel> model,
                          std::span<const std::span<const std::byte>> members,
                          double backup_fpr, double threshold) {
  // Count members whose model score falls below the threshold.
  // These would be false negatives; they need the backup filter.
  size_t n_backup = 0;
  for (const auto &key : members) {
    if (model->predict(key) < threshold) {
      ++n_backup;
    }
  }

  // Construct backup filter — minimum 1 expected item to satisfy the
  // BloomFilter precondition even when the model covers every member.
  BloomFilter<> backup{
      BloomFilterParams{n_backup == 0 ? size_t{1} : n_backup, backup_fpr}};

  // Insert false-negative members so they are never missed.
  for (const auto &key : members) {
    if (model->predict(key) < threshold) {
      backup.insert(key);
    }
  }

  return LearnedBloomFilter{std::move(model), threshold, std::move(backup)};
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

bool LearnedBloomFilter::contains(std::span<const std::byte> key) const
    noexcept {
  return model_->predict(key) >= threshold_ || backup_.contains(key);
}

bool LearnedBloomFilter::contains(std::string_view key) const noexcept {
  return contains(
      std::as_bytes(std::span<const char>{key.data(), key.size()}));
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

double LearnedBloomFilter::threshold() const noexcept { return threshold_; }

size_t LearnedBloomFilter::backup_count() const noexcept {
  return backup_.inserted_count();
}

size_t LearnedBloomFilter::memory_bytes() const noexcept {
  return backup_.bit_count() / 8U;
}

// ---------------------------------------------------------------------------
// Serialization — save
// ---------------------------------------------------------------------------

void LearnedBloomFilter::save(std::ostream &os) const {
  // Serialize model and backup filter into memory strings so we know sizes.
  std::ostringstream model_ss;
  model_->save(model_ss);
  const std::string model_str = model_ss.str();

  std::ostringstream backup_ss;
  backup_.save(backup_ss);
  const std::string backup_str = backup_ss.str();

  // Build the full payload in a single buffer for CRC computation.
  std::vector<uint8_t> buf;
  buf.reserve(30 + model_str.size() + backup_str.size());

  const auto push_u8 = [&](uint8_t val) { buf.push_back(val); };
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

  // Magic + version.
  for (const auto byte : LLF_MAGIC) {
    push_u8(byte);
  }
  push_u16(LLF_VERSION);

  // Threshold as IEEE 754 double stored little-endian.
  push_u64(std::bit_cast<uint64_t>(threshold_));

  // Model bytes (length-prefixed).
  push_u64(static_cast<uint64_t>(model_str.size()));
  for (const char chr : model_str) {
    push_u8(static_cast<uint8_t>(chr));
  }

  // Backup bytes (length-prefixed).
  push_u64(static_cast<uint64_t>(backup_str.size()));
  for (const char chr : backup_str) {
    push_u8(static_cast<uint8_t>(chr));
  }

  // CRC-32 over all preceding bytes.
  const uint32_t checksum = detail::crc32(std::as_bytes(std::span{buf}));
  push_u32(checksum);

  os.write(reinterpret_cast<const char *>(buf.data()),
           static_cast<std::streamsize>(buf.size()));
}

// ---------------------------------------------------------------------------
// Serialization — load
// ---------------------------------------------------------------------------

LearnedBloomFilter LearnedBloomFilter::load(std::istream &is) {
  // Fixed prefix: magic(4) + version(2) + threshold(8) = 14 bytes.
  constexpr size_t PREFIX_BYTES = 14U;
  std::vector<uint8_t> buf(PREFIX_BYTES);
  if (!is.read(reinterpret_cast<char *>(buf.data()),
               static_cast<std::streamsize>(PREFIX_BYTES))) {
    throw std::runtime_error("LearnedBloomFilter::load: truncated header");
  }

  // Validate magic.
  if (buf[0] != LLF_MAGIC[0] || buf[1] != LLF_MAGIC[1] ||
      buf[2] != LLF_MAGIC[2] || buf[3] != LLF_MAGIC[3]) {
    throw std::runtime_error(
        "LearnedBloomFilter::load: invalid magic bytes");
  }

  // Validate version.
  const auto version =
      static_cast<uint16_t>(buf[4]) | (static_cast<uint16_t>(buf[5]) << 8U);
  if (version != LLF_VERSION) {
    throw std::runtime_error(
        "LearnedBloomFilter::load: unsupported version " +
        std::to_string(version));
  }

  // Parse threshold from buf[6..13] (LE uint64 → double via bit_cast).
  uint64_t raw_threshold = 0;
  for (int i = 0; i < 8; ++i) {
    raw_threshold |= static_cast<uint64_t>(buf[6U + static_cast<size_t>(i)])
                     << (8U * static_cast<unsigned>(i));
  }
  const double threshold = std::bit_cast<double>(raw_threshold);

  // Helper: read a length-prefixed blob, appending all bytes to buf.
  // Returns {offset_of_blob_data_in_buf, blob_byte_count}.
  const auto read_blob =
      [&](const char *err_count,
          const char *err_data) -> std::pair<size_t, size_t> {
    std::array<uint8_t, 8> count_buf{};
    if (!is.read(reinterpret_cast<char *>(count_buf.data()), 8)) {
      throw std::runtime_error(err_count);
    }
    for (const auto b : count_buf) {
      buf.push_back(b);
    }

    uint64_t count = 0;
    for (int i = 0; i < 8; ++i) {
      count |= static_cast<uint64_t>(count_buf[static_cast<size_t>(i)])
               << (8U * static_cast<unsigned>(i));
    }

    const size_t data_offset = buf.size();
    buf.resize(data_offset + static_cast<size_t>(count));
    if (count > 0 &&
        !is.read(reinterpret_cast<char *>(buf.data() + data_offset),
                 static_cast<std::streamsize>(count))) {
      throw std::runtime_error(err_data);
    }
    return {data_offset, static_cast<size_t>(count)};
  };

  const auto [model_offset, model_count] =
      read_blob("LearnedBloomFilter::load: truncated model byte count",
                "LearnedBloomFilter::load: truncated model bytes");

  const auto [backup_offset, backup_count] =
      read_blob("LearnedBloomFilter::load: truncated backup byte count",
                "LearnedBloomFilter::load: truncated backup bytes");

  // Read and validate CRC-32.
  std::array<uint8_t, 4> crc_buf{};
  if (!is.read(reinterpret_cast<char *>(crc_buf.data()), 4)) {
    throw std::runtime_error("LearnedBloomFilter::load: truncated CRC");
  }
  const uint32_t stored_crc =
      static_cast<uint32_t>(crc_buf[0]) |
      (static_cast<uint32_t>(crc_buf[1]) << 8U) |
      (static_cast<uint32_t>(crc_buf[2]) << 16U) |
      (static_cast<uint32_t>(crc_buf[3]) << 24U);
  const uint32_t computed_crc = detail::crc32(std::as_bytes(std::span{buf}));
  if (computed_crc != stored_crc) {
    throw std::runtime_error(
        "LearnedBloomFilter::load: CRC-32 mismatch — data may be corrupted");
  }

  // Reconstruct model from its embedded bytes (null-byte safe).
  const std::string model_str(
      reinterpret_cast<const char *>(buf.data() + model_offset), model_count);
  std::istringstream model_ss{model_str};
  auto model = MembershipModel::load(model_ss);

  // Reconstruct backup BloomFilter.
  const std::string backup_str(
      reinterpret_cast<const char *>(buf.data() + backup_offset),
      backup_count);
  std::istringstream backup_ss{backup_str};
  BloomFilter<> backup = BloomFilter<>::load(backup_ss);

  return LearnedBloomFilter{std::move(model), threshold, std::move(backup)};
}

} // namespace lbf
