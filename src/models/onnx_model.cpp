// This translation unit is intentionally empty when LBF_ENABLE_ONNX is not defined.
// The #ifdef wrapper means clang-tidy sees an empty file in the default CI build.
#ifdef LBF_ENABLE_ONNX

#include "lbf/models/onnx_model.hpp"

#include "lbf/detail/crc32.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <onnxruntime_cxx_api.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lbf {

// ---------------------------------------------------------------------------
// Serialization constants
// ---------------------------------------------------------------------------

constexpr std::array<uint8_t, 4> LOX_MAGIC = {0x4CU, 0x4FU, 0x58U, 0x00U};
constexpr uint16_t LOX_VERSION = 1U;
// NGramHasher sub-record is always 34 bytes (4 magic + 2 version + 3×8 fields + 4 CRC).
constexpr size_t NGRAM_RECORD_BYTES = 34U;

// ---------------------------------------------------------------------------
// PIMPL implementation struct
// ---------------------------------------------------------------------------

struct OnnxModel::Impl {
  NGramHasher hasher_;
  std::vector<uint8_t> onnx_bytes_;
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "lbf"};
  Ort::Session session_{nullptr};
  std::string input_name_;
  std::string output_name_;

  void create_session() {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    session_ = Ort::Session{env_, onnx_bytes_.data(), onnx_bytes_.size(), opts};

    Ort::AllocatorWithDefaultOptions alloc;
    {
      auto ptr = session_.GetInputNameAllocated(0, alloc);
      input_name_ = std::string{ptr.get()};
    }
    {
      auto ptr = session_.GetOutputNameAllocated(0, alloc);
      output_name_ = std::string{ptr.get()};
    }
  }
};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

OnnxModel::OnnxModel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

std::unique_ptr<OnnxModel> OnnxModel::load_from_file(const std::filesystem::path &onnx_path,
                                                     NGramConfig feature_config) {
  std::ifstream ifs{onnx_path, std::ios::binary};
  if (!ifs) {
    throw std::runtime_error("OnnxModel::load_from_file: cannot open " + onnx_path.string());
  }
  std::vector<uint8_t> bytes{std::istreambuf_iterator<char>{ifs}, std::istreambuf_iterator<char>{}};
  if (!ifs && !ifs.eof()) {
    throw std::runtime_error("OnnxModel::load_from_file: read error for " + onnx_path.string());
  }

  auto impl = std::make_unique<Impl>();
  impl->hasher_ = NGramHasher{feature_config};
  impl->onnx_bytes_ = std::move(bytes);
  impl->create_session();
  return std::unique_ptr<OnnxModel>{new OnnxModel{std::move(impl)}};
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

namespace {

[[nodiscard]] double onnx_stable_sigmoid(double z) noexcept {
  if (z >= 0.0) {
    return 1.0 / (1.0 + std::exp(-z));
  }
  const double e = std::exp(z);
  return e / (1.0 + e);
}

} // namespace

double OnnxModel::predict(std::span<const std::byte> key) const noexcept {
  try {
    const size_t dim = impl_->hasher_.config().feature_dim_;

    thread_local std::vector<float> feat_buf;
    thread_local std::vector<std::pair<uint32_t, float>> features;

    feat_buf.assign(dim, 0.0F);
    impl_->hasher_.hash(key, features);
    for (const auto &[idx, cnt] : features) {
      feat_buf[idx] = cnt;
    }

    std::array<int64_t, 2> shape{1LL, static_cast<int64_t>(dim)};
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(mem_info, feat_buf.data(), feat_buf.size(),
                                                        shape.data(), shape.size());

    std::array<const char *, 1> input_names{impl_->input_name_.c_str()};
    std::array<const char *, 1> output_names{impl_->output_name_.c_str()};

    auto outputs = impl_->session_.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor,
                                       1, output_names.data(), 1);

    // Reset only the touched feature buckets (sparse update).
    for (const auto &[idx, cnt] : features) {
      feat_buf[idx] = 0.0F;
    }

    const float logit = *outputs[0].GetTensorData<float>();
    return onnx_stable_sigmoid(static_cast<double>(logit));
  } catch (...) {
    return 0.5; // fallback on ORT error — neutral score
  }
}

// ---------------------------------------------------------------------------
// Serialization — save
// ---------------------------------------------------------------------------

void OnnxModel::save(std::ostream &os) const {
  std::vector<uint8_t> buf;

  const auto push_u8 = [&](uint8_t v) { buf.push_back(v); };
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

  // Magic + version.
  for (const auto byte : LOX_MAGIC) {
    push_u8(byte);
  }
  push_u16(LOX_VERSION);

  // NGramHasher sub-record (captured via ostringstream into main buffer).
  {
    std::ostringstream hasher_ss;
    impl_->hasher_.save(hasher_ss);
    const std::string hasher_str = hasher_ss.str();
    for (const char c : hasher_str) {
      push_u8(static_cast<uint8_t>(c));
    }
  }

  // ONNX model bytes (count + raw bytes).
  push_u64(static_cast<uint64_t>(impl_->onnx_bytes_.size()));
  for (const uint8_t byte : impl_->onnx_bytes_) {
    push_u8(byte);
  }

  // CRC-32 trailer covering all preceding bytes.
  const uint32_t checksum = detail::crc32(std::as_bytes(std::span{buf}));
  push_u32(checksum);

  os.write(reinterpret_cast<const char *>(buf.data()), static_cast<std::streamsize>(buf.size()));
}

// ---------------------------------------------------------------------------
// Serialization — load
// ---------------------------------------------------------------------------

std::unique_ptr<OnnxModel> OnnxModel::load_from(std::istream &is) {
  // Layout: magic(4) + version(2) + ngram_record(34) + onnx_count(8) +
  //         onnx_bytes(N) + crc(4).
  // Read fixed prefix (6 + 34 = 40 bytes) into the CRC buffer upfront.
  constexpr size_t fixed_prefix = 6U + NGRAM_RECORD_BYTES;
  std::vector<uint8_t> buf(fixed_prefix);
  if (!is.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(fixed_prefix))) {
    throw std::runtime_error("OnnxModel::load_from: truncated header");
  }

  // Validate magic.
  if (buf[0] != LOX_MAGIC[0] || buf[1] != LOX_MAGIC[1] || buf[2] != LOX_MAGIC[2] ||
      buf[3] != LOX_MAGIC[3]) {
    throw std::runtime_error("OnnxModel::load_from: invalid magic bytes");
  }

  // Validate version.
  const auto version = static_cast<uint16_t>(buf[4]) | (static_cast<uint16_t>(buf[5]) << 8U);
  if (version != LOX_VERSION) {
    throw std::runtime_error("OnnxModel::load_from: unsupported version " +
                             std::to_string(version));
  }

  // Reconstruct NGramHasher from the 34 embedded bytes (buf[6..40)).
  // The substring may contain null bytes so we use the two-iterator constructor.
  const std::string hasher_raw(reinterpret_cast<const char *>(&buf[6]), NGRAM_RECORD_BYTES);
  std::istringstream hasher_ss{hasher_raw};
  NGramHasher hasher = NGramHasher::load(hasher_ss); // validates inner CRC

  // Read onnx_byte_count (8 bytes LE) and append to buf.
  std::array<uint8_t, 8> count_buf{};
  if (!is.read(reinterpret_cast<char *>(count_buf.data()), 8)) {
    throw std::runtime_error("OnnxModel::load_from: truncated onnx_byte_count");
  }
  for (const auto b : count_buf) {
    buf.push_back(b);
  }

  uint64_t onnx_byte_count = 0;
  for (int i = 0; i < 8; ++i) {
    onnx_byte_count |= static_cast<uint64_t>(count_buf[static_cast<size_t>(i)])
                       << (8U * static_cast<unsigned>(i));
  }

  // Read raw ONNX bytes and append to buf.
  const size_t pre_onnx = buf.size();
  buf.resize(pre_onnx + static_cast<size_t>(onnx_byte_count));
  if (!is.read(reinterpret_cast<char *>(buf.data() + pre_onnx),
               static_cast<std::streamsize>(onnx_byte_count))) {
    throw std::runtime_error("OnnxModel::load_from: truncated ONNX payload");
  }

  // Read and validate CRC-32.
  std::array<uint8_t, 4> crc_buf{};
  if (!is.read(reinterpret_cast<char *>(crc_buf.data()), 4)) {
    throw std::runtime_error("OnnxModel::load_from: truncated CRC");
  }
  const uint32_t stored_crc =
      static_cast<uint32_t>(crc_buf[0]) | (static_cast<uint32_t>(crc_buf[1]) << 8U) |
      (static_cast<uint32_t>(crc_buf[2]) << 16U) | (static_cast<uint32_t>(crc_buf[3]) << 24U);
  const uint32_t computed_crc = detail::crc32(std::as_bytes(std::span{buf}));
  if (computed_crc != stored_crc) {
    throw std::runtime_error("OnnxModel::load_from: CRC-32 mismatch — data may be corrupted");
  }

  // Build model from the accumulated buffer.
  auto impl = std::make_unique<Impl>();
  impl->hasher_ = hasher;
  impl->onnx_bytes_ =
      std::vector<uint8_t>{buf.begin() + static_cast<std::ptrdiff_t>(pre_onnx), buf.end()};
  impl->create_session();
  return std::unique_ptr<OnnxModel>{new OnnxModel{std::move(impl)}};
}

} // namespace lbf

#endif // LBF_ENABLE_ONNX
