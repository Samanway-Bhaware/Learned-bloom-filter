#include "lbf/model.hpp"

#include "lbf/models/logistic_regression.hpp"

#ifdef LBF_ENABLE_ONNX
#include "lbf/models/onnx_model.hpp"
#endif

#include <array>
#include <cstdint>
#include <stdexcept>

namespace lbf {

void MembershipModel::predict_batch(std::span<const std::span<const std::byte>> keys,
                                    std::span<double> out_scores) const {
  for (size_t i = 0; i < keys.size(); ++i) {
    out_scores[i] = predict(keys[i]);
  }
}

std::unique_ptr<MembershipModel> MembershipModel::load(std::istream &is) {
  // Peek at the 4-byte magic without consuming it.
  std::array<uint8_t, 4> magic{};
  if (!is.read(reinterpret_cast<char *>(magic.data()), 4)) {
    throw std::runtime_error("MembershipModel::load: truncated stream");
  }
  is.seekg(-4, std::ios::cur);

  // Dispatch to the matching subclass.
  if (magic[0] == 0x4CU && magic[1] == 0x4CU && magic[2] == 0x52U && magic[3] == 0x00U) {
    return std::make_unique<LogisticRegressionModel>(LogisticRegressionModel::load_from(is));
  }

#ifdef LBF_ENABLE_ONNX
  if (magic[0] == 0x4CU && magic[1] == 0x4FU && magic[2] == 0x58U && magic[3] == 0x00U) {
    return OnnxModel::load_from(is);
  }
#endif

  throw std::runtime_error("MembershipModel::load: unrecognised model magic bytes");
}

} // namespace lbf
