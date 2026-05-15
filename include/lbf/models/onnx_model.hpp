#pragma once

/// @file models/onnx_model.hpp
/// @brief ONNX Runtime inference backend for @ref MembershipModel.
///
/// This header is a no-op unless the CMake flag @c LBF_ENABLE_ONNX is @c ON.
/// When enabled, @ref OnnxModel wraps an ONNX Runtime session behind a PIMPL
/// so that public headers carry zero ORT dependencies.

#ifdef LBF_ENABLE_ONNX

#include "lbf/features/ngram_hasher.hpp"
#include "lbf/model.hpp"

#include <filesystem>
#include <istream>
#include <memory>
#include <ostream>
#include <span>

namespace lbf {

/// @brief Membership model backed by an ONNX Runtime inference session.
///
/// A pre-trained ONNX model (e.g. exported from PyTorch via
/// @c torch.onnx.export) is loaded and executed by the ORT C++ API.
/// Inputs are dense @c float32 feature vectors of size @c feature_dim_,
/// produced by an internal @ref NGramHasher. The model must output a single
/// scalar logit per sample; that logit is passed through a stable sigmoid to
/// produce the membership score.
///
/// @par PIMPL isolation
/// ORT headers (@c onnxruntime_cxx_api.h) are fully confined to the @ref Impl
/// translation unit. Consumers of this header have no transitive ORT dependency.
///
/// @par Serialization format
/// @c save()/@c load_from() store the raw ONNX model bytes inline alongside the
/// @ref NGramConfig, so the original @c .onnx file is not required after loading.
/// Binary layout (little-endian integers):
/// @code
///   Offset  Size    Field
///   ------  ------  -----
///      0      4     Magic: 0x4C 0x4F 0x58 0x00 ("LOX\0")
///      4      2     Version: uint16_t = 1
///      6      ?     NGramHasher sub-record (variable, self-described)
///      ?      8     onnx_byte_count: uint64_t
///      ?    onnx    Raw ONNX model bytes
///      ?      4     CRC-32 over all preceding bytes
/// @endcode
class OnnxModel : public MembershipModel {
public:
  // -------------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------------

  /// @brief Load an @ref OnnxModel from a @c .onnx file on disk.
  ///
  /// Reads and caches the model bytes, then creates an ORT inference session.
  /// The @p feature_config must exactly match the configuration used when the
  /// ONNX model was trained.
  ///
  /// @param onnx_path      Path to the @c .onnx file.
  /// @param feature_config N-gram config that produced training features.
  /// @throws std::runtime_error      If the file cannot be read or ORT session
  ///                                 creation fails.
  [[nodiscard]] static std::unique_ptr<OnnxModel>
  load_from_file(const std::filesystem::path &onnx_path, NGramConfig feature_config);

  // -------------------------------------------------------------------------
  // Prediction
  // -------------------------------------------------------------------------

  /// @brief Predict membership probability for a single @p key.
  ///
  /// Extracts n-gram features, assembles a dense @c float32 input tensor,
  /// runs ORT inference, and applies a stable sigmoid to the scalar logit.
  ///
  /// @return Score in @c [0, 1].
  [[nodiscard]] double predict(std::span<const std::byte> key) const noexcept override;

  // -------------------------------------------------------------------------
  // Serialization
  // -------------------------------------------------------------------------

  /// @brief Serialize this model (NGramConfig + raw ONNX bytes) to @p os.
  /// @throws std::ios_base::failure  On write error (if exceptions enabled).
  void save(std::ostream &os) const override;

  /// @brief Deserialize from @p is. Invoked by @ref MembershipModel::load().
  ///
  /// Reconstructs the ORT session from the embedded ONNX bytes.
  ///
  /// @throws std::runtime_error      On magic/version mismatch, CRC failure,
  ///                                 or ORT session creation failure.
  /// @throws std::ios_base::failure  On read error.
  [[nodiscard]] static std::unique_ptr<OnnxModel> load_from(std::istream &is);

private:
  /// @brief PIMPL implementation type — defined only in onnx_model.cpp.
  struct Impl;

  std::unique_ptr<Impl> impl_;

  explicit OnnxModel(std::unique_ptr<Impl> impl);
};

} // namespace lbf

#endif // LBF_ENABLE_ONNX
