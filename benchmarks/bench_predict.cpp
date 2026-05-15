/// @file bench_predict.cpp
/// @brief Micro-benchmarks for the model-layer hot paths.
///
/// Targets (Phase 3 Step 5):
///   - NGramHasher::hash()         — feature extraction
///   - LogisticRegressionModel::predict() — full inference path
///
/// Target: predict() < 500 ns / key on a modern desktop CPU.
///
/// Build:   cmake -B build -DLBF_BUILD_BENCHMARKS=ON && cmake --build build
/// Run:     ./build/benchmarks/bench_predict
/// Filter:  ./build/benchmarks/bench_predict --benchmark_filter=Predict

#include "lbf/features/ngram_hasher.hpp"
#include "lbf/models/logistic_regression.hpp"

#include <benchmark/benchmark.h>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

// Representative URL-like key (~50 bytes) to exercise multiple n-gram windows.
const std::string kSampleKey = "https://example.com/user/profile/settings?tab=security";

constexpr size_t kFeatureDim = 65536;

lbf::NGramConfig default_config() {
  return lbf::NGramConfig{.min_n_ = 3, .max_n_ = 5, .feature_dim_ = kFeatureDim};
}

std::span<const std::byte> key_bytes(const std::string &s) {
  return std::as_bytes(std::span<const char>{s.data(), s.size()});
}

} // namespace

// ---------------------------------------------------------------------------
// BM_NGramHasherHash — measures feature-extraction throughput
// ---------------------------------------------------------------------------

static void BM_NGramHasherHash(benchmark::State &state) {
  const lbf::NGramHasher hasher{default_config()};
  const auto bytes = key_bytes(kSampleKey);
  std::vector<std::pair<uint32_t, float>> out;
  out.reserve(256);

  for (auto _ : state) {
    hasher.hash(bytes, out);
    benchmark::DoNotOptimize(out); // non-const lvalue — avoids deprecated const-ref overload
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
  state.SetLabel("key_len=" + std::to_string(kSampleKey.size()) + "B");
}
BENCHMARK(BM_NGramHasherHash)->MinTime(0.5);

// ---------------------------------------------------------------------------
// BM_LogisticRegressionPredict — measures full end-to-end inference
// ---------------------------------------------------------------------------

static void BM_LogisticRegressionPredict(benchmark::State &state) {
  // Non-zero weights exercise the sparse dot product, not just hash().
  std::vector<float> weights(kFeatureDim, 0.01F);
  const lbf::LogisticRegressionModel model{lbf::NGramHasher{default_config()}, std::move(weights),
                                           0.0F};
  const auto bytes = key_bytes(kSampleKey);

  for (auto _ : state) {
    double score = model.predict(bytes); // non-const: avoids deprecated const-ref DoNotOptimize
    benchmark::DoNotOptimize(score);
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
  state.SetLabel("dim=" + std::to_string(kFeatureDim));
}
BENCHMARK(BM_LogisticRegressionPredict)->MinTime(0.5);
