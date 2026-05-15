/// @file bench_learned_bf.cpp
/// @brief Comprehensive LearnedBloomFilter benchmarks (Phase 5).
///
/// Measures:
///   - Training time (reported as a counter, not in the timed loop)
///   - Query throughput (member hit path + non-member miss path)
///   - False-positive rate with Wilson 95 % CI
///   - Per-operation latency percentiles (p50/p90/p99/p999)
///   - Memory breakdown (model weights + backup filter)
///   - Backup count (members not covered by the model)
///
/// For each dataset the benchmark directly mirrors bench_classical_bf so
/// the Python plotter can compare the two side-by-side.
///
/// Note on training: model training happens in benchmark *setup* (before the
/// iteration loop), so only inference is timed.  Training time is reported
/// separately via state.counters["lbf_train_ms"].
///
/// Run:
///   LBF_BENCH_DATASETS_DIR=benchmarks/datasets \
///     ./build/release/benchmarks/bench_learned_bf \
///     --benchmark_out=results/learned_bf.json \
///     --benchmark_out_format=json \
///     --benchmark_repetitions=5
///
/// Step 2: uniform_int dataset only (n=100k to keep training < 5 s).

#include "bench_dataset.hpp"

#include "lbf/learned_bloom_filter.hpp"
#include "lbf/models/logistic_regression.hpp"

#include <benchmark/benchmark.h>
#include <chrono>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Per-benchmark configuration
// ---------------------------------------------------------------------------

struct LbfCfg {
  const char *dataset_file;
  const char *dataset_name;
  size_t n_members;    // member set size
  size_t n_train_neg;  // non-members used for training (supervised contrast)
  double fpr_target;   // backup filter FPR target
  double threshold;    // model decision threshold (0.5 = MAP)
  // NGram / training hyper-params.
  size_t feature_dim;
  size_t epochs;
  bool is_fixture = false; // true → benchmarks/fixtures/, false → benchmarks/datasets/
};

// ---------------------------------------------------------------------------
// Helper: build byte-span views over a stable string vector.
// ---------------------------------------------------------------------------

std::vector<std::span<const std::byte>>
make_spans(const std::vector<std::string> &strings) {
  std::vector<std::span<const std::byte>> spans;
  spans.reserve(strings.size());
  for (const auto &s : strings) {
    spans.push_back(
        std::as_bytes(std::span<const char>{s.data(), s.size()}));
  }
  return spans;
}

// ---------------------------------------------------------------------------
// Miss-path (non-member) query benchmark
// ---------------------------------------------------------------------------

static void BM_LearnedBF_Contains_Miss(benchmark::State &state,
                                       LbfCfg cfg) {
  const std::string path = bench::resolve_path(cfg.dataset_file, cfg.is_fixture);
  const auto ds = bench::load_tsv(path, cfg.dataset_name, cfg.n_members,
                                  cfg.n_train_neg + 100'000);

  if (!ds.ok()) {
    state.SkipWithError(("dataset not found: " + path).c_str());
    return;
  }

  // --- Training + LBF construction (not timed) -----------------------------
  const size_t n_mem = std::min(cfg.n_members, ds.members.size());
  const size_t n_neg = std::min(cfg.n_train_neg, ds.nonmembers.size());

  std::vector<std::string> train_strings;
  train_strings.reserve(n_mem + n_neg);
  std::vector<uint8_t> labels;
  labels.reserve(n_mem + n_neg);
  for (size_t i = 0; i < n_mem; ++i) {
    train_strings.push_back(ds.members[i]);
    labels.push_back(uint8_t{1});
  }
  for (size_t i = 0; i < n_neg; ++i) {
    train_strings.push_back(ds.nonmembers[i]);
    labels.push_back(uint8_t{0});
  }
  const auto train_spans = make_spans(train_strings);

  const lbf::NGramConfig ngram{.min_n_ = 3,
                                .max_n_ = 5,
                                .feature_dim_ = cfg.feature_dim};
  const lbf::TrainingConfig train_cfg{.learning_rate_ = 0.1,
                                      .l2_regularization_ = 1e-4,
                                      .epochs_ = cfg.epochs,
                                      .batch_size_ = 256,
                                      .momentum_ = 0.9,
                                      .random_seed_ = 42,
                                      .verbose_ = false};

  const auto t0 = std::chrono::steady_clock::now();
  auto result = lbf::LogisticRegressionModel::train(
      std::span<const std::span<const std::byte>>{train_spans},
      std::span<const uint8_t>{labels}, ngram, train_cfg);
  const auto t1 = std::chrono::steady_clock::now();
  const double train_ms = static_cast<double>(
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

  const size_t model_mem = result.first.memory_bytes();
  auto model_ptr =
      std::make_unique<lbf::LogisticRegressionModel>(std::move(result.first));

  // Member spans for backup-filter construction.
  std::vector<std::string> member_strs(
      ds.members.begin(),
      ds.members.begin() + static_cast<std::ptrdiff_t>(n_mem));
  const auto member_spans = make_spans(member_strs);

  auto lbf = lbf::LearnedBloomFilter::build(
      std::move(model_ptr),
      std::span<const std::span<const std::byte>>{member_spans},
      cfg.fpr_target, cfg.threshold);

  // Non-member probe pool (exclude those used for training).
  const size_t probe_start = n_neg;
  std::vector<std::string> probes(
      ds.nonmembers.begin() +
          static_cast<std::ptrdiff_t>(probe_start),
      ds.nonmembers.end());

  if (probes.empty()) {
    state.SkipWithError("no held-out non-members available for probing");
    return;
  }

  // ----- Setup measurements (not timed) ------------------------------------

  const auto fpr = bench::measure_fpr(
      [&](std::string_view k) { return lbf.contains(k); }, probes);

  const auto pct = bench::measure_percentiles(
      [&](std::string_view k) { return lbf.contains(k); }, probes);

  // ----- Counters ----------------------------------------------------------
  // fastpath_frac: fraction of member queries answered by the model alone
  // (no backup filter hit required).  = (n_members - backup_count) / n_members.
  const double fastpath_frac = (n_mem > 0)
      ? static_cast<double>(n_mem - lbf.backup_count()) / static_cast<double>(n_mem)
      : 0.0;

  state.counters["lbf_n_members"] = static_cast<double>(n_mem);
  state.counters["lbf_fpr_target"] = cfg.fpr_target;
  state.counters["lbf_threshold"] = cfg.threshold;
  state.counters["lbf_backup_count"] =
      static_cast<double>(lbf.backup_count());
  state.counters["lbf_fastpath_frac"] = fastpath_frac;
  state.counters["lbf_model_memory_bytes"] =
      static_cast<double>(model_mem);
  state.counters["lbf_backup_memory_bytes"] =
      static_cast<double>(lbf.memory_bytes());
  state.counters["lbf_total_memory_bytes"] =
      static_cast<double>(model_mem + lbf.memory_bytes());
  state.counters["lbf_train_ms"] = train_ms;
  bench::emit_fpr(state, fpr);
  bench::emit_pct(state, pct);

  // ----- Timed loop: steady-state miss-path queries ------------------------
  size_t idx = 0;
  const size_t np = probes.size();
  for (auto _ : state) {
    bool r = lbf.contains(std::string_view{probes[idx % np]});
    benchmark::DoNotOptimize(r);
    ++idx;
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

// ---------------------------------------------------------------------------
// Hit-path (member) query benchmark
// ---------------------------------------------------------------------------

static void BM_LearnedBF_Contains_Hit(benchmark::State &state, LbfCfg cfg) {
  const std::string path = bench::resolve_path(cfg.dataset_file, cfg.is_fixture);
  const auto ds = bench::load_tsv(path, cfg.dataset_name, cfg.n_members,
                                  cfg.n_train_neg);

  if (!ds.ok()) {
    state.SkipWithError(("dataset not found: " + path).c_str());
    return;
  }

  const size_t n_mem = std::min(cfg.n_members, ds.members.size());
  const size_t n_neg = std::min(cfg.n_train_neg, ds.nonmembers.size());

  std::vector<std::string> train_strings;
  train_strings.reserve(n_mem + n_neg);
  std::vector<uint8_t> labels;
  labels.reserve(n_mem + n_neg);
  for (size_t i = 0; i < n_mem; ++i) {
    train_strings.push_back(ds.members[i]);
    labels.push_back(uint8_t{1});
  }
  for (size_t i = 0; i < n_neg; ++i) {
    train_strings.push_back(ds.nonmembers[i]);
    labels.push_back(uint8_t{0});
  }
  const auto train_spans = make_spans(train_strings);

  const lbf::NGramConfig ngram{.min_n_ = 3,
                                .max_n_ = 5,
                                .feature_dim_ = cfg.feature_dim};
  const lbf::TrainingConfig train_cfg{.learning_rate_ = 0.1,
                                      .l2_regularization_ = 1e-4,
                                      .epochs_ = cfg.epochs,
                                      .batch_size_ = 256,
                                      .momentum_ = 0.9,
                                      .random_seed_ = 42,
                                      .verbose_ = false};

  auto result = lbf::LogisticRegressionModel::train(
      std::span<const std::span<const std::byte>>{train_spans},
      std::span<const uint8_t>{labels}, ngram, train_cfg);

  const size_t model_mem = result.first.memory_bytes();
  auto model_ptr =
      std::make_unique<lbf::LogisticRegressionModel>(std::move(result.first));

  std::vector<std::string> member_strs(
      ds.members.begin(),
      ds.members.begin() + static_cast<std::ptrdiff_t>(n_mem));
  const auto member_spans = make_spans(member_strs);

  auto lbf = lbf::LearnedBloomFilter::build(
      std::move(model_ptr),
      std::span<const std::span<const std::byte>>{member_spans},
      cfg.fpr_target, cfg.threshold);

  // Percentiles on the hit path.
  const auto pct = bench::measure_percentiles(
      [&](std::string_view k) { return lbf.contains(k); }, member_strs);

  const double fastpath_frac = (n_mem > 0)
      ? static_cast<double>(n_mem - lbf.backup_count()) / static_cast<double>(n_mem)
      : 0.0;

  state.counters["lbf_n_members"] = static_cast<double>(n_mem);
  state.counters["lbf_threshold"] = cfg.threshold;
  state.counters["lbf_backup_count"] =
      static_cast<double>(lbf.backup_count());
  state.counters["lbf_fastpath_frac"] = fastpath_frac;
  state.counters["lbf_model_memory_bytes"] =
      static_cast<double>(model_mem);
  state.counters["lbf_backup_memory_bytes"] =
      static_cast<double>(lbf.memory_bytes());
  state.counters["lbf_total_memory_bytes"] =
      static_cast<double>(model_mem + lbf.memory_bytes());
  bench::emit_pct(state, pct);

  size_t idx = 0;
  for (auto _ : state) {
    bool r = lbf.contains(std::string_view{member_strs[idx % n_mem]});
    benchmark::DoNotOptimize(r);
    ++idx;
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

// ---------------------------------------------------------------------------
// Dataset: uniform random 64-bit integers
// ---------------------------------------------------------------------------
//
// n=100k only for LBF: training on 100k takes ~1-3 s (acceptable).
// For n=1M with this model, training would take ~30-60 s which is too slow
// for benchmark setup; LBF on large uniform-int is a known losing case anyway.

constexpr LbfCfg UNIFORM_100K = {
    "uniform_int.tsv", "uniform_int",
    /*n_members=*/100'000, /*n_train_neg=*/100'000,
    /*fpr_target=*/0.01, /*threshold=*/0.5,
    /*feature_dim=*/4096, /*epochs=*/10};

// ---------------------------------------------------------------------------
// Dataset: English words fixture  (real words vs matched-length random strings)
// ---------------------------------------------------------------------------
//
// AUC(sklearn proxy) = 0.9959.  Generate: bash benchmarks/fixtures/gen_words_en.sh
// Validate: python3 scripts/validate_auc.py
//
// Three N values × three τ values = 9 configs.
// n_train_neg is set equal to n_members; the remainder forms the held-out
// probe pool (200k total negatives in the fixture).

// --- N = 10k ---
constexpr LbfCfg WORDS_EN_10K_T03 = {
    .dataset_file = "words_en.tsv", .dataset_name = "words_en",
    .n_members = 10'000, .n_train_neg = 10'000,
    .fpr_target = 0.01, .threshold = 0.3,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr LbfCfg WORDS_EN_10K_T05 = {
    .dataset_file = "words_en.tsv", .dataset_name = "words_en",
    .n_members = 10'000, .n_train_neg = 10'000,
    .fpr_target = 0.01, .threshold = 0.5,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr LbfCfg WORDS_EN_10K_T07 = {
    .dataset_file = "words_en.tsv", .dataset_name = "words_en",
    .n_members = 10'000, .n_train_neg = 10'000,
    .fpr_target = 0.01, .threshold = 0.7,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};

// --- N = 50k ---
constexpr LbfCfg WORDS_EN_50K_T03 = {
    .dataset_file = "words_en.tsv", .dataset_name = "words_en",
    .n_members = 50'000, .n_train_neg = 50'000,
    .fpr_target = 0.01, .threshold = 0.3,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr LbfCfg WORDS_EN_50K_T05 = {
    .dataset_file = "words_en.tsv", .dataset_name = "words_en",
    .n_members = 50'000, .n_train_neg = 50'000,
    .fpr_target = 0.01, .threshold = 0.5,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr LbfCfg WORDS_EN_50K_T07 = {
    .dataset_file = "words_en.tsv", .dataset_name = "words_en",
    .n_members = 50'000, .n_train_neg = 50'000,
    .fpr_target = 0.01, .threshold = 0.7,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};

// --- N = 100k ---
constexpr LbfCfg WORDS_EN_100K_T03 = {
    .dataset_file = "words_en.tsv", .dataset_name = "words_en",
    .n_members = 100'000, .n_train_neg = 100'000,
    .fpr_target = 0.01, .threshold = 0.3,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr LbfCfg WORDS_EN_100K_T05 = {
    .dataset_file = "words_en.tsv", .dataset_name = "words_en",
    .n_members = 100'000, .n_train_neg = 100'000,
    .fpr_target = 0.01, .threshold = 0.5,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr LbfCfg WORDS_EN_100K_T07 = {
    .dataset_file = "words_en.tsv", .dataset_name = "words_en",
    .n_members = 100'000, .n_train_neg = 100'000,
    .fpr_target = 0.01, .threshold = 0.7,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};

// ---------------------------------------------------------------------------
// SGD convergence experiment: epochs=50/100, τ=0.7, N=100k
// Not in the main N×τ sweep; used to validate the "would likely close" claim.
// ---------------------------------------------------------------------------

constexpr LbfCfg WORDS_EN_100K_T07_E50 = {
    .dataset_file = "words_en.tsv", .dataset_name = "words_en",
    .n_members = 100'000, .n_train_neg = 100'000,
    .fpr_target = 0.01, .threshold = 0.7,
    .feature_dim = 16384, .epochs = 50, .is_fixture = true};

constexpr LbfCfg WORDS_EN_100K_T07_E100 = {
    .dataset_file = "words_en.tsv", .dataset_name = "words_en",
    .n_members = 100'000, .n_train_neg = 100'000,
    .fpr_target = 0.01, .threshold = 0.7,
    .feature_dim = 16384, .epochs = 100, .is_fixture = true};

} // namespace

BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, uniform_100k, UNIFORM_100K)
    ->MinTime(0.5)
    ->Repetitions(5)
    ->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Hit, uniform_100k, UNIFORM_100K)
    ->MinTime(0.5)
    ->Repetitions(5)
    ->ReportAggregatesOnly(false);

// ---------------------------------------------------------------------------
// English words fixture benchmarks — N × τ sweep
// ---------------------------------------------------------------------------
// Miss path (FPR, Wilson CI, latency, memory, fastpath_frac)

BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, words_en_10k_t03, WORDS_EN_10K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, words_en_10k_t05, WORDS_EN_10K_T05)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, words_en_10k_t07, WORDS_EN_10K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, words_en_50k_t03, WORDS_EN_50K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, words_en_50k_t05, WORDS_EN_50K_T05)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, words_en_50k_t07, WORDS_EN_50K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, words_en_100k_t03, WORDS_EN_100K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, words_en_100k_t05, WORDS_EN_100K_T05)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, words_en_100k_t07, WORDS_EN_100K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

// Hit path (latency, fastpath_frac)
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Hit, words_en_10k_t03, WORDS_EN_10K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Hit, words_en_10k_t05, WORDS_EN_10K_T05)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Hit, words_en_10k_t07, WORDS_EN_10K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Hit, words_en_50k_t03, WORDS_EN_50K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Hit, words_en_50k_t05, WORDS_EN_50K_T05)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Hit, words_en_50k_t07, WORDS_EN_50K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Hit, words_en_100k_t03, WORDS_EN_100K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Hit, words_en_100k_t05, WORDS_EN_100K_T05)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Hit, words_en_100k_t07, WORDS_EN_100K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

// SGD convergence experiment (Miss path only; 3 reps to keep training time reasonable)
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, words_en_100k_t07_e50, WORDS_EN_100K_T07_E50)
    ->MinTime(0.1)->Repetitions(3)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, words_en_100k_t07_e100, WORDS_EN_100K_T07_E100)
    ->MinTime(0.1)->Repetitions(3)->ReportAggregatesOnly(false);

// ---------------------------------------------------------------------------
// Dataset: Tranco URL fixture — N × τ sweep
// ---------------------------------------------------------------------------
// AUC(sklearn proxy) = 0.9991. Generate: bash benchmarks/fixtures/gen_urls_tranco.sh

constexpr LbfCfg URLS_TRANCO_10K_T03 = {
    .dataset_file = "urls_tranco.tsv", .dataset_name = "urls_tranco",
    .n_members = 10'000, .n_train_neg = 10'000,
    .fpr_target = 0.01, .threshold = 0.3,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr LbfCfg URLS_TRANCO_10K_T05 = {
    .dataset_file = "urls_tranco.tsv", .dataset_name = "urls_tranco",
    .n_members = 10'000, .n_train_neg = 10'000,
    .fpr_target = 0.01, .threshold = 0.5,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr LbfCfg URLS_TRANCO_10K_T07 = {
    .dataset_file = "urls_tranco.tsv", .dataset_name = "urls_tranco",
    .n_members = 10'000, .n_train_neg = 10'000,
    .fpr_target = 0.01, .threshold = 0.7,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr LbfCfg URLS_TRANCO_50K_T03 = {
    .dataset_file = "urls_tranco.tsv", .dataset_name = "urls_tranco",
    .n_members = 50'000, .n_train_neg = 50'000,
    .fpr_target = 0.01, .threshold = 0.3,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr LbfCfg URLS_TRANCO_50K_T05 = {
    .dataset_file = "urls_tranco.tsv", .dataset_name = "urls_tranco",
    .n_members = 50'000, .n_train_neg = 50'000,
    .fpr_target = 0.01, .threshold = 0.5,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr LbfCfg URLS_TRANCO_50K_T07 = {
    .dataset_file = "urls_tranco.tsv", .dataset_name = "urls_tranco",
    .n_members = 50'000, .n_train_neg = 50'000,
    .fpr_target = 0.01, .threshold = 0.7,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr LbfCfg URLS_TRANCO_100K_T03 = {
    .dataset_file = "urls_tranco.tsv", .dataset_name = "urls_tranco",
    .n_members = 100'000, .n_train_neg = 100'000,
    .fpr_target = 0.01, .threshold = 0.3,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr LbfCfg URLS_TRANCO_100K_T05 = {
    .dataset_file = "urls_tranco.tsv", .dataset_name = "urls_tranco",
    .n_members = 100'000, .n_train_neg = 100'000,
    .fpr_target = 0.01, .threshold = 0.5,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr LbfCfg URLS_TRANCO_100K_T07 = {
    .dataset_file = "urls_tranco.tsv", .dataset_name = "urls_tranco",
    .n_members = 100'000, .n_train_neg = 100'000,
    .fpr_target = 0.01, .threshold = 0.7,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};

BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, urls_tranco_10k_t03, URLS_TRANCO_10K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, urls_tranco_10k_t05, URLS_TRANCO_10K_T05)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, urls_tranco_10k_t07, URLS_TRANCO_10K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, urls_tranco_50k_t03, URLS_TRANCO_50K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, urls_tranco_50k_t05, URLS_TRANCO_50K_T05)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, urls_tranco_50k_t07, URLS_TRANCO_50K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, urls_tranco_100k_t03, URLS_TRANCO_100K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, urls_tranco_100k_t05, URLS_TRANCO_100K_T05)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Miss, urls_tranco_100k_t07, URLS_TRANCO_100K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Hit, urls_tranco_10k_t03, URLS_TRANCO_10K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Hit, urls_tranco_10k_t05, URLS_TRANCO_10K_T05)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Hit, urls_tranco_10k_t07, URLS_TRANCO_10K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Hit, urls_tranco_50k_t03, URLS_TRANCO_50K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Hit, urls_tranco_50k_t05, URLS_TRANCO_50K_T05)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Hit, urls_tranco_50k_t07, URLS_TRANCO_50K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Hit, urls_tranco_100k_t03, URLS_TRANCO_100K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Hit, urls_tranco_100k_t05, URLS_TRANCO_100K_T05)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_LearnedBF_Contains_Hit, urls_tranco_100k_t07, URLS_TRANCO_100K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
