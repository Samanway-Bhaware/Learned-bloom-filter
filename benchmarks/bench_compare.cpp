/// @file bench_compare.cpp
/// @brief Head-to-head: ClassicalBF vs LearnedBloomFilter on real datasets.
///
/// Both filters are built on the same member set and probed against the same
/// held-out non-members, so every metric is directly comparable.
///
/// Step 3 / corrected dataset: English words fixture
///   Positives = /usr/share/dict/words  (real words)
///   Negatives = matched-length random alphabetic strings
///   AUC(sklearn proxy) = 0.9959
///   Generate: bash benchmarks/fixtures/gen_words_en.sh
///   Validate: python3 scripts/validate_auc.py
///
/// Sweep: N ∈ {10k, 50k, 100k} × τ ∈ {0.3, 0.5, 0.7}.
///
/// Key counters reported for the Pareto comparison:
///   lbf_fpr_measured          — actual FPR (Wilson CI also emitted)
///   lbf_memory_bytes          — classical BF memory
///   lbf_total_memory_bytes    — LBF total (model + backup)
///   lbf_fastpath_frac         — (n_mem - backup_count) / n_mem
///   lbf_p50_ns / lbf_p99_ns  — latency percentiles
///
/// Run:
///   bash benchmarks/fixtures/gen_words_en.sh
///   ./build/release/benchmarks/bench_compare \
///     --benchmark_filter=words_en \
///     --benchmark_out=results/compare.json \
///     --benchmark_out_format=json \
///     --benchmark_repetitions=5

#include "bench_dataset.hpp"

#include "lbf/bloom_filter.hpp"
#include "lbf/learned_bloom_filter.hpp"
#include "lbf/models/logistic_regression.hpp"

#include <benchmark/benchmark.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Shared configuration — superset of BenchCfg and LbfCfg
// ---------------------------------------------------------------------------

struct CompCfg {
  const char *dataset_file;
  const char *dataset_name;
  size_t n_members;
  double fpr_target;
  // LBF extras
  size_t n_train_neg;
  double threshold;
  size_t feature_dim;
  size_t epochs;
  bool is_fixture = false;
};

// ---------------------------------------------------------------------------
// Helper: byte-span views over a stable string vector
// ---------------------------------------------------------------------------

std::vector<std::span<const std::byte>>
make_byte_spans(const std::vector<std::string> &strings) {
  std::vector<std::span<const std::byte>> spans;
  spans.reserve(strings.size());
  for (const auto &s : strings) {
    spans.push_back(std::as_bytes(std::span<const char>{s.data(), s.size()}));
  }
  return spans;
}

// ---------------------------------------------------------------------------
// Classical BF — miss path (non-members)
// ---------------------------------------------------------------------------

static void BM_Compare_Classical_Miss(benchmark::State &state, CompCfg cfg) {
  const std::string path = bench::resolve_path(cfg.dataset_file, cfg.is_fixture);
  const auto ds = bench::load_tsv(path, cfg.dataset_name, cfg.n_members,
                                  /*max_nm=*/200'000);
  if (!ds.ok()) {
    const std::string hint = cfg.is_fixture
        ? "\n  Run: bash benchmarks/fixtures/gen_words_en.sh"
        : "";
    state.SkipWithError(("dataset not found: " + path + hint).c_str());
    return;
  }

  const size_t n = std::min(cfg.n_members, ds.members.size());
  lbf::BloomFilter<> bf{
      lbf::BloomFilterParams{.expected_items_ = n, .target_fpr_ = cfg.fpr_target}};
  for (size_t i = 0; i < n; ++i) {
    bf.insert(std::string_view{ds.members[i]});
  }

  const auto fpr = bench::measure_fpr(
      [&](std::string_view k) { return bf.contains(k); }, ds.nonmembers);
  const auto pct = bench::measure_percentiles(
      [&](std::string_view k) { return bf.contains(k); }, ds.nonmembers);

  state.counters["lbf_n_members"] = static_cast<double>(n);
  state.counters["lbf_fpr_target"] = cfg.fpr_target;
  state.counters["lbf_memory_bytes"] =
      static_cast<double>(bf.bit_count() / 8);
  bench::emit_fpr(state, fpr);
  bench::emit_pct(state, pct);

  size_t idx = 0;
  const size_t nm = ds.nonmembers.size();
  for (auto _ : state) {
    bool r = bf.contains(std::string_view{ds.nonmembers[idx % nm]});
    benchmark::DoNotOptimize(r);
    ++idx;
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

// ---------------------------------------------------------------------------
// Classical BF — hit path (members)
// ---------------------------------------------------------------------------

static void BM_Compare_Classical_Hit(benchmark::State &state, CompCfg cfg) {
  const std::string path = bench::resolve_path(cfg.dataset_file, cfg.is_fixture);
  const auto ds =
      bench::load_tsv(path, cfg.dataset_name, cfg.n_members, /*max_nm=*/0);
  if (!ds.ok()) {
    state.SkipWithError(("dataset not found: " + path).c_str());
    return;
  }

  const size_t n = std::min(cfg.n_members, ds.members.size());
  lbf::BloomFilter<> bf{
      lbf::BloomFilterParams{.expected_items_ = n, .target_fpr_ = cfg.fpr_target}};
  for (size_t i = 0; i < n; ++i) {
    bf.insert(std::string_view{ds.members[i]});
  }

  const auto pct = bench::measure_percentiles(
      [&](std::string_view k) { return bf.contains(k); }, ds.members);

  state.counters["lbf_n_members"] = static_cast<double>(n);
  state.counters["lbf_memory_bytes"] =
      static_cast<double>(bf.bit_count() / 8);
  bench::emit_pct(state, pct);

  size_t idx = 0;
  for (auto _ : state) {
    bool r = bf.contains(std::string_view{ds.members[idx % n]});
    benchmark::DoNotOptimize(r);
    ++idx;
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

// ---------------------------------------------------------------------------
// Learned BF — miss path (non-members)
// ---------------------------------------------------------------------------

static void BM_Compare_Learned_Miss(benchmark::State &state, CompCfg cfg) {
  const std::string path = bench::resolve_path(cfg.dataset_file, cfg.is_fixture);
  const auto ds = bench::load_tsv(path, cfg.dataset_name, cfg.n_members,
                                  cfg.n_train_neg + 100'000);
  if (!ds.ok()) {
    const std::string hint = cfg.is_fixture
        ? "\n  Run: bash benchmarks/fixtures/gen_words_en.sh"
        : "";
    state.SkipWithError(("dataset not found: " + path + hint).c_str());
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
  const auto train_spans = make_byte_spans(train_strings);

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

  std::vector<std::string> member_strs(
      ds.members.begin(),
      ds.members.begin() + static_cast<std::ptrdiff_t>(n_mem));
  const auto member_spans = make_byte_spans(member_strs);

  auto lbf_inst = lbf::LearnedBloomFilter::build(
      std::move(model_ptr),
      std::span<const std::span<const std::byte>>{member_spans},
      cfg.fpr_target, cfg.threshold);

  // Held-out probes: non-members not used in training.
  std::vector<std::string> probes(
      ds.nonmembers.begin() + static_cast<std::ptrdiff_t>(n_neg),
      ds.nonmembers.end());
  if (probes.empty()) {
    state.SkipWithError("no held-out non-members available for probing");
    return;
  }

  const auto fpr = bench::measure_fpr(
      [&](std::string_view k) { return lbf_inst.contains(k); }, probes);
  const auto pct = bench::measure_percentiles(
      [&](std::string_view k) { return lbf_inst.contains(k); }, probes);

  const double fastpath_frac = (n_mem > 0)
      ? static_cast<double>(n_mem - lbf_inst.backup_count()) /
            static_cast<double>(n_mem)
      : 0.0;

  state.counters["lbf_n_members"] = static_cast<double>(n_mem);
  state.counters["lbf_fpr_target"] = cfg.fpr_target;
  state.counters["lbf_threshold"] = cfg.threshold;
  state.counters["lbf_backup_count"] =
      static_cast<double>(lbf_inst.backup_count());
  state.counters["lbf_fastpath_frac"] = fastpath_frac;
  state.counters["lbf_model_memory_bytes"] = static_cast<double>(model_mem);
  state.counters["lbf_backup_memory_bytes"] =
      static_cast<double>(lbf_inst.memory_bytes());
  state.counters["lbf_total_memory_bytes"] =
      static_cast<double>(model_mem + lbf_inst.memory_bytes());
  state.counters["lbf_train_ms"] = train_ms;
  bench::emit_fpr(state, fpr);
  bench::emit_pct(state, pct);

  size_t idx = 0;
  const size_t np = probes.size();
  for (auto _ : state) {
    bool r = lbf_inst.contains(std::string_view{probes[idx % np]});
    benchmark::DoNotOptimize(r);
    ++idx;
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

// ---------------------------------------------------------------------------
// Learned BF — hit path (members)
// ---------------------------------------------------------------------------

static void BM_Compare_Learned_Hit(benchmark::State &state, CompCfg cfg) {
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
  const auto train_spans = make_byte_spans(train_strings);

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
  const auto member_spans = make_byte_spans(member_strs);

  auto lbf_inst = lbf::LearnedBloomFilter::build(
      std::move(model_ptr),
      std::span<const std::span<const std::byte>>{member_spans},
      cfg.fpr_target, cfg.threshold);

  const auto pct = bench::measure_percentiles(
      [&](std::string_view k) { return lbf_inst.contains(k); }, member_strs);

  const double fastpath_frac = (n_mem > 0)
      ? static_cast<double>(n_mem - lbf_inst.backup_count()) /
            static_cast<double>(n_mem)
      : 0.0;

  state.counters["lbf_n_members"] = static_cast<double>(n_mem);
  state.counters["lbf_threshold"] = cfg.threshold;
  state.counters["lbf_backup_count"] =
      static_cast<double>(lbf_inst.backup_count());
  state.counters["lbf_fastpath_frac"] = fastpath_frac;
  state.counters["lbf_model_memory_bytes"] = static_cast<double>(model_mem);
  state.counters["lbf_backup_memory_bytes"] =
      static_cast<double>(lbf_inst.memory_bytes());
  state.counters["lbf_total_memory_bytes"] =
      static_cast<double>(model_mem + lbf_inst.memory_bytes());
  bench::emit_pct(state, pct);

  size_t idx = 0;
  for (auto _ : state) {
    bool r = lbf_inst.contains(std::string_view{member_strs[idx % n_mem]});
    benchmark::DoNotOptimize(r);
    ++idx;
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

// ---------------------------------------------------------------------------
// Dataset configurations — English words fixture
// ---------------------------------------------------------------------------

// --- N = 10k ---
constexpr CompCfg WORDS_EN_10K = {
    .dataset_file = "words_en.tsv", .dataset_name = "words_en",
    .n_members = 10'000, .fpr_target = 0.01,
    .n_train_neg = 10'000, .threshold = 0.5,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr CompCfg WORDS_EN_10K_T03 = {
    .dataset_file = "words_en.tsv", .dataset_name = "words_en",
    .n_members = 10'000, .fpr_target = 0.01,
    .n_train_neg = 10'000, .threshold = 0.3,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr CompCfg WORDS_EN_10K_T07 = {
    .dataset_file = "words_en.tsv", .dataset_name = "words_en",
    .n_members = 10'000, .fpr_target = 0.01,
    .n_train_neg = 10'000, .threshold = 0.7,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};

// --- N = 50k ---
constexpr CompCfg WORDS_EN_50K = {
    .dataset_file = "words_en.tsv", .dataset_name = "words_en",
    .n_members = 50'000, .fpr_target = 0.01,
    .n_train_neg = 50'000, .threshold = 0.5,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr CompCfg WORDS_EN_50K_T03 = {
    .dataset_file = "words_en.tsv", .dataset_name = "words_en",
    .n_members = 50'000, .fpr_target = 0.01,
    .n_train_neg = 50'000, .threshold = 0.3,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr CompCfg WORDS_EN_50K_T07 = {
    .dataset_file = "words_en.tsv", .dataset_name = "words_en",
    .n_members = 50'000, .fpr_target = 0.01,
    .n_train_neg = 50'000, .threshold = 0.7,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};

// --- N = 100k ---
constexpr CompCfg WORDS_EN_100K = {
    .dataset_file = "words_en.tsv", .dataset_name = "words_en",
    .n_members = 100'000, .fpr_target = 0.01,
    .n_train_neg = 100'000, .threshold = 0.5,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr CompCfg WORDS_EN_100K_T03 = {
    .dataset_file = "words_en.tsv", .dataset_name = "words_en",
    .n_members = 100'000, .fpr_target = 0.01,
    .n_train_neg = 100'000, .threshold = 0.3,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr CompCfg WORDS_EN_100K_T07 = {
    .dataset_file = "words_en.tsv", .dataset_name = "words_en",
    .n_members = 100'000, .fpr_target = 0.01,
    .n_train_neg = 100'000, .threshold = 0.7,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};

} // namespace

// ---------------------------------------------------------------------------
// Registration — Classical BF (N sweep, τ not applicable)
// ---------------------------------------------------------------------------

BENCHMARK_CAPTURE(BM_Compare_Classical_Miss, words_en_10k,  WORDS_EN_10K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Classical_Miss, words_en_50k,  WORDS_EN_50K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Classical_Miss, words_en_100k, WORDS_EN_100K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_Compare_Classical_Hit, words_en_10k,  WORDS_EN_10K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Classical_Hit, words_en_50k,  WORDS_EN_50K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Classical_Hit, words_en_100k, WORDS_EN_100K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

// ---------------------------------------------------------------------------
// Registration — Learned BF (N × τ sweep)
// ---------------------------------------------------------------------------

BENCHMARK_CAPTURE(BM_Compare_Learned_Miss, words_en_10k_t03,  WORDS_EN_10K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Miss, words_en_10k_t05,  WORDS_EN_10K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Miss, words_en_10k_t07,  WORDS_EN_10K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_Compare_Learned_Miss, words_en_50k_t03,  WORDS_EN_50K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Miss, words_en_50k_t05,  WORDS_EN_50K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Miss, words_en_50k_t07,  WORDS_EN_50K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_Compare_Learned_Miss, words_en_100k_t03, WORDS_EN_100K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Miss, words_en_100k_t05, WORDS_EN_100K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Miss, words_en_100k_t07, WORDS_EN_100K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_Compare_Learned_Hit, words_en_10k_t03,  WORDS_EN_10K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Hit, words_en_10k_t05,  WORDS_EN_10K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Hit, words_en_10k_t07,  WORDS_EN_10K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_Compare_Learned_Hit, words_en_50k_t03,  WORDS_EN_50K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Hit, words_en_50k_t05,  WORDS_EN_50K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Hit, words_en_50k_t07,  WORDS_EN_50K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_Compare_Learned_Hit, words_en_100k_t03, WORDS_EN_100K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Hit, words_en_100k_t05, WORDS_EN_100K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Hit, words_en_100k_t07, WORDS_EN_100K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

// ---------------------------------------------------------------------------
// Dataset configurations — Tranco URL fixture
// ---------------------------------------------------------------------------
// AUC(sklearn proxy) = 0.9991. Generate: bash benchmarks/fixtures/gen_urls_tranco.sh

namespace {

constexpr CompCfg URLS_TRANCO_10K = {
    .dataset_file = "urls_tranco.tsv", .dataset_name = "urls_tranco",
    .n_members = 10'000, .fpr_target = 0.01,
    .n_train_neg = 10'000, .threshold = 0.5,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr CompCfg URLS_TRANCO_10K_T03 = {
    .dataset_file = "urls_tranco.tsv", .dataset_name = "urls_tranco",
    .n_members = 10'000, .fpr_target = 0.01,
    .n_train_neg = 10'000, .threshold = 0.3,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr CompCfg URLS_TRANCO_10K_T07 = {
    .dataset_file = "urls_tranco.tsv", .dataset_name = "urls_tranco",
    .n_members = 10'000, .fpr_target = 0.01,
    .n_train_neg = 10'000, .threshold = 0.7,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr CompCfg URLS_TRANCO_50K = {
    .dataset_file = "urls_tranco.tsv", .dataset_name = "urls_tranco",
    .n_members = 50'000, .fpr_target = 0.01,
    .n_train_neg = 50'000, .threshold = 0.5,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr CompCfg URLS_TRANCO_50K_T03 = {
    .dataset_file = "urls_tranco.tsv", .dataset_name = "urls_tranco",
    .n_members = 50'000, .fpr_target = 0.01,
    .n_train_neg = 50'000, .threshold = 0.3,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr CompCfg URLS_TRANCO_50K_T07 = {
    .dataset_file = "urls_tranco.tsv", .dataset_name = "urls_tranco",
    .n_members = 50'000, .fpr_target = 0.01,
    .n_train_neg = 50'000, .threshold = 0.7,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr CompCfg URLS_TRANCO_100K = {
    .dataset_file = "urls_tranco.tsv", .dataset_name = "urls_tranco",
    .n_members = 100'000, .fpr_target = 0.01,
    .n_train_neg = 100'000, .threshold = 0.5,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr CompCfg URLS_TRANCO_100K_T03 = {
    .dataset_file = "urls_tranco.tsv", .dataset_name = "urls_tranco",
    .n_members = 100'000, .fpr_target = 0.01,
    .n_train_neg = 100'000, .threshold = 0.3,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};
constexpr CompCfg URLS_TRANCO_100K_T07 = {
    .dataset_file = "urls_tranco.tsv", .dataset_name = "urls_tranco",
    .n_members = 100'000, .fpr_target = 0.01,
    .n_train_neg = 100'000, .threshold = 0.7,
    .feature_dim = 16384, .epochs = 20, .is_fixture = true};

} // namespace

// Classical BF — URL N sweep
BENCHMARK_CAPTURE(BM_Compare_Classical_Miss, urls_tranco_10k,  URLS_TRANCO_10K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Classical_Miss, urls_tranco_50k,  URLS_TRANCO_50K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Classical_Miss, urls_tranco_100k, URLS_TRANCO_100K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_Compare_Classical_Hit, urls_tranco_10k,  URLS_TRANCO_10K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Classical_Hit, urls_tranco_50k,  URLS_TRANCO_50K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Classical_Hit, urls_tranco_100k, URLS_TRANCO_100K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

// Learned BF — URL N × τ sweep
BENCHMARK_CAPTURE(BM_Compare_Learned_Miss, urls_tranco_10k_t03,  URLS_TRANCO_10K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Miss, urls_tranco_10k_t05,  URLS_TRANCO_10K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Miss, urls_tranco_10k_t07,  URLS_TRANCO_10K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Miss, urls_tranco_50k_t03,  URLS_TRANCO_50K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Miss, urls_tranco_50k_t05,  URLS_TRANCO_50K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Miss, urls_tranco_50k_t07,  URLS_TRANCO_50K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Miss, urls_tranco_100k_t03, URLS_TRANCO_100K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Miss, urls_tranco_100k_t05, URLS_TRANCO_100K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Miss, urls_tranco_100k_t07, URLS_TRANCO_100K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_Compare_Learned_Hit, urls_tranco_10k_t03,  URLS_TRANCO_10K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Hit, urls_tranco_10k_t05,  URLS_TRANCO_10K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Hit, urls_tranco_10k_t07,  URLS_TRANCO_10K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Hit, urls_tranco_50k_t03,  URLS_TRANCO_50K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Hit, urls_tranco_50k_t05,  URLS_TRANCO_50K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Hit, urls_tranco_50k_t07,  URLS_TRANCO_50K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Hit, urls_tranco_100k_t03, URLS_TRANCO_100K_T03)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Hit, urls_tranco_100k_t05, URLS_TRANCO_100K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_Compare_Learned_Hit, urls_tranco_100k_t07, URLS_TRANCO_100K_T07)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
