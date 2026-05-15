/// @file bench_classical_bf.cpp
/// @brief Comprehensive classical BloomFilter benchmarks (Phase 5).
///
/// Measures insert throughput, query throughput, false-positive rate
/// (with Wilson 95 % CI), and per-operation latency percentiles across
/// multiple datasets and filter sizes.
///
/// Datasets are loaded from TSV files at LBF_BENCH_DATASETS_DIR (env var)
/// or the cmake-defined LBF_BENCH_DATASETS_DIR_DEFAULT.
///
/// Run:
///   LBF_BENCH_DATASETS_DIR=benchmarks/datasets \
///     ./build/release/benchmarks/bench_classical_bf \
///     --benchmark_out=results/classical_bf.json \
///     --benchmark_out_format=json \
///     --benchmark_repetitions=5
///
/// Step 2: uniform_int dataset only.
/// Steps 3-4: words_en and urls_tranco datasets added below each dataset block.

#include "bench_dataset.hpp"

#include "lbf/bloom_filter.hpp"

#include <benchmark/benchmark.h>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Per-benchmark configuration
// ---------------------------------------------------------------------------

struct BenchCfg {
  const char *dataset_file; // basename of the TSV file
  const char *dataset_name; // human-readable tag embedded in counters
  size_t n_members;         // member count to load and insert
  double fpr_target;        // target false-positive rate
  bool is_fixture = false;  // true → benchmarks/fixtures/, false → benchmarks/datasets/
};

// ---------------------------------------------------------------------------
// Fixture: BloomFilter built once per BENCHMARK_CAPTURE instantiation.
// GBench calls the function body once before starting the iteration loop,
// so expensive setup (loading, building, measuring) happens outside timing.
// ---------------------------------------------------------------------------

// Insert throughput — rebuild a fresh filter each benchmark run so we always
// measure real (not duplicate) insertions.
static void BM_ClassicalBF_Insert(benchmark::State &state, BenchCfg cfg) {
  const std::string path = bench::resolve_path(cfg.dataset_file, cfg.is_fixture);
  const auto ds =
      bench::load_tsv(path, cfg.dataset_name, cfg.n_members, /*max_nm=*/0);

  if (!ds.ok()) {
    const std::string hint = cfg.is_fixture
        ? "\n  Run: bash benchmarks/fixtures/gen_words_en.sh"
        : "\n  Run: python3 benchmarks/datasets/gen_uniform_int.py --out " + path;
    state.SkipWithError(("dataset not found: " + path + hint).c_str());
    return;
  }

  const size_t n = std::min(cfg.n_members, ds.members.size());
  state.counters["lbf_n_members"] = static_cast<double>(n);
  state.counters["lbf_fpr_target"] = cfg.fpr_target;

  // Pre-allocate a fresh filter (not pre-filled, so every insert is real).
  lbf::BloomFilter<> bf{
      lbf::BloomFilterParams{.expected_items_ = n, .target_fpr_ = cfg.fpr_target}};

  state.counters["lbf_bit_count"] = static_cast<double>(bf.bit_count());
  state.counters["lbf_memory_bytes"] =
      static_cast<double>(bf.bit_count() / 8);

  // Timed loop: cycle through keys.
  size_t idx = 0;
  for (auto _ : state) {
    bf.insert(std::string_view{ds.members[idx % n]});
    ++idx;
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

// Query throughput + FPR + percentiles on non-members (miss path).
static void BM_ClassicalBF_Contains_Miss(benchmark::State &state,
                                         BenchCfg cfg) {
  const std::string path = bench::resolve_path(cfg.dataset_file, cfg.is_fixture);
  const auto ds = bench::load_tsv(path, cfg.dataset_name, cfg.n_members,
                                  /*max_nm=*/200'000);

  if (!ds.ok()) {
    state.SkipWithError(("dataset not found: " + path).c_str());
    return;
  }

  const size_t n = std::min(cfg.n_members, ds.members.size());

  // Build the filter.
  lbf::BloomFilter<> bf{
      lbf::BloomFilterParams{.expected_items_ = n, .target_fpr_ = cfg.fpr_target}};
  for (size_t i = 0; i < n; ++i) {
    bf.insert(std::string_view{ds.members[i]});
  }

  // ----- Setup measurements (not timed) ------------------------------------

  // FPR + Wilson CI.
  const auto fpr = bench::measure_fpr(
      [&](std::string_view k) { return bf.contains(k); }, ds.nonmembers);

  // Per-operation latency percentiles on the miss path.
  const auto pct = bench::measure_percentiles(
      [&](std::string_view k) { return bf.contains(k); }, ds.nonmembers);

  // ----- Counters ----------------------------------------------------------
  state.counters["lbf_n_members"] = static_cast<double>(n);
  state.counters["lbf_fpr_target"] = cfg.fpr_target;
  state.counters["lbf_bit_count"] = static_cast<double>(bf.bit_count());
  state.counters["lbf_memory_bytes"] =
      static_cast<double>(bf.bit_count() / 8);
  bench::emit_fpr(state, fpr);
  bench::emit_pct(state, pct);

  // ----- Timed loop: steady-state miss-path queries ------------------------
  size_t idx = 0;
  const size_t nm = ds.nonmembers.size();
  for (auto _ : state) {
    bool r = bf.contains(std::string_view{ds.nonmembers[idx % nm]});
    benchmark::DoNotOptimize(r);
    ++idx;
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

// Query throughput + percentiles on members (hit path).
static void BM_ClassicalBF_Contains_Hit(benchmark::State &state,
                                        BenchCfg cfg) {
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

  // Per-operation latency percentiles on the hit path.
  const auto pct = bench::measure_percentiles(
      [&](std::string_view k) { return bf.contains(k); }, ds.members);

  state.counters["lbf_n_members"] = static_cast<double>(n);
  state.counters["lbf_fpr_target"] = cfg.fpr_target;
  state.counters["lbf_bit_count"] = static_cast<double>(bf.bit_count());
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
// Dataset: uniform random 64-bit integers
// ---------------------------------------------------------------------------
//
// Expectation: LBF cannot outperform classical BF here — model gets ~50 % AUC.
// The classical BF results are the baseline against which LBF is compared.

constexpr BenchCfg UNIFORM_100K = {
    "uniform_int.tsv", "uniform_int", 100'000, 0.01};
constexpr BenchCfg UNIFORM_1M = {
    "uniform_int.tsv", "uniform_int", 1'000'000, 0.01};

// ---------------------------------------------------------------------------
// Dataset: English words fixture
// ---------------------------------------------------------------------------
//
// Positives = real /usr/share/dict/words; negatives = matched-length random
// alphabetic strings.  AUC of learned model: ~0.996 (sklearn proxy).
// Generate: bash benchmarks/fixtures/gen_words_en.sh
// Validate: python3 scripts/validate_auc.py
//
// Classical BF: FPR ≈ fpr_target regardless of member/non-member structure.
// These configs sweep N to show memory scaling.

constexpr BenchCfg WORDS_EN_10K = {
    .dataset_file = "words_en.tsv",
    .dataset_name = "words_en",
    .n_members    = 10'000,
    .fpr_target   = 0.01,
    .is_fixture   = true,
};
constexpr BenchCfg WORDS_EN_50K = {
    .dataset_file = "words_en.tsv",
    .dataset_name = "words_en",
    .n_members    = 50'000,
    .fpr_target   = 0.01,
    .is_fixture   = true,
};
constexpr BenchCfg WORDS_EN_100K = {
    .dataset_file = "words_en.tsv",
    .dataset_name = "words_en",
    .n_members    = 100'000,
    .fpr_target   = 0.01,
    .is_fixture   = true,
};

} // namespace

// Insert benchmarks — 5 repetitions for trial-level statistics.
BENCHMARK_CAPTURE(BM_ClassicalBF_Insert, uniform_100k, UNIFORM_100K)
    ->MinTime(0.5)
    ->Repetitions(5)
    ->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_ClassicalBF_Insert, uniform_1M, UNIFORM_1M)
    ->MinTime(0.5)
    ->Repetitions(5)
    ->ReportAggregatesOnly(false);

// Miss-path (non-member) query benchmarks — includes FPR + Wilson CI.
BENCHMARK_CAPTURE(BM_ClassicalBF_Contains_Miss, uniform_100k, UNIFORM_100K)
    ->MinTime(0.5)
    ->Repetitions(5)
    ->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_ClassicalBF_Contains_Miss, uniform_1M, UNIFORM_1M)
    ->MinTime(0.5)
    ->Repetitions(5)
    ->ReportAggregatesOnly(false);

// Hit-path (member) query benchmarks.
BENCHMARK_CAPTURE(BM_ClassicalBF_Contains_Hit, uniform_100k, UNIFORM_100K)
    ->MinTime(0.5)
    ->Repetitions(5)
    ->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_ClassicalBF_Contains_Hit, uniform_1M, UNIFORM_1M)
    ->MinTime(0.5)
    ->Repetitions(5)
    ->ReportAggregatesOnly(false);

// ---------------------------------------------------------------------------
// English words fixture benchmarks — N ∈ {10k, 50k, 100k}
// ---------------------------------------------------------------------------

BENCHMARK_CAPTURE(BM_ClassicalBF_Insert, words_en_10k, WORDS_EN_10K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_ClassicalBF_Insert, words_en_50k, WORDS_EN_50K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_ClassicalBF_Insert, words_en_100k, WORDS_EN_100K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_ClassicalBF_Contains_Miss, words_en_10k, WORDS_EN_10K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_ClassicalBF_Contains_Miss, words_en_50k, WORDS_EN_50K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_ClassicalBF_Contains_Miss, words_en_100k, WORDS_EN_100K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_ClassicalBF_Contains_Hit, words_en_10k, WORDS_EN_10K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_ClassicalBF_Contains_Hit, words_en_50k, WORDS_EN_50K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_ClassicalBF_Contains_Hit, words_en_100k, WORDS_EN_100K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

// ---------------------------------------------------------------------------
// Dataset: Tranco URL fixture  (top domains vs matched-segment synthetic)
// ---------------------------------------------------------------------------
//
// Positives  = Tranco top-200k domains (Alexa+Umbrella+Majestic+Farsight blend)
// Negatives  = synthetic domains: same dot-separated segment lengths, random
//              lowercase alphanum content.
// AUC(sklearn proxy) = 0.9991 — highly learnable (common TLD n-grams).
// Generate: bash benchmarks/fixtures/gen_urls_tranco.sh
// Validate: python3.13 scripts/validate_auc.py --tsv benchmarks/fixtures/urls_tranco.tsv

constexpr BenchCfg URLS_TRANCO_10K = {
    .dataset_file = "urls_tranco.tsv",
    .dataset_name = "urls_tranco",
    .n_members    = 10'000,
    .fpr_target   = 0.01,
    .is_fixture   = true,
};
constexpr BenchCfg URLS_TRANCO_50K = {
    .dataset_file = "urls_tranco.tsv",
    .dataset_name = "urls_tranco",
    .n_members    = 50'000,
    .fpr_target   = 0.01,
    .is_fixture   = true,
};
constexpr BenchCfg URLS_TRANCO_100K = {
    .dataset_file = "urls_tranco.tsv",
    .dataset_name = "urls_tranco",
    .n_members    = 100'000,
    .fpr_target   = 0.01,
    .is_fixture   = true,
};

// URL fixture benchmarks — N ∈ {10k, 50k, 100k}
BENCHMARK_CAPTURE(BM_ClassicalBF_Insert, urls_tranco_10k,  URLS_TRANCO_10K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_ClassicalBF_Insert, urls_tranco_50k,  URLS_TRANCO_50K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_ClassicalBF_Insert, urls_tranco_100k, URLS_TRANCO_100K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_ClassicalBF_Contains_Miss, urls_tranco_10k,  URLS_TRANCO_10K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_ClassicalBF_Contains_Miss, urls_tranco_50k,  URLS_TRANCO_50K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_ClassicalBF_Contains_Miss, urls_tranco_100k, URLS_TRANCO_100K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);

BENCHMARK_CAPTURE(BM_ClassicalBF_Contains_Hit, urls_tranco_10k,  URLS_TRANCO_10K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_ClassicalBF_Contains_Hit, urls_tranco_50k,  URLS_TRANCO_50K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
BENCHMARK_CAPTURE(BM_ClassicalBF_Contains_Hit, urls_tranco_100k, URLS_TRANCO_100K)
    ->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(false);
