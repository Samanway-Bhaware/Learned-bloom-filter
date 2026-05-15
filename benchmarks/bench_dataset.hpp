/// @file bench_dataset.hpp
/// @brief Shared dataset loading and measurement utilities for Phase 5 benchmarks.
///
/// Not checked by clang-tidy (benchmarks/ is excluded from the CI tidy scan).
///
/// Dataset files live in LBF_BENCH_DATASETS_DIR (env var) or in the
/// cmake-defined LBF_BENCH_DATASETS_DIR_DEFAULT (absolute path set at
/// configure time).  Each file is a TSV: <key>\t<label> (label ∈ {0,1}).
/// Members (label=1) come first in the file; non-members (label=0) follow.

#pragma once

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace bench {

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

struct Dataset {
  std::string name;
  std::vector<std::string> members;
  std::vector<std::string> nonmembers;

  bool ok() const { return !members.empty(); }
};

// ---------------------------------------------------------------------------
// Measurement result types
// ---------------------------------------------------------------------------

struct FprResult {
  double measured = 0.0;
  double wilson_lo95 = 0.0;
  double wilson_hi95 = 0.0;
  size_t fp_count = 0;
  size_t n_probes = 0;
};

struct PercentileResult {
  double p50_ns = 0.0;
  double p90_ns = 0.0;
  double p99_ns = 0.0;
  double p999_ns = 0.0;
};

// ---------------------------------------------------------------------------
// Path resolution
// ---------------------------------------------------------------------------

inline std::string datasets_dir() {
  const char *env = std::getenv("LBF_BENCH_DATASETS_DIR");
  if (env != nullptr) {
    return std::string{env};
  }
#ifdef LBF_BENCH_DATASETS_DIR_DEFAULT
  return LBF_BENCH_DATASETS_DIR_DEFAULT;
#else
  return "benchmarks/datasets";
#endif
}

/// Directory for curated fixture files (real-word datasets, etc.).
/// Override at runtime with LBF_BENCH_FIXTURES_DIR env var.
inline std::string fixtures_dir() {
  const char *env = std::getenv("LBF_BENCH_FIXTURES_DIR");
  if (env != nullptr) {
    return std::string{env};
  }
#ifdef LBF_BENCH_FIXTURES_DIR_DEFAULT
  return LBF_BENCH_FIXTURES_DIR_DEFAULT;
#else
  return "benchmarks/fixtures";
#endif
}

/// Resolve the full path for a dataset file.
/// @param filename  Basename of the TSV file.
/// @param is_fixture  If true, look in fixtures_dir(); otherwise datasets_dir().
inline std::string resolve_path(const std::string &filename, bool is_fixture) {
  return (is_fixture ? fixtures_dir() : datasets_dir()) + "/" + filename;
}

// ---------------------------------------------------------------------------
// TSV loader
// ---------------------------------------------------------------------------

/// Load a benchmark TSV dataset.
///
/// @param path         Full path to the .tsv file.
/// @param name         Human-readable dataset name stored in Dataset::name.
/// @param max_members  Cap on members loaded (useful to test smaller N without
///                     a separate file).
/// @param max_nonmembers  Cap on non-members loaded.
/// @return Dataset whose ok() is false if the file could not be opened.
inline Dataset load_tsv(const std::string &path, const std::string &name,
                        size_t max_members = SIZE_MAX,
                        size_t max_nonmembers = SIZE_MAX) {
  Dataset ds;
  ds.name = name;

  std::ifstream ifs{path};
  if (!ifs.is_open()) {
    return ds; // ds.ok() == false
  }

  std::string line;
  while (std::getline(ifs, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back(); // strip Windows CR
    }
    if (line.empty()) {
      continue;
    }
    const auto tab = line.rfind('\t');
    if (tab == std::string::npos) {
      continue;
    }
    if (line[tab + 1] == '1' && ds.members.size() < max_members) {
      ds.members.push_back(line.substr(0, tab));
    } else if (line[tab + 1] == '0' && ds.nonmembers.size() < max_nonmembers) {
      ds.nonmembers.push_back(line.substr(0, tab));
    }
  }
  return ds;
}

// ---------------------------------------------------------------------------
// FPR measurement with Wilson 95 % confidence interval
// ---------------------------------------------------------------------------

/// Query the filter on up to n_probes non-members, count false positives,
/// and compute the Wilson score 95 % confidence interval.
///
/// @tparam ContainsFn  Callable: (std::string_view) -> bool.
template <typename ContainsFn>
FprResult measure_fpr(ContainsFn &&contains_fn,
                      const std::vector<std::string> &nonmembers,
                      size_t n_probes = 100'000) {
  n_probes = std::min(n_probes, nonmembers.size());
  if (n_probes == 0) {
    return {};
  }

  size_t fp = 0;
  volatile bool sink = false; // prevent optimisation of contains_fn away
  for (size_t i = 0; i < n_probes; ++i) {
    const bool r = contains_fn(std::string_view{nonmembers[i]});
    sink = sink ^ r;
    if (r) {
      ++fp;
    }
  }
  (void)sink;

  const double p_hat =
      static_cast<double>(fp) / static_cast<double>(n_probes);
  const double n = static_cast<double>(n_probes);
  constexpr double z = 1.96; // 95 % two-tailed
  const double z2n = (z * z) / n;
  const double denom = 1.0 + z2n;
  const double center = (p_hat + z2n * 0.5) / denom;
  const double margin =
      (z / denom) *
      std::sqrt(p_hat * (1.0 - p_hat) / n + z2n / (4.0 * n));

  return {p_hat, std::max(0.0, center - margin), center + margin, fp,
          n_probes};
}

// ---------------------------------------------------------------------------
// Per-operation latency percentiles
// ---------------------------------------------------------------------------

/// Time n_samples individual calls to contains_fn and return sorted
/// percentiles (p50 / p90 / p99 / p999).
///
/// Measured with std::chrono::steady_clock so the values include realistic
/// branch-prediction warm-up.  Keys are cycled from @p keys.
///
/// @tparam ContainsFn  Callable: (std::string_view) -> bool.
template <typename ContainsFn>
PercentileResult measure_percentiles(ContainsFn &&contains_fn,
                                     const std::vector<std::string> &keys,
                                     size_t n_samples = 10'000) {
  if (keys.empty()) {
    return {};
  }
  n_samples = std::max(n_samples, size_t{1000}); // need ≥1000 for p999

  std::vector<double> timings;
  timings.reserve(n_samples);

  volatile bool sink = false;
  for (size_t i = 0; i < n_samples; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool r = contains_fn(std::string_view{keys[i % keys.size()]});
    const auto t1 = std::chrono::steady_clock::now();
    sink = sink ^ r;
    timings.push_back(static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
            .count()));
  }
  (void)sink;

  std::sort(timings.begin(), timings.end());
  const auto pct = [&](double p) -> double {
    const size_t idx = std::min(
        static_cast<size_t>(p * static_cast<double>(n_samples)), n_samples - 1);
    return timings[idx];
  };

  return {pct(0.500), pct(0.900), pct(0.990), pct(0.999)};
}

// ---------------------------------------------------------------------------
// Counter helpers (avoid repetitive casting in benchmark bodies)
// ---------------------------------------------------------------------------

/// Emit all FprResult fields into a benchmark state counter map.
inline void emit_fpr(benchmark::State &state, const FprResult &r) {
  state.counters["lbf_fpr_measured"] = r.measured;
  state.counters["lbf_fpr_wilson_lo95"] = r.wilson_lo95;
  state.counters["lbf_fpr_wilson_hi95"] = r.wilson_hi95;
  state.counters["lbf_fp_count"] = static_cast<double>(r.fp_count);
  state.counters["lbf_n_probes"] = static_cast<double>(r.n_probes);
}

/// Emit all PercentileResult fields into a benchmark state counter map.
inline void emit_pct(benchmark::State &state, const PercentileResult &p) {
  state.counters["lbf_p50_ns"] = p.p50_ns;
  state.counters["lbf_p90_ns"] = p.p90_ns;
  state.counters["lbf_p99_ns"] = p.p99_ns;
  state.counters["lbf_p999_ns"] = p.p999_ns;
}

} // namespace bench
