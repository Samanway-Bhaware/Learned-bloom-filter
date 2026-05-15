/// @file train_main.cpp
/// @brief lbf-train — Train a Learned Bloom Filter logistic regression model.
///
/// Reads a labelled TSV file (one <key><tab><label> per line, label in {0,1}),
/// trains a sparse logistic regression model via mini-batch SGD, and serialises
/// the result to a binary file loadable by @ref MembershipModel::load().
///
/// Usage:
/// @code
///   lbf-train -i members.tsv -o model.lbf [options]
/// @endcode

#include "lbf/models/logistic_regression.hpp"

#include <algorithm>
#include <cstdint>
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  // ---------------------------------------------------------------------------
  // Argument parsing
  // ---------------------------------------------------------------------------

  // clang-format off
  cxxopts::Options options{
      "lbf-train",
      "Train a Learned Bloom Filter logistic regression model.\n"
      "Input: TSV file with one <key><tab><label> per line.\n"
      "       label must be 0 (non-member) or 1 (member)."};

  options.add_options()
    ("i,input",      "Input TSV file path (required)",
                     cxxopts::value<std::string>())
    ("o,output",     "Output model file path (required)",
                     cxxopts::value<std::string>())
    ("min-n",        "Minimum n-gram length >= 1",
                     cxxopts::value<size_t>()->default_value("3"))
    ("max-n",        "Maximum n-gram length >= min-n",
                     cxxopts::value<size_t>()->default_value("5"))
    ("feature-dim",  "Feature space size, must be a power of 2",
                     cxxopts::value<size_t>()->default_value("65536"))
    ("epochs",       "Number of training epochs",
                     cxxopts::value<size_t>()->default_value("10"))
    ("lr",           "Learning rate eta",
                     cxxopts::value<double>()->default_value("0.1"))
    ("l2",           "L2 regularization coefficient lambda",
                     cxxopts::value<double>()->default_value("0.0001"))
    ("batch-size",   "Mini-batch size",
                     cxxopts::value<size_t>()->default_value("256"))
    ("momentum",     "SGD momentum coefficient beta",
                     cxxopts::value<double>()->default_value("0.9"))
    ("seed",         "Random seed for reproducibility",
                     cxxopts::value<uint64_t>()->default_value("42"))
    ("v,verbose",    "Print per-epoch loss and accuracy to stderr")
    ("h,help",       "Print this help message and exit");
  // clang-format on

  std::string input_path;
  std::string output_path;
  lbf::NGramConfig ngram_config;
  lbf::TrainingConfig train_config;

  try {
    auto result = options.parse(argc, argv);

    if (result.count("help") != 0U) {
      std::cout << options.help() << '\n';
      return 0;
    }

    if (result.count("input") == 0U || result.count("output") == 0U) {
      std::cerr << "Error: --input and --output are required.\n\n" << options.help() << '\n';
      return 1;
    }

    input_path = result["input"].as<std::string>();
    output_path = result["output"].as<std::string>();

    ngram_config.min_n_ = result["min-n"].as<size_t>();
    ngram_config.max_n_ = result["max-n"].as<size_t>();
    ngram_config.feature_dim_ = result["feature-dim"].as<size_t>();

    train_config.epochs_ = result["epochs"].as<size_t>();
    train_config.learning_rate_ = result["lr"].as<double>();
    train_config.l2_regularization_ = result["l2"].as<double>();
    train_config.batch_size_ = result["batch-size"].as<size_t>();
    train_config.momentum_ = result["momentum"].as<double>();
    train_config.random_seed_ = result["seed"].as<uint64_t>();
    train_config.verbose_ = result.count("verbose") != 0U;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\nRun with --help for usage.\n";
    return 1;
  }

  // ---------------------------------------------------------------------------
  // Load training data from TSV
  // ---------------------------------------------------------------------------

  std::ifstream ifs{input_path};
  if (!ifs) {
    std::cerr << "Error: cannot open input file: " << input_path << '\n';
    return 1;
  }

  std::vector<std::string> key_strings;
  std::vector<uint8_t> labels;
  std::string line;
  size_t line_num = 0;

  while (std::getline(ifs, line)) {
    ++line_num;

    // Strip Windows-style CR if present.
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) {
      continue;
    }

    const auto tab_pos = line.find('\t');
    if (tab_pos == std::string::npos) {
      std::cerr << "Error: line " << line_num << ": missing tab separator.\n";
      return 1;
    }

    const std::string label_str = line.substr(tab_pos + 1U);
    if (label_str != "0" && label_str != "1") {
      std::cerr << "Error: line " << line_num << ": label must be '0' or '1', got '" << label_str
                << "'.\n";
      return 1;
    }

    key_strings.push_back(line.substr(0, tab_pos));
    labels.push_back(label_str == "1" ? uint8_t{1} : uint8_t{0});
  }

  if (key_strings.empty()) {
    std::cerr << "Error: no valid examples found in: " << input_path << '\n';
    return 1;
  }

  const auto n_pos_signed = std::count(labels.begin(), labels.end(), uint8_t{1});
  const auto n_pos = static_cast<size_t>(n_pos_signed);

  std::cerr << "[lbf-train] loaded " << key_strings.size() << " examples (" << n_pos
            << " positive, " << (key_strings.size() - n_pos) << " negative) from " << input_path
            << '\n';
  std::cerr << "[lbf-train] n=[" << ngram_config.min_n_ << "," << ngram_config.max_n_
            << "]  dim=" << ngram_config.feature_dim_ << "  epochs=" << train_config.epochs_
            << "  lr=" << train_config.learning_rate_ << "  l2=" << train_config.l2_regularization_
            << "  batch=" << train_config.batch_size_ << '\n';

  // Build spans over the stable key_strings storage.
  std::vector<std::span<const std::byte>> key_spans;
  key_spans.reserve(key_strings.size());
  for (const auto &s : key_strings) {
    key_spans.push_back(std::as_bytes(std::span<const char>{s.data(), s.size()}));
  }

  // ---------------------------------------------------------------------------
  // Train and save
  // ---------------------------------------------------------------------------

  try {
    auto [model, metrics] = lbf::LogisticRegressionModel::train(
        std::span<const std::span<const std::byte>>{key_spans}, std::span<const uint8_t>{labels},
        ngram_config, train_config);

    std::cerr << "[lbf-train] final AUC = " << metrics.final_auc_ << '\n';

    std::ofstream ofs{output_path, std::ios::binary};
    if (!ofs) {
      std::cerr << "Error: cannot create output file: " << output_path << '\n';
      return 1;
    }
    model.save(ofs);
    ofs.close();

    std::cerr << "[lbf-train] model saved to " << output_path << " (~" << model.memory_bytes()
              << " weight bytes)\n";

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }

  return 0;
}
