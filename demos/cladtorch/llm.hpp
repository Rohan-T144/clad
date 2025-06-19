#pragma once

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include "cladtorch/cladtorch.hpp"

namespace gpt2 {

using namespace cladtorch;
using FTensor = Tensor<float>;
using ITensor = Tensor<int>;

struct Config {
  int max_seq_len;
  int vocab_size;
  int padded_vocab_size;
  int num_layers;
  int num_heads;
  int channels;

  int head_size() const { return channels / num_heads; }
  int mlp_hidden_size() const { return 4 * channels; }
};

class Linear {
public:
  FTensor weight, bias;
  Linear(int in_features, int out_features) : weight({out_features, in_features}), bias({out_features}) {}
  FTensor forward(const FTensor& input) const { return linear(input, weight, bias); }
};

class LayerNorm {
public:
  FTensor weight, bias;
  explicit LayerNorm(int channels) : weight({channels}), bias({channels}) {}
  FTensor forward(const FTensor& input) const { return input.norm() * weight + bias; }
};

class Encoder {
public:
  FTensor wte, wpe;
  Encoder(int padded_vocab_size, int max_seq_len, int channels)
      : wte({padded_vocab_size, channels}), wpe({max_seq_len, channels}) {}
  FTensor forward(const ITensor& input, const ITensor& input_pos) const {
    return wte.lookup(input) + wpe.lookup(input_pos);
  }
};

class CausalSelfAttention {
public:
  Linear qkv, proj;
  int num_heads, channels, head_size;

  CausalSelfAttention(int num_heads, int channels)
      : qkv(channels, 3 * channels), proj(channels, channels), num_heads(num_heads), channels(channels),
        head_size(channels / num_heads) {
    assert(channels % num_heads == 0 && "channels must be divisible by num_heads");
  }

  FTensor forward(const FTensor& input) const {
    const auto& shape = input.shape();
    const int B = shape[0];
    const int T = shape[1];

    // Compute Q, K, V
    auto qkv_out = qkv.forward(input);
    auto qkv_split = qkv_out.split(channels, 2);
    auto q = qkv_split[0].reshape({B, T, num_heads, head_size}).transpose(1, 2);
    auto k = qkv_split[1].reshape({B, T, num_heads, head_size}).transpose(1, 2);
    auto v = qkv_split[2].reshape({B, T, num_heads, head_size}).transpose(1, 2);

    // Attention computation
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_size));
    auto scores = matmul(q, k.transpose(2, 3)) * scale;
    auto weights = softmax(scores, true, 0);
    auto out = matmul(weights, v);

    // Reshape and project
    out = out.transpose(1, 2).reshape({B, T, channels});
    return proj.forward(out);
  }
};

class Block {
public:
  LayerNorm ln1;
  CausalSelfAttention attn;
  LayerNorm ln2;
  Linear mlp_fc, mlp_proj;

  Block(int num_heads, int channels)
      : ln1(channels), attn(num_heads, channels), ln2(channels), mlp_fc(channels, 4 * channels),
        mlp_proj(4 * channels, channels) {}

  FTensor forward(const FTensor& input) const {
    // Attention block with residual connection
    auto x = input + attn.forward(ln1.forward(input));

    // MLP block with residual connection
    auto mlp_out = mlp_proj.forward(gelu(mlp_fc.forward(ln2.forward(x))));
    return x + mlp_out;
  }
};

class Transformer {
public:
  Encoder encoder;
  std::vector<Block> blocks;
  LayerNorm ln_f;

  explicit Transformer(const Config& config)
      : encoder(config.padded_vocab_size, config.max_seq_len, config.channels), ln_f(config.channels) {
    blocks.reserve(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i)
      blocks.emplace_back(config.num_heads, config.channels);
  }

  FTensor forward(const ITensor& input, const ITensor& input_pos) const {
    auto x = encoder.forward(input, input_pos);
    for (const auto& block : blocks)
      x = block.forward(x);
    return ln_f.forward(x);
  }
};

class GPT2 {
private:
  static constexpr int MAGIC_NUMBER = 20240326;
  static constexpr int VERSION = 3;
  static constexpr int HEADER_SIZE = 256;

  template <typename Func> void for_each_parameter(Func func) {
    // Embedding parameters
    func(&transformer.encoder.wte);
    func(&transformer.encoder.wpe);

    // Block parameters in checkpoint order
    for (auto& block : transformer.blocks)
      func(&block.ln1.weight);
    for (auto& block : transformer.blocks)
      func(&block.ln1.bias);
    for (auto& block : transformer.blocks)
      func(&block.attn.qkv.weight);
    for (auto& block : transformer.blocks)
      func(&block.attn.qkv.bias);
    for (auto& block : transformer.blocks)
      func(&block.attn.proj.weight);
    for (auto& block : transformer.blocks)
      func(&block.attn.proj.bias);
    for (auto& block : transformer.blocks)
      func(&block.ln2.weight);
    for (auto& block : transformer.blocks)
      func(&block.ln2.bias);
    for (auto& block : transformer.blocks)
      func(&block.mlp_fc.weight);
    for (auto& block : transformer.blocks)
      func(&block.mlp_fc.bias);
    for (auto& block : transformer.blocks)
      func(&block.mlp_proj.weight);
    for (auto& block : transformer.blocks)
      func(&block.mlp_proj.bias);

    // Final layer norm
    func(&transformer.ln_f.weight);
    func(&transformer.ln_f.bias);
  }

  static Config read_config_from_file(FILE* file) {
    int header[HEADER_SIZE];
    if (fread(header, sizeof(int), HEADER_SIZE, file) != HEADER_SIZE)
      throw std::runtime_error("Failed to read checkpoint header");

    if (header[0] != MAGIC_NUMBER)
      throw std::runtime_error("Invalid magic number in checkpoint");
    if (header[1] != VERSION)
      throw std::runtime_error("Unsupported checkpoint version");

    Config config{
        header[2], // max_seq_len
        header[3], // vocab_size
        header[7], // padded_vocab_size
        header[4], // num_layers
        header[5], // num_heads
        header[6]  // channels
    };

    std::cerr << "[GPT-2 Config] seq_len:" << config.max_seq_len << " vocab:" << config.vocab_size
              << " layers:" << config.num_layers << " heads:" << config.num_heads << " channels:" << config.channels
              << '\n';

    return config;
  }

  void load_weights_from_file(FILE* file) {
    num_parameters = 0;
    for_each_parameter([&](FTensor* tensor) {
      const int elements = tensor->num_elements();
      if (fread(tensor->data(), sizeof(float), static_cast<size_t>(elements), file) != static_cast<size_t>(elements))
        throw std::runtime_error("Failed to read tensor data");
      num_parameters += elements;
    });
  }

public:
  Config config;
  Transformer transformer;
  int num_parameters = 0;

  explicit GPT2(const Config& cfg) : config(cfg), transformer(cfg) {}

  explicit GPT2(const std::string& checkpoint_path)
      : config(load_config_from_checkpoint(checkpoint_path)), transformer(config) {
    load_weights_from_checkpoint(checkpoint_path);
  }

  FTensor forward(const ITensor& input) const {
    const auto& shape = input.shape();
    const int B = shape[0];
    const int T = shape[1];

    // Create position indices
    ITensor input_pos({B, T});
    for (int b = 0; b < B; ++b)
      for (int t = 0; t < T; ++t)
        input_pos.at(b, t) = t;

    auto hidden = transformer.forward(input, input_pos);
    auto logits = matmul(hidden, transformer.encoder.wte.transpose(0, 1));
    return softmax(logits, false, config.vocab_size);
  }

  std::vector<FTensor*> get_parameter_tensors() {
    std::vector<FTensor*> params;
    for_each_parameter([&](FTensor* tensor) { params.push_back(tensor); });
    return params;
  }

  static Config load_config_from_checkpoint(const std::string& path) {
    auto file = std::unique_ptr<FILE, decltype(&fclose)>(fopen(path.c_str(), "rb"), fclose);
    if (!file)
      throw std::runtime_error("Could not open checkpoint: " + path);
    return read_config_from_file(file.get());
  }

  void load_weights_from_checkpoint(const std::string& path) {
    auto file = std::unique_ptr<FILE, decltype(&fclose)>(fopen(path.c_str(), "rb"), fclose);
    if (!file)
      throw std::runtime_error("Could not open checkpoint: " + path);

    // Skip header
    fseek(file.get(), HEADER_SIZE * sizeof(int), SEEK_SET);
    load_weights_from_file(file.get());

    std::cerr << "Loaded " << num_parameters << " parameters from " << path << '\n';
  }

  void load_checkpoint(const std::string& path) {
    auto file = std::unique_ptr<FILE, decltype(&fclose)>(fopen(path.c_str(), "rb"), fclose);
    if (!file)
      throw std::runtime_error("Could not open checkpoint: " + path);

    auto file_config = read_config_from_file(file.get());

    // Verify config matches
    if (file_config.max_seq_len != config.max_seq_len || file_config.vocab_size != config.vocab_size ||
        file_config.num_layers != config.num_layers || file_config.num_heads != config.num_heads ||
        file_config.channels != config.channels || file_config.padded_vocab_size != config.padded_vocab_size) {
      throw std::runtime_error("Configuration mismatch with checkpoint");
    }

    load_weights_from_file(file.get());
    std::cerr << "Checkpoint loaded: " << num_parameters << " parameters" << '\n';
  }
};

} // namespace gpt2