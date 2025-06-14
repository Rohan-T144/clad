#include "cladtorch/cladtorch.hpp"

using namespace cladtorch;
using FTensor = Tensor<float>;
using ITensor = Tensor<int>;

// Define the GPT2Config structure
struct GPT2Config {
  int max_seq_len;       // Maximum sequence length
  int vocab_size;        // Vocabulary size, e.g. 50257
  int padded_vocab_size; // Padded to e.g. %128==0, 50304
  int num_layers;        // Number of transformer layers
  int num_heads;         // Number of attention heads
  int channels;          // Hidden size (number of channels)
};

// Define the Encoder structure for input embeddings
struct Encoder {
  FTensor wte; // Word token embeddings, (padded_vocab_size, channels)
  FTensor wpe; // Word position embeddings, (max_seq_len, channels)
  // forward pass for embedding layer
  FTensor forward(const ITensor& input, const ITensor& input_pos) {
    return wte.lookup(input) + wpe.lookup(input_pos);  // Element-wise addition of token and position embeddings
  }
};

// Define the Linear structure for linear layers
struct Linear {
  FTensor weight; // Weights of the linear layer
  FTensor bias;   // Biases of the linear layer
  // Constructor to initialize the linear layer weights and biases
  Linear(size_t in_dims, size_t out_dims) : weight({out_dims, in_dims}), bias({out_dims}) {}
  // Forward pass for linear layer
  FTensor forward(const FTensor& input) {
    return matmul(weight, input) + bias; // Matrix multiplication followed by bias addition
  }
};

struct LayerNorm {
  FTensor weight; // LayerNorm weights
  FTensor bias;   // LayerNorm biases
  // Constructor to initialize the LayerNorm weights and biases
  LayerNorm(FTensor weight, FTensor bias) : weight(std::move(weight)), bias(std::move(bias)) {}
  // Forward pass for LayerNorm
  FTensor forward(const FTensor& input) {
    return input.norm() * weight + bias; // Normalize input and apply weights and biases
  }
};

struct CausalSelfAttention {
  // FTensor qkv;        // Query, Key, Value weights
  Linear qkv;      // channels -> 3 * channels // for q, k, v
  Linear attproj; // channels -> channels // for attention projection
  int num_heads;      // Number of attention heads
  int channels;       // Number of channels (hidden size)
  int head_size;      // Size of each attention head

  // Constructor to initialize the attention layer
  CausalSelfAttention(int num_heads, int channels)
      : qkv(channels, 3*channels), attproj(channels, channels),
        num_heads(num_heads), channels(channels), head_size(channels / num_heads) {}
  
  // Forward pass for causal self-attention
  FTensor forward(const FTensor& input) {
    // Implement the forward pass logic for causal self-attention here
    auto qkv_split = qkv.forward(input).split(channels, 2); // Get Q, K, V from input
    // auto q = qkv_split[0].view({-1, num_heads, head_size}).transpose(1, 2); // (B, NH, T, HS)
    // auto k = qkv_split[1].view({-1, num_heads, head_size}).transpose(1, 2); // (B, NH, T, HS)
    // auto v = qkv_split[2].view({-1, num_heads, head_size}).transpose(1, 2); // (B, NH, T, HS)
    
    return input; // Placeholder return
  }
};

// Define the Block structure for transformer layers
struct Block {
  LayerNorm ln1;     // LayerNorm for first layer
  CausalSelfAttention attn; // Causal self-attention layer
  LayerNorm ln2;     // LayerNorm for second layer
  Linear fc;         // Fully connected layer; gelu activation
  Linear fc_proj;    // Fully connected projection layer
};

struct Transformer {
  Encoder encoder; // Encoder layer
  std::vector<Block> blocks; // List of transformer blocks
  LayerNorm ln_f;            // Final LayerNorm
};

// Define the GPT2 model structure
struct GPT2 {
  GPT2Config config; // Configuration parameters
  Transformer transformer; // Transformer model
  Linear lm_head;               // Language model head
  // std::vector<FTensor*> params; // All parameters of the model
  int num_parameters;           // Total number of parameters

  int B; // Current batch size (B)
  int T;    // Current sequence length (T)
};
