#include "cladtorch/cladtorch.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

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
  
  // Constructor to initialize embeddings
  Encoder(int padded_vocab_size, int max_seq_len, int channels) 
      : wte({padded_vocab_size, channels}), wpe({max_seq_len, channels}) {
    // Initialize with random values (in practice, these would be loaded from checkpoint)
    wte.fill(0.1f);
    wpe.fill(0.1f);
  }
  
  // forward pass for embedding layer
  FTensor forward(const ITensor& input, const ITensor& input_pos) {
    std::cout << "[DEBUG] Encoder::forward - Input shape: ";
    for (auto dim : input.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    std::cout << "[DEBUG] Encoder::forward - Position shape: ";
    for (auto dim : input_pos.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    auto wte_out = wte.lookup(input);
    std::cout << "[DEBUG] Encoder::forward - Token embedding shape: ";
    for (auto dim : wte_out.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    auto wpe_out = wpe.lookup(input_pos);
    std::cout << "[DEBUG] Encoder::forward - Position embedding shape: ";
    for (auto dim : wpe_out.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    auto result = wte_out + wpe_out;
    std::cout << "[DEBUG] Encoder::forward - Output shape: ";
    for (auto dim : result.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    return result;  // Element-wise addition of token and position embeddings
  }
};

// Define the Linear structure for linear layers
struct Linear {
  FTensor weight; // Weights of the linear layer
  FTensor bias;   // Biases of the linear layer
  // Constructor to initialize the linear layer weights and biases
  Linear(int in_dims, int out_dims) : weight({out_dims, in_dims}), bias({out_dims}) {
    // Initialize weights with small random values and biases to zero
    weight.fill(0.1f);
    bias.fill(0.0f);
  }
  // Forward pass for linear layer
  FTensor forward(const FTensor& input) {
    std::cout << "[DEBUG] Linear::forward - Input shape: ";
    for (auto dim : input.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    std::cout << "[DEBUG] Linear::forward - Weight shape: ";
    for (auto dim : weight.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    auto matmul_result = matmul(weight, input);
    std::cout << "[DEBUG] Linear::forward - Matmul result shape: ";
    for (auto dim : matmul_result.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    auto result = matmul_result + bias;
    std::cout << "[DEBUG] Linear::forward - Final output shape: ";
    for (auto dim : result.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    return result; // Matrix multiplication followed by bias addition
  }
};

struct LayerNorm {
  FTensor weight; // LayerNorm weights
  FTensor bias;   // LayerNorm biases
  
  // Constructor to initialize the LayerNorm weights and biases
  LayerNorm(int channels) : weight({channels}), bias({channels}) {
    // Initialize weights to 1.0 and biases to 0.0
    weight.fill(1.0f);
    bias.fill(0.0f);
  }
  
  // Forward pass for LayerNorm
  FTensor forward(const FTensor& input) {
    std::cout << "[DEBUG] LayerNorm::forward - Input shape: ";
    for (auto dim : input.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    auto normalized = input.norm();
    std::cout << "[DEBUG] LayerNorm::forward - Normalized shape: ";
    for (auto dim : normalized.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    // weight.print();
    auto weighted = normalized * weight;
    auto result = weighted + bias;
    std::cout << "[DEBUG] LayerNorm::forward - Output shape: ";
    for (auto dim : result.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    return result; // Normalize input and apply weights and biases
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
        num_heads(num_heads), channels(channels), head_size(channels / num_heads) {
    // Verify that channels is divisible by num_heads
    assert(channels % num_heads == 0 && "channels must be divisible by num_heads");
  }
  
  // Forward pass for causal self-attention
  FTensor forward(const FTensor& input) {
    std::cout << "[DEBUG] CausalSelfAttention::forward - Starting attention" << std::endl;
    
    auto input_shape = input.shape();
    int B = input_shape[0];  // batch size
    int T = input_shape[1];  // sequence length
    
    std::cout << "[DEBUG] CausalSelfAttention::forward - Input shape: B=" << B << ", T=" << T << ", C=" << input_shape[2] << std::endl;
    std::cout << "[DEBUG] CausalSelfAttention::forward - num_heads=" << num_heads << ", head_size=" << head_size << std::endl;
    
    // Get Q, K, V from input
    std::cout << "[DEBUG] CausalSelfAttention::forward - Computing QKV projection" << std::endl;
    auto qkv_out = qkv.forward(input);
    std::cout << "[DEBUG] CausalSelfAttention::forward - QKV output shape: ";
    for (auto dim : qkv_out.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    std::cout << "[DEBUG] CausalSelfAttention::forward - Splitting QKV" << std::endl;
    auto qkv_split = qkv_out.split(channels, 2); 
    
    auto q = qkv_split[0];  // (B, T, channels)
    auto k = qkv_split[1];  // (B, T, channels)
    auto v = qkv_split[2];  // (B, T, channels)
    
    std::cout << "[DEBUG] CausalSelfAttention::forward - Q shape: ";
    for (auto dim : q.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    // Reshape for multi-head attention: (B, T, channels) -> (B, T, num_heads, head_size)
    std::cout << "[DEBUG] CausalSelfAttention::forward - Reshaping Q, K, V for multi-head" << std::endl;
    q = q.view({B, T, num_heads, head_size});
    k = k.view({B, T, num_heads, head_size});
    v = v.view({B, T, num_heads, head_size});
    
    std::cout << "[DEBUG] CausalSelfAttention::forward - Q after reshape: ";
    for (auto dim : q.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    // Transpose to (B, num_heads, T, head_size)
    std::cout << "[DEBUG] CausalSelfAttention::forward - Transposing Q, K, V" << std::endl;
    q = q.transpose(1, 2);
    k = k.transpose(1, 2);
    v = v.transpose(1, 2);
    
    std::cout << "[DEBUG] CausalSelfAttention::forward - Q after transpose: ";
    for (auto dim : q.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    // Compute attention scores: Q @ K^T
    std::cout << "[DEBUG] CausalSelfAttention::forward - Computing attention scores" << std::endl;
    auto k_transposed = k.transpose(2, 3);  // (B, num_heads, head_size, T)
    std::cout << "[DEBUG] CausalSelfAttention::forward - K transposed shape: ";
    for (auto dim : k_transposed.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    auto scores = matmul(q, k_transposed);  // (B, num_heads, T, T)
    std::cout << "[DEBUG] CausalSelfAttention::forward - Attention scores shape: ";
    for (auto dim : scores.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    // Scale by sqrt(head_size)
    std::cout << "[DEBUG] CausalSelfAttention::forward - Scaling scores" << std::endl;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    scores = scores * scale;
    
    // Apply softmax to get attention weights
    std::cout << "[DEBUG] CausalSelfAttention::forward - Applying softmax" << std::endl;
    auto attn_weights = softmax(scores);  // (B, num_heads, T, T)
    
    // Apply attention weights to values
    std::cout << "[DEBUG] CausalSelfAttention::forward - Applying attention to values" << std::endl;
    auto attn_out = matmul(attn_weights, v);  // (B, num_heads, T, head_size)
    std::cout << "[DEBUG] CausalSelfAttention::forward - Attention output shape: ";
    for (auto dim : attn_out.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    // Transpose back and reshape
    std::cout << "[DEBUG] CausalSelfAttention::forward - Transposing back" << std::endl;
    attn_out = attn_out.transpose(1, 2);  // (B, T, num_heads, head_size)
    std::cout << "[DEBUG] CausalSelfAttention::forward - After transpose back: ";
    for (auto dim : attn_out.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    attn_out = attn_out.view({B, T, channels});  // (B, T, channels)
    std::cout << "[DEBUG] CausalSelfAttention::forward - After final reshape: ";
    for (auto dim : attn_out.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    // Apply output projection
    std::cout << "[DEBUG] CausalSelfAttention::forward - Applying output projection" << std::endl;
    auto result = attproj.forward(attn_out);
    std::cout << "[DEBUG] CausalSelfAttention::forward - Final result shape: ";
    for (auto dim : result.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    return result;
  }
};

// Define the Block structure for transformer layers
struct Block {
  LayerNorm ln1;     // LayerNorm for first layer
  CausalSelfAttention attn; // Causal self-attention layer
  LayerNorm ln2;     // LayerNorm for second layer
  Linear fc;         // Fully connected layer; gelu activation
  Linear fc_proj;    // Fully connected projection layer

  // Constructor to initialize the block
  Block(int num_heads, int channels) 
      : ln1(channels),
        attn(num_heads, channels),
        ln2(channels),
        fc(channels, 4 * channels),  // GPT-2 uses 4x expansion in MLP
        fc_proj(4 * channels, channels) {}

  // Forward pass for transformer block
  FTensor forward(const FTensor& input) {
    std::cout << "[DEBUG] Block::forward - Starting transformer block" << std::endl;
    std::cout << "[DEBUG] Block::forward - Input shape: ";
    for (auto dim : input.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    // Pre-norm architecture: LayerNorm -> Attention -> Residual
    std::cout << "[DEBUG] Block::forward - Applying first LayerNorm" << std::endl;
    auto normed1 = ln1.forward(input);
    
    std::cout << "[DEBUG] Block::forward - Applying attention" << std::endl;
    auto attn_out = attn.forward(normed1);
    
    std::cout << "[DEBUG] Block::forward - Adding residual connection 1" << std::endl;
    auto residual1 = input + attn_out;  // Residual connection
    std::cout << "[DEBUG] Block::forward - After residual 1 shape: ";
    for (auto dim : residual1.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    // Pre-norm architecture: LayerNorm -> MLP -> Residual  
    std::cout << "[DEBUG] Block::forward - Applying second LayerNorm" << std::endl;
    auto normed2 = ln2.forward(residual1);
    
    std::cout << "[DEBUG] Block::forward - Applying first FC layer" << std::endl;
    auto fc_out = fc.forward(normed2);
    std::cout << "[DEBUG] Block::forward - FC output shape: ";
    for (auto dim : fc_out.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    // Apply GELU activation element-wise
    std::cout << "[DEBUG] Block::forward - Applying GELU activation" << std::endl;
    auto gelu_out = gelu(fc_out);  // GELU activation function
    
    std::cout << "[DEBUG] Block::forward - Applying projection layer" << std::endl;
    auto proj_out = fc_proj.forward(gelu_out);
    
    std::cout << "[DEBUG] Block::forward - Adding residual connection 2" << std::endl;
    auto residual2 = residual1 + proj_out;  // Residual connection
    std::cout << "[DEBUG] Block::forward - Final block output shape: ";
    for (auto dim : residual2.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    return residual2;
  }
};

struct Transformer {
  Encoder encoder; // Encoder layer
  std::vector<Block> blocks; // List of transformer blocks
  LayerNorm ln_f;            // Final LayerNorm

  // Constructor to initialize the transformer
  Transformer(const GPT2Config& config) 
      : encoder(config.padded_vocab_size, config.max_seq_len, config.channels),
        ln_f(config.channels) {
    // Initialize blocks
    blocks.reserve(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
      blocks.emplace_back(config.num_heads, config.channels);
    }
  }

  // Forward pass for transformer
  FTensor forward(const ITensor& input, const ITensor& input_pos) {
    std::cout << "[DEBUG] Transformer::forward - Starting transformer" << std::endl;
    auto x = encoder.forward(input, input_pos);
    std::cout << "[DEBUG] Transformer::forward - After encoder, shape: ";
    for (auto dim : x.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    // Pass through all transformer blocks
    for (int i = 0; i < blocks.size(); ++i) {
      std::cout << "[DEBUG] Transformer::forward - Processing block " << i << std::endl;
      x = blocks[i].forward(x);
      std::cout << "[DEBUG] Transformer::forward - After block " << i << ", shape: ";
      for (auto dim : x.shape()) std::cout << dim << " ";
      std::cout << std::endl;
    }
    
    // Final layer normalization
    std::cout << "[DEBUG] Transformer::forward - Applying final LayerNorm" << std::endl;
    x = ln_f.forward(x);
    std::cout << "[DEBUG] Transformer::forward - Final transformer output shape: ";
    for (auto dim : x.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    return x;
  }
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

  // Constructor to initialize GPT2 model
  GPT2(const GPT2Config& config) 
      : config(config), 
        transformer(config),
        lm_head(config.channels, config.padded_vocab_size),
        num_parameters(0), B(0), T(0) {}

  // Forward pass for GPT2
  FTensor forward(const ITensor& input, const ITensor& input_pos) {
    std::cout << "[DEBUG] GPT2::forward - Starting GPT2 forward pass" << std::endl;
    auto x = transformer.forward(input, input_pos);
    
    std::cout << "[DEBUG] GPT2::forward - Applying language model head" << std::endl;
    auto logits = lm_head.forward(x);
    std::cout << "[DEBUG] GPT2::forward - Final logits shape: ";
    for (auto dim : logits.shape()) std::cout << dim << " ";
    std::cout << std::endl;
    
    return logits;
  }
};

// Test function to verify our implementation
int main() {
  // Create a small test configuration
  GPT2Config config;
  config.max_seq_len = 128;
  config.vocab_size = 1000;
  config.padded_vocab_size = 1024;  // Padded to nearest power of 2
  config.num_layers = 2;
  config.num_heads = 4;
  config.channels = 64;

  // Create model
  GPT2 model(config);

  // Create test input
  ITensor input({1, 10});  // Batch size 1, sequence length 10
  ITensor input_pos({1, 10});  // Position indices
  
  // Fill with test data
  for (int i = 0; i < 10; ++i) {
    input.at(0, i) = i % config.vocab_size;
    input_pos.at(0, i) = i;
  }

  std::cout << "=== Testing GPT2 forward pass ===" << std::endl;
  std::cout << "Config - max_seq_len: " << config.max_seq_len << std::endl;
  std::cout << "Config - vocab_size: " << config.vocab_size << std::endl;
  std::cout << "Config - padded_vocab_size: " << config.padded_vocab_size << std::endl;
  std::cout << "Config - num_layers: " << config.num_layers << std::endl;
  std::cout << "Config - num_heads: " << config.num_heads << std::endl;
  std::cout << "Config - channels: " << config.channels << std::endl;
  
  try {
    auto output = model.forward(input, input_pos);
    std::cout << "=== Forward pass successful! ===" << std::endl;
    std::cout << "Final output shape: ";
    for (auto dim : output.shape()) {
      std::cout << dim << " ";
    }
    std::cout << std::endl;
  } catch (const std::exception& e) {
    std::cout << "=== ERROR OCCURRED ===" << std::endl;
    std::cout << "Error: " << e.what() << std::endl;
  }

  return 0;
}
