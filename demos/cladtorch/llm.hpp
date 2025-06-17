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
    auto wte_out = wte.lookup(input);
    auto wpe_out = wpe.lookup(input_pos);
    auto result = wte_out + wpe_out;
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
    // For Linear layer: input @ weight.T + bias
    // Since our weight is stored as (out_features, in_features), we need to transpose it
    auto weight_transposed = weight.transpose(0, 1); // Now (out_features, in_features) -> (in_features, out_features)
    auto matmul_result = matmul(input, weight_transposed);
    auto result = matmul_result + bias;
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
    auto normalized = input.norm();
    auto weighted = normalized * weight;
    auto result = weighted + bias;
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
    
    auto input_shape = input.shape();
    int B = input_shape[0];  // batch size
    int T = input_shape[1];  // sequence length
    
    
    // Get Q, K, V from input
    auto qkv_out = qkv.forward(input);
    
    auto qkv_split = qkv_out.split(channels, 2); 
    
    auto q = qkv_split[0];  // (B, T, channels)
    auto k = qkv_split[1];  // (B, T, channels)
    auto v = qkv_split[2];  // (B, T, channels)
    
    
    // Reshape for multi-head attention: (B, T, channels) -> (B, T, num_heads, head_size)
    q = q.reshape({B, T, num_heads, head_size});
    k = k.reshape({B, T, num_heads, head_size});
    v = v.reshape({B, T, num_heads, head_size});
    
    
    // Transpose to (B, num_heads, T, head_size)
    q = q.transpose(1, 2);
    k = k.transpose(1, 2);
    v = v.transpose(1, 2);
    
    
    // Compute attention scores: Q @ K^T
    auto k_transposed = k.transpose(2, 3);  // (B, num_heads, head_size, T)
    
    auto scores = matmul(q, k_transposed);  // (B, num_heads, T, T)
    
    // Scale by sqrt(head_size)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    scores = scores * scale;
    
    // Apply softmax to get attention weights
    auto attn_weights = softmax(scores);  // (B, num_heads, T, T)
    
    // Apply attention weights to values
    auto attn_out = matmul(attn_weights, v);  // (B, num_heads, T, head_size)
    
    // Transpose back and reshape
    attn_out = attn_out.transpose(1, 2);  // (B, T, num_heads, head_size)
    
    attn_out = attn_out.reshape({B, T, channels});  // (B, T, channels)
    
    // Apply output projection
    auto result = attproj.forward(attn_out);
    
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
    
    // Pre-norm architecture: LayerNorm -> Attention -> Residual
    auto normed1 = ln1.forward(input);
    
    auto attn_out = attn.forward(normed1);
    
    auto residual1 = input + attn_out;  // Residual connection
    
    // Pre-norm architecture: LayerNorm -> MLP -> Residual  
    auto normed2 = ln2.forward(residual1);
    
    auto fc_out = fc.forward(normed2);
    
    // Apply GELU activation element-wise
    auto gelu_out = gelu(fc_out);  // GELU activation function
    
    auto proj_out = fc_proj.forward(gelu_out);
    
    auto residual2 = residual1 + proj_out;  // Residual connection
    
    return residual2;
  }
};

struct Transformer {
  Encoder encoder; // Encoder layer
  std::vector<Block> blocks; // List of transformer blocks
  LayerNorm ln_f; // Final layer normalization

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
    auto x = encoder.forward(input, input_pos);
    
    // Pass through all transformer blocks
    for (int i = 0; i < blocks.size(); ++i) {
      x = blocks[i].forward(x);
    }
    
    // Apply final layer normalization
    x = ln_f.forward(x);
    
    return x;
  }
};

// Define the GPT2 model structure
struct GPT2 {
  GPT2Config config; // Configuration parameters
  Transformer transformer; // Transformer model

  int num_parameters;    // Total number of parameters

  int B; // Current batch size (B)
  int T;    // Current sequence length (T)

  // Constructor to initialize GPT2 model
  GPT2(const GPT2Config& config) 
      : config(config), 
        transformer(config),
        B(0), T(0) {}

  // Constructor to load model from checkpoint file (reads hyperparameters and weights)
  GPT2(const std::string& checkpoint_path) 
      : config(load_config_from_checkpoint(checkpoint_path)),
        transformer(config),
        B(0), T(0) {
    load_weights_from_checkpoint(checkpoint_path);
  }

  // Forward pass for GPT2
  FTensor forward(const ITensor& input) {
    const auto B = input.shape()[0]; // Batch size
    const auto T = input.shape()[1]; // Sequence length
    ITensor input_pos({B, T}); // Position indices
    for (int i = 0; i < T; ++i) {
      for (int b = 0; b < B; ++b) {
        input_pos.at(b, i) = i; // Fill position indices
      }
    }
    
    auto logits = transformer.forward(input, input_pos);
        
    // TODO: causal mask for logits if needed; vocab size/padding?
    auto probs = softmax(logits);
    
    return probs;
  }

  // Type alias for config for template compatibility
  using ConfigType = GPT2Config;

  // Get all parameter tensors in the correct order for checkpoint loading
  std::vector<FTensor*> get_parameter_tensors() {
    std::vector<FTensor*> params;
    
    // Embedding parameters
    params.push_back(&transformer.encoder.wte);
    params.push_back(&transformer.encoder.wpe);

    // Block parameters (in the specific order from the checkpoint format)
    // For each parameter type, iterate through all layers
    
    // ln1 weights and biases
    for (int l = 0; l < config.num_layers; l++) {
      params.push_back(&transformer.blocks[l].ln1.weight);
    }
    for (int l = 0; l < config.num_layers; l++) {
      params.push_back(&transformer.blocks[l].ln1.bias);
    }
    
    // qkv weights and biases  
    for (int l = 0; l < config.num_layers; l++) {
      params.push_back(&transformer.blocks[l].attn.qkv.weight);
    }
    for (int l = 0; l < config.num_layers; l++) {
      params.push_back(&transformer.blocks[l].attn.qkv.bias);
    }
    
    // attention projection weights and biases
    for (int l = 0; l < config.num_layers; l++) {
      params.push_back(&transformer.blocks[l].attn.attproj.weight);
    }
    for (int l = 0; l < config.num_layers; l++) {
      params.push_back(&transformer.blocks[l].attn.attproj.bias);
    }
    
    // ln2 weights and biases
    for (int l = 0; l < config.num_layers; l++) {
      params.push_back(&transformer.blocks[l].ln2.weight);
    }
    for (int l = 0; l < config.num_layers; l++) {
      params.push_back(&transformer.blocks[l].ln2.bias);
    }
    
    // fc weights and biases
    for (int l = 0; l < config.num_layers; l++) {
      params.push_back(&transformer.blocks[l].fc.weight);
    }
    for (int l = 0; l < config.num_layers; l++) {
      params.push_back(&transformer.blocks[l].fc.bias);
    }
    
    // fc projection weights and biases
    for (int l = 0; l < config.num_layers; l++) {
      params.push_back(&transformer.blocks[l].fc_proj.weight);
    }
    for (int l = 0; l < config.num_layers; l++) {
      params.push_back(&transformer.blocks[l].fc_proj.bias);
    }

    // Final layer norm weights and biases
    params.push_back(&transformer.ln_f.weight);
    params.push_back(&transformer.ln_f.bias);

    return params;
  }

  // Load model weights from checkpoint file
  // Static method to read configuration from checkpoint file
  static GPT2Config load_config_from_checkpoint(const std::string& checkpoint_path) {
    FILE* model_file = fopen(checkpoint_path.c_str(), "rb");
    if (model_file == nullptr) {
      throw std::runtime_error("Could not open the model checkpoint file: " + checkpoint_path);
    }

    int model_header[256];
    fread(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) {
      fclose(model_file);
      throw std::runtime_error("Bad magic number in model checkpoint file: " + checkpoint_path);
    }
    if (model_header[1] != 3) {
      fclose(model_file);
      throw std::runtime_error("Bad version number in model checkpoint file: " + checkpoint_path);
    }

    // Extract hyperparameters from header
    int maxT = model_header[2];
    int V = model_header[3];
    int L = model_header[4];
    int NH = model_header[5];
    int C = model_header[6];
    int Vp = model_header[7];

    fclose(model_file);
    
    std::cerr << "[GPT-2 Checkpoint Config]:" << std::endl;
    std::cerr << "max_seq_len: " << maxT << std::endl;
    std::cerr << "vocab_size: " << V << std::endl;
    std::cerr << "padded_vocab_size: " << Vp << std::endl;
    std::cerr << "num_layers: " << L << std::endl;
    std::cerr << "num_heads: " << NH << std::endl;
    std::cerr << "channels: " << C << std::endl;

    return GPT2Config{maxT, V, Vp, L, NH, C};
  }

  void load_weights_from_checkpoint(const std::string& checkpoint_path) {
    FILE* model_file = fopen(checkpoint_path.c_str(), "rb");
    if (model_file == nullptr) {
      throw std::runtime_error("Could not open the model checkpoint file: " + checkpoint_path);
    }

    int model_header[256];
    fread(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) {
      fclose(model_file);
      throw std::runtime_error("Bad magic number in model checkpoint file: " + checkpoint_path);
    }
    if (model_header[1] != 3) {
      fclose(model_file);
      throw std::runtime_error("Bad version number in model checkpoint file: " + checkpoint_path);
    }

    // Get all parameter tensors in the correct order
    auto params = get_parameter_tensors();

    // Load the parameters
    num_parameters = 0;
    for (auto* tensor : params) {
      int num_elements = tensor->num_elements();
      num_parameters += num_elements;
      size_t bytes_read = fread(tensor->data(), sizeof(float), num_elements, model_file);
      if (bytes_read != num_elements) {
        fclose(model_file);
        throw std::runtime_error("Failed to read tensor data from checkpoint file");
      }
    }
    
    fclose(model_file);

    // std::cerr << "Number of Parameters: " << num_parameters << std::endl;
    // std::cerr << "Weights loaded successfully!" << std::endl;
  }

  void load_checkpoint(const std::string& checkpoint_path) {
    FILE* model_file = fopen(checkpoint_path.c_str(), "rb");
    if (model_file == nullptr) {
      throw std::runtime_error("Could not open the model checkpoint file: " + checkpoint_path);
    }

    int model_header[256];
    fread(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) {
      throw std::runtime_error("Bad magic number in model checkpoint file: " + checkpoint_path);
    }
    if (model_header[1] != 3) {
      throw std::runtime_error("Bad version number in model checkpoint file: " + checkpoint_path);
    }

    // Verify hyperparameters match
    int maxT = model_header[2];
    int V = model_header[3];
    int L = model_header[4];
    int NH = model_header[5];
    int C = model_header[6];
    int Vp = model_header[7];

    if (maxT != config.max_seq_len || V != config.vocab_size || 
        L != config.num_layers || NH != config.num_heads || 
        C != config.channels || Vp != config.padded_vocab_size) {
      fclose(model_file);
      throw std::runtime_error("Model configuration mismatch with checkpoint file");
    }

    std::cerr << "[GPT-2 Checkpoint Loading]:" << std::endl;
    std::cerr << "max_seq_len: " << maxT << std::endl;
    std::cerr << "vocab_size: " << V << std::endl;
    std::cerr << "padded_vocab_size: " << Vp << std::endl;
    std::cerr << "num_layers: " << L << std::endl;
    std::cerr << "num_heads: " << NH << std::endl;
    std::cerr << "channels: " << C << std::endl;

    // Get all parameter tensors in the correct order
    auto params = get_parameter_tensors();

    // Load the parameters
    num_parameters = 0;
    for (auto* tensor : params) {
      int num_elements = tensor->num_elements();
      num_parameters += num_elements;
      size_t bytes_read = fread(tensor->data(), sizeof(float), num_elements, model_file);
      if (bytes_read != num_elements) {
        fclose(model_file);
        throw std::runtime_error("Failed to read tensor data from checkpoint file");
      }
    }
    
    fclose(model_file);

    std::cerr << "Number of Parameters: " << num_parameters << std::endl;
    std::cerr << "Checkpoint loaded successfully!" << std::endl;
  }
};