#include "llm.hpp"

// Test function to verify our implementation
// Helper function to create a test checkpoint file for demonstration
void create_test_checkpoint(const std::string& checkpoint_path, const GPT2Config& config) {
  FILE* file = fopen(checkpoint_path.c_str(), "wb");
  if (!file) {
    throw std::runtime_error("Could not create test checkpoint file");
  }

  // Write header
  int header[256] = {0};
  header[0] = 20240326;  // Magic number
  header[1] = 3;         // Version
  header[2] = config.max_seq_len;
  header[3] = config.vocab_size;
  header[4] = config.num_layers;
  header[5] = config.num_heads;
  header[6] = config.channels;
  header[7] = config.padded_vocab_size;
  
  fwrite(header, sizeof(int), 256, file);

  // Create a temporary model to get parameter tensors
  GPT2 temp_model(config);
  auto params = temp_model.get_parameter_tensors();

  // Write random weights for each parameter
  srand(42); // Fixed seed for reproducible test weights
  for (auto* tensor : params) {
    for (int i = 0; i < tensor->num_elements(); i++) {
      float value = 0.1f * (rand() % 200 - 100) / 100.0f;  // Random values between -0.1 and 0.1
      fwrite(&value, sizeof(float), 1, file);
    }
  }
  
  fclose(file);
  std::cout << "Created test checkpoint: " << checkpoint_path << std::endl;
}

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
  
  // Test 1: Forward pass with random weights
  std::cout << "\n=== Test 1: Forward pass with random weights ===" << std::endl;
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
    return 1;
  }

  // Test 2: Create and load from checkpoint
  std::cout << "\n=== Test 2: Checkpoint loading ===" << std::endl;
  try {
    std::string checkpoint_path = "test_checkpoint.bin";
    // std::string checkpoint_path = "gpt2_124M.bin";
    // Create a test checkpoint file
    // create_test_checkpoint(checkpoint_path, config);
    
    // Create a new model and load from checkpoint
    GPT2 model_from_checkpoint(config);
    model_from_checkpoint.load_checkpoint(checkpoint_path);
    
    // Test forward pass with loaded weights
    std::cout << "\n=== Testing forward pass with loaded weights ===" << std::endl;
    auto output2 = model_from_checkpoint.forward(input, input_pos);
    std::cout << "=== Checkpoint loading and forward pass successful! ===" << std::endl;
    std::cout << "Final output shape: ";
    for (auto dim : output2.shape()) {
      std::cout << dim << " ";
    }
    std::cout << std::endl;
    
    // Clean up test file
    // remove(checkpoint_path.c_str());
    
  } catch (const std::exception& e) {
    std::cout << "=== CHECKPOINT ERROR OCCURRED ===" << std::endl;
    std::cout << "Error: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "\n=== All tests completed successfully! ===" << std::endl;
  return 0;
}
