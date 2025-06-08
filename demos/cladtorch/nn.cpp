#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <tuple>
#include <vector>

// #include "clad/Differentiator/Differentiator.h"
using namespace std;

const float FRAND_MAX = (float)RAND_MAX;

// nn tensor dimensions
constexpr int INPUT_SIZE = 3;
constexpr int OUTPUT_SIZE = 3;
#include "cladtorch/cladtorch.hpp"  // Using basetorch.hpp instead of clad_tensor.hpp
using namespace cladtorch;

// Define tensor types
using InputTensor = Tensor<float, INPUT_SIZE>;
using OutputTensor = Tensor<float, OUTPUT_SIZE>;
using WeightsTensor = Tensor<float, OUTPUT_SIZE, INPUT_SIZE>;
using BiasTensor = Tensor<float, OUTPUT_SIZE>;

struct SingleLayer {
  WeightsTensor W;  // Weight matrix from input to output
  BiasTensor b;     // Bias vector

  // Initialize shape - constructor is now using template-based Tensor
  SingleLayer() = default;  // Default constructor, Tensor initialization happens automatically

  // Forward pass for single layer
  OutputTensor forward(const InputTensor& input) const {
    // logits_out = W·input + b (no activation function, just linear)
    return mat_vec_mul(W, input) + b;
  }

  void update_weights(const SingleLayer& d_l, float learning_rate) {
    W -= d_l.W * learning_rate;
    b -= d_l.b * learning_rate;
  }
};

struct NeuralNetwork {
  SingleLayer layer;

  void zero_gradients() {  // Operates on a d_nn struct passed externally
    layer.W.fill(0.0f);
    layer.b.fill(0.0f);
  }

  void update_weights(const NeuralNetwork& d_nn, float learning_rate) {
    layer.update_weights(d_nn.layer, learning_rate);
  }

  // Forward pass for prediction
  OutputTensor forward(const InputTensor& input) const {
    return softmax(layer.forward(input));
  }
};

// Loss function for the neural network
// nn is passed by value for clad to differentiate its members.
float nn_loss(const NeuralNetwork nn, const InputTensor input, int target) {
  auto out = cross_entropy_loss(nn.forward(input), target);  // Returns scalar tensor with the loss
  return out.scalar(); // Get the scalar value from the tensor
}

// Simple classification function for generating synthetic data
int classify_fruit(float color, float size, float weight) {
  double y = 0.8 * color + 1.0 * size + 1.0 * weight;
  if (y < 0.9)
    return 0;
  else if (y < 1.8)
    return 1;
  else
    return 2;
}

std::array<float, 3> generate_random_fruit() {
  return {(float)rand() / FRAND_MAX, (float)rand() / FRAND_MAX, (float)rand() / FRAND_MAX};
}

// Generate dataset for training and validation
tuple<vector<vector<float>>, vector<int>, vector<vector<float>>, vector<int>> generate_dataset(int train_size,
                                                                                               int val_size) {
  vector<vector<float>> train_inputs, val_inputs;
  vector<int> train_targets, val_targets;

  // Generate training data
  for (int i = 0; i < train_size; i++) {
    auto [x0, x1, x2] = generate_random_fruit();
    int y = classify_fruit(x0, x1, x2);
    train_inputs.push_back({x0, x1, x2});
    train_targets.push_back(y);
  }

  // Generate validation data
  for (int i = 0; i < val_size; i++) {
    auto [x0, x1, x2] = generate_random_fruit();
    int y = classify_fruit(x0, x1, x2);
    val_inputs.push_back({x0, x1, x2});
    val_targets.push_back(y);
  }

  return {train_inputs, train_targets, val_inputs, val_targets};
}

int main() {
  // Generate dataset
  int train_size = 10000, val_size = 100;
  auto [train_inputs, train_targets, val_inputs, val_targets] = generate_dataset(train_size, val_size);
  InputTensor a{};
  
  // Initialize neural network and weights with small random values
  NeuralNetwork nn;
  for (int i = 0; i < OUTPUT_SIZE * INPUT_SIZE; ++i) nn.layer.W._data[i] = (float)rand() / FRAND_MAX - 0.5f;
  for (int i = 0; i < OUTPUT_SIZE; ++i) nn.layer.b._data[i] = (float)rand() / FRAND_MAX - 0.5f;

  // Gradient struct (same structure as network)
  NeuralNetwork d_nn;

  float learning_rate = 0.004f;
  float input_buffer[INPUT_SIZE];  // Reusable buffer for inputs

  // Test the model
  cout << "\nTesting the model on random examples:\n";
  int correct = 0;
  for (int i = 0; i < 30; ++i) {
    // Generate test case
    auto [x0, x1, x2] = generate_random_fruit();
    int true_class = classify_fruit(x0, x1, x2);

    // Set input buffer
    input_buffer[0] = x0;
    input_buffer[1] = x1;
    input_buffer[2] = x2;

    InputTensor input;
    for (int j = 0; j < INPUT_SIZE; ++j) input.at(j) = input_buffer[j];
    
    // Run forward pass
    OutputTensor probs_out = nn.forward(input);

    // Find predicted class
    int pred_class = 0;
    for (int j = 1; j < OUTPUT_SIZE; ++j) {
      if (probs_out.at(j) > probs_out.at(pred_class)) {
        pred_class = j;
      }
    }

    // Check if prediction is correct
    bool is_correct = (true_class == pred_class);
    if (is_correct) correct++;

    cout << "Test " << i << ": Input [" << x0 << ", " << x1 << ", " << x2 << "], True: " << true_class
         << ", Pred: " << pred_class << (is_correct ? " (correct)" : " (wrong)") << endl;
  }
  cout << "\nAccuracy: " << (correct / 30.0) * 100 << "%" << endl;
  return 0;
}
