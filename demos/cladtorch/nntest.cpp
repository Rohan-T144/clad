#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <tuple>
#include <vector>

#include <clad/Differentiator/Differentiator.h>
#include <clad/Differentiator/STLBuiltins.h>
#include <clad/Differentiator/CladtorchBuiltins.h>
#include <cladtorch/simpletorch.hpp>

using namespace std;

const float FRAND_MAX = (float)RAND_MAX;

// nn tensor dimensions
constexpr int INPUT_SIZE = 3;
constexpr int HIDDEN_SIZE = 32;
constexpr int OUTPUT_SIZE = 3;
using namespace cladtorch;
using FTensor = Tensor<float>;

// Define tensor types
struct Layer1 {
  FTensor W; // Weight matrix from input to output
  FTensor b;             // Bias vector
  // Default constructor, Tensor initialization happens automatically
  Layer1() : W({HIDDEN_SIZE, INPUT_SIZE}), b({HIDDEN_SIZE}) {};
  // Forward pass for single layer
  FTensor forward(const FTensor& input) const {
    auto w = matmul(W, input);
    auto bout = w + b; // Add bias
    auto ret = gelu(bout); // Apply GELU activation
    return ret;
  }
  void update_weights(const Layer1& d_l, float learning_rate) {
    W -= d_l.W * learning_rate;
    b -= d_l.b * learning_rate;
  }
};

struct Layer2 {
  FTensor W; // Weight matrix from hidden to output
  FTensor b;              // Bias vector
  // Default constructor, Tensor initialization happens automatically
  Layer2() : W({OUTPUT_SIZE, HIDDEN_SIZE}), b({OUTPUT_SIZE}) {};
  // Forward pass for Layer 2
  FTensor forward(const FTensor& hidden) const { 
    auto w = matmul(W, hidden);
    auto bout = w + b; // Add bias
    return bout;
  }
  void update_weights(const Layer2& d_l, float learning_rate) {
    W -= d_l.W * learning_rate;
    b -= d_l.b * learning_rate;
  }
};

struct NeuralNetwork {
  Layer1 l1;
  Layer2 l2;
  void zero_gradients() {
    l1.W.fill(0.0f);
    l1.b.fill(0.0f);
    l2.W.fill(0.0f);
    l2.b.fill(0.0f);
  }
  void update_weights(const NeuralNetwork& d_nn, float learning_rate) {
    l1.update_weights(d_nn.l1, learning_rate);
    l2.update_weights(d_nn.l2, learning_rate);
  }
  // Forward pass for prediction
  FTensor forward(const FTensor& input) const { 
    auto l1_out = l1.forward(input); // Forward through first layer
    auto l2_out = l2.forward(l1_out); // Forward through second layer
    auto soft = softmax(l2_out, false, 0); // Apply softmax to output layer
    return soft; // Return probabilities
  }
};

// Loss function for the neural network
// Note: input is now a template-based Tensor. Target is still int.
// nn is passed by value for clad to differentiate its members.
float nn_loss(const NeuralNetwork nn, const FTensor input, int target) {
  Tensor probs_buffer = nn.forward(input);
  auto out = cross_entropy_loss(probs_buffer, target); // Returns 1D tensor with the loss
  return out.scalar();                                 // Get the scalar value from the 1D tensor
}

// Simple classification function for generating synthetic data
int classify_input(const float a[3]) {
  double y = 0.8 * a[0] + 1.0 * a[1] + 1.0 * a[2];
  if (y < 0.9)
    return 0;
  else if (y < 1.8)
    return 1;
  else
    return 2;
}

array<float, 3> generate_random_input() {
  return {(float)rand() / FRAND_MAX, (float)rand() / FRAND_MAX, (float)rand() / FRAND_MAX};
}

// Generate dataset for training and validation
tuple<vector<array<float, 3>>, vector<int>, vector<array<float, 3>>, vector<int>> generate_dataset(int train_size,
                                                                                                   int val_size) {
  vector<array<float, 3>> train_inputs, val_inputs;
  vector<int> train_targets, val_targets;

  // Generate training data
  for (int i = 0; i < train_size; i++) {
    auto fr = generate_random_input();
    int y = classify_input(fr.data());
    train_inputs.push_back(fr);
    train_targets.push_back(y);
  }

  // Generate validation data
  for (int i = 0; i < val_size; i++) {
    auto fr = generate_random_input();
    int y = classify_input(fr.data());
    val_inputs.push_back(fr);
    val_targets.push_back(y);
  }

  return {train_inputs, train_targets, val_inputs, val_targets};
}

int main() {
  // Generate dataset
  int train_size = 10000, val_size = 100;
  auto [train_inputs, train_targets, val_inputs, val_targets] = generate_dataset(train_size, val_size);
  // Initialize neural network and weights with small random values
  NeuralNetwork nn;
  for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; ++i)
    nn.l1.W._data[i] = (float)rand() / FRAND_MAX - 0.5f;
  for (int i = 0; i < HIDDEN_SIZE; ++i)
    nn.l1.b._data[i] = (float)rand() / FRAND_MAX - 0.5f;
  for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; ++i)
    nn.l2.W._data[i] = (float)rand() / FRAND_MAX - 0.5f;
  for (int i = 0; i < OUTPUT_SIZE; ++i)
    nn.l2.b._data[i] = (float)rand() / FRAND_MAX - 0.5f;

  // Gradient struct (same structure as network)
  NeuralNetwork d_nn;
  
  auto grad = clad::gradient(nn_loss, "0");
  grad.dump(); // Dump the gradient function for debugging
  FTensor input({INPUT_SIZE}, 0.0f); // Dummy input for gradient execution
  grad.execute(nn, input, 0, &d_nn);

  float learning_rate = 0.01f;
  
  cout << "Training the model...\n";
  for (int epoch = 0; epoch < 20; ++epoch) {
    float total_loss = 0.0f;

    // Training phase
    // for (size_t i = 0; i < train_inputs.size(); ++i) {
    for (size_t i = 0; i < train_inputs.size(); ++i) {
      // std::cerr << "Training on input " << i + 1 << "/" << train_inputs.size() << endl;
      FTensor input({3}, train_inputs[i].data());
      int target = train_targets[i];
      float loss = nn_loss(nn, input, target);
      total_loss += loss;

      // Compute gradients
      d_nn.zero_gradients(); // Reset gradients
      grad.execute(nn, input, target, &d_nn);
      nn.update_weights(d_nn, 0.01f); // Update weights with learning rate
    }
    cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / train_inputs.size() << endl;
  }

  // Test the model
  cout << "\nTesting the model on random examples:\n";
  int correct = 0;
  for (int i = 0; i < 30; ++i) {
    // Generate test case
    auto input_data = generate_random_input();
    int true_class = classify_input(input_data.data());
    FTensor input({3}, input_data.data());
    FTensor probs_out = nn.forward(input);

    // Find predicted class
    int pred_class = 0;
    for (int j = 1; j < OUTPUT_SIZE; ++j)
      if (probs_out.at(j) > probs_out.at(pred_class))
        pred_class = j;

    // Check if prediction is correct
    bool is_correct = (true_class == pred_class);
    if (is_correct)
      correct++;

    cout << "Test " << i << ": Input [" << input_data[0] << ", " << input_data[1] << ", " << input_data[2]
         << "], True: " << true_class << ", Pred: " << pred_class << (is_correct ? " (correct)" : " (wrong)") << endl;
  }
  cout << "\nAccuracy: " << (correct / 30.0) * 100 << "%" << endl;
  return 0;
}
