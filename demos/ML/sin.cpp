#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "clad/Differentiator/Differentiator.h"

// Network to learn y = sin(x)
struct Net : torch::nn::Module {
  Net() {
    // An MLP: 1 input -> 64 hidden -> 64 hidden -> 1 output
    fc1 = register_module("fc1", torch::nn::Linear(1, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 64));
    fc3 = register_module("fc3", torch::nn::Linear(64, 1));
  }
  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    x = fc3->forward(x);
    return x;
  }
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

// Create a global instance of the network. For a real application,
// this might be managed as part of a larger class or passed by reference etc.
Net net;

// =================================================================================
// The Custom Derivative for Clad
// This is the bridge between Clad's world and libtorch.
// =================================================================================
namespace clad::custom_derivatives {

CUDA_HOST_DEVICE
ValueAndPushforward<double, double>
call_torch_model_pushforward(double x, double d_x) {
  // --- Step 1: Prepare Tensors for libtorch ---
  // Convert the scalar C++ double inputs into torch::Tensor objects
  // We need to enable gradient tracking on the input tensor 't' to get its gradient later.
  const auto opt = torch::TensorOptions().requires_grad(true);
  torch::Tensor t = torch::full({1}, x, opt);

  // 'd_x' is the tangent from Clad's forward-mode AD. We put it in a tensor
  // to pass to torch's backward function.
  torch::Tensor d_t_tangent = torch::full({1}, d_x);

  // --- Step 2: Forward Pass through the torch model ---
  torch::Tensor ret = net.forward(t);

  // --- Step 3: Backward Pass using torch's autograd to get the derivative ---
  ret.backward(d_t_tangent);

  // --- Step 4: Extract results and return them to Clad ---
  // Get the scalar value of the model's output
  double result_val = ret.item<double>();

  // Get the computed gradient from the input tensor 't'
  // This is our final derivative: (dy/dx) * d_x
  double derivative_val = t.grad().item<double>();

  return {result_val, derivative_val};
}
}  // namespace clad::custom_derivatives

// This is the C++ function we want Clad to differentiate.
double call_torch_model(double x) {
  // This code is NOT executed when Clad differentiates. It's only for a direct C++ call (e.g., for inference).
  torch::NoGradGuard no_grad;
  torch::Tensor input = torch::full({1}, x);
  return net.forward(input).item<double>();
}

// A wrapper function that Clad will see.
double differentiable_code_wrapper(double x) { return call_torch_model(x); }

int main() {
  // =================================================================================
  // Part 1: Train the libtorch model to approximate y = sin(x)
  // =================================================================================
  std::cout << "--- Part 1: Training PyTorch Model to learn y = sin(x) ---" << std::endl;

  torch::optim::Adam optimizer(net.parameters(), torch::optim::AdamOptions(1e-3));

  // Generate training data
  torch::Tensor train_x = torch::linspace(-M_PI, M_PI, 2000).unsqueeze(1);
  torch::Tensor train_y = torch::sin(train_x);

  for (int epoch = 0; epoch < 2000; ++epoch) {
    optimizer.zero_grad();
    torch::Tensor prediction = net.forward(train_x);
    torch::Tensor loss = torch::nn::functional::mse_loss(prediction, train_y);
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 500 == 0) {
      std::cout << "Epoch [" << epoch + 1 << "/2000], Loss: " << loss.item<double>() << std::endl;
    }
  }
  std::cout << "--- Training Complete ---" << std::endl;

  // =================================================================================
  // Part 2: Use Clad to differentiate the *trained* model
  // =================================================================================
  std::cout << "\n--- Part 2: Differentiating with Clad + Custom Derivative ---" << std::endl;

  double x_val = 3 * M_PI / 4.0;
  double grad_val = 0;

  auto df = clad::gradient(differentiable_code_wrapper);
  // df.dump();
  df.execute(x_val, &grad_val);

  // =================================================================================
  // Part 3: Verification
  // =================================================================================
  std::cout << "\n--- Part 3: Verification ---" << std::endl;
  std::cout.precision(6);
  std::cout << "Input value x: " << x_val << " (PI/4)" << std::endl;
  std::cout << "Model output f(x) [should be ~sin(PI/4)]: " << call_torch_model(x_val) << std::endl;
  std::cout << "True value sin(PI/4):                      " << sin(x_val) << std::endl;
  std::cout << std::endl;
  std::cout << "Clad computed gradient f'(x):              " << grad_val << std::endl;
  std::cout << "True gradient cos(PI/4):                   " << cos(x_val) << std::endl;

  return 0;
}