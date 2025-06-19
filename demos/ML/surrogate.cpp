#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "clad/Differentiator/Differentiator.h"

// =================================================================================
// The "True" Simulation: Low-Pass RC Filter
// Calculates voltage attenuation based on signal frequency and component values.
// =================================================================================
float true_rc_filter_model(float freq_kHz, float R_kOhms, float C_nF) {
  if (freq_kHz < 1e-9) return 1.0;
  // Convert units to base SI for the physics formula:
  // kHz -> Hz, kOhms -> Ohms, nF -> F
  float freq_Hz = freq_kHz * 1000.0f;
  float R_Ohms = R_kOhms * 1000.0f;
  float C_F = C_nF * 1e-9f;

  float omega = 2.0f * M_PI * freq_Hz;
  return 1.0f / std::sqrt(1.0f + std::pow(omega * R_Ohms * C_F, 2.0f));
}

// =================================================================================
// The Surrogate Model (Neural Network)
// =================================================================================
struct Net : torch::nn::Module {
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
  Net() {
    // 3 inputs (freq, R, C) -> 128 -> 128 -> 1 output (attenuation)
    fc1 = register_module("fc1", torch::nn::Linear(3, 128));
    fc2 = register_module("fc2", torch::nn::Linear(128, 128));
    fc3 = register_module("fc3", torch::nn::Linear(128, 1));
  }
  torch::Tensor forward(torch::Tensor x) {
    // Use tanh activation as it's generally stable and maps to [-1, 1]
    x = torch::tanh(fc1->forward(x));
    x = torch::tanh(fc2->forward(x));
    x = fc3->forward(x);
    return x;
  }
};

Net surrogate_net;

// =================================================================================
// Struct to hold our differentiable circuit meta-parameters
// =================================================================================
struct CircuitParams {
  float R_kOhms;
  float C_nF;
};

// =================================================================================
// The Custom Derivative for Clad
// =================================================================================
namespace clad::custom_derivatives {

// This is the pullback for `run_surrogate`. Clad will automatically calculate
// the derivative of the loss function and pass it in as `_d_predicted`.
void run_surrogate_pullback(const CircuitParams& params, const float freq_kHz, float _d_predicted,
                            CircuitParams* _d_params, float* _d_freq_kHz) {
  // --- 1. Prepare Tensors for LibTorch ---
  // We want gradients w.r.t R and C, but not frequency.
  auto options_grad = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
  auto options_no_grad = torch::TensorOptions().dtype(torch::kFloat32);

  torch::Tensor freq_tensor = torch::tensor({freq_kHz}, options_no_grad);
  torch::Tensor R_tensor = torch::tensor({params.R_kOhms}, options_grad);
  torch::Tensor C_tensor = torch::tensor({params.C_nF}, options_grad);

  torch::Tensor input_tensor = torch::cat({freq_tensor, R_tensor, C_tensor}, 0);

  // --- 2. Forward & Backward Pass ---
  // Perform a forward pass to build the graph
  torch::Tensor predicted_attenuation = surrogate_net.forward(input_tensor);
  // Use the incoming adjoint from Clad to seed the backward pass
  predicted_attenuation.backward(torch::tensor({_d_predicted}));

  // --- 3. Extract Gradients ---
  // The gradients for our meta-parameters are now in their .grad fields.
  _d_params->R_kOhms = R_tensor.grad().item<float>();
  _d_params->C_nF = C_tensor.grad().item<float>();
  // We don't care about the gradient w.r.t frequency, but the signature requires the pointer.
  if (_d_freq_kHz) *_d_freq_kHz = 0;
}

}  // namespace clad::custom_derivatives

// =================================================================================
// C++ Functions for Clad to Differentiate
// =================================================================================

// The function Clad will find a custom derivative for.
float run_surrogate(const CircuitParams& params, const float freq_kHz) {
  torch::NoGradGuard no_grad;
  torch::Tensor input_tensor = torch::tensor({freq_kHz, params.R_kOhms, params.C_nF}, torch::kFloat32);
  return surrogate_net.forward(input_tensor).item<float>();
}

// The overall loss function using the surrogate model. Clad differentiates this.
float loss_surrogate(const CircuitParams& params, const float freq_kHz, const float target_attenuation) {
  auto prediction = run_surrogate(params, freq_kHz);
  return (prediction - target_attenuation) * (prediction - target_attenuation);  // MSE
}

// The loss function using the true physics model, for verification.
float loss_real(const CircuitParams& params, const float freq_kHz, const float target_attenuation) {
  auto prediction = true_rc_filter_model(freq_kHz, params.R_kOhms, params.C_nF);
  return (prediction - target_attenuation) * (prediction - target_attenuation);  // MSE
}

int main() {
  // =================================================================================
  // Part 1: Train the surrogate model
  // =================================================================================
  std::cout << "--- Part 1: Training surrogate model for RC Filter ---" << std::endl;
  torch::optim::Adam optimizer(surrogate_net.parameters(), torch::optim::AdamOptions(1e-3));

  int n_samples = 5000;
  // By choosing sensible units (kHz, kOhms, nF), the numerical values
  // for our inputs are all in a similar, stable range (e.g., 1-100).
  // This avoids the need for an explicit normalization step.
  torch::Tensor train_freq = torch::rand({n_samples, 1}, torch::kFloat32) * 99.0f + 1.0f;  // 1 to 100 kHz
  torch::Tensor train_R = torch::rand({n_samples, 1}, torch::kFloat32) * 9.9f + 0.1f;      // 0.1 to 10 kOhms
  torch::Tensor train_C = torch::rand({n_samples, 1}, torch::kFloat32) * 99.0f + 1.0f;     // 1 to 100 nF

  torch::Tensor train_x = torch::cat({train_freq, train_R, train_C}, 1);

  // Generate training labels from the "true" simulation
  std::vector<float> y_vec;
  y_vec.reserve(n_samples);
  for (int i = 0; i < n_samples; ++i) {
    y_vec.push_back(
        true_rc_filter_model(train_freq[i].item<float>(), train_R[i].item<float>(), train_C[i].item<float>()));
  }
  torch::Tensor train_y = torch::from_blob(y_vec.data(), {n_samples, 1}, torch::kFloat32).clone();

  // Training loop
  for (int epoch = 0; epoch < 3000; ++epoch) {
    optimizer.zero_grad();
    torch::Tensor prediction = surrogate_net.forward(train_x);
    torch::Tensor loss = torch::nn::functional::mse_loss(prediction, train_y);
    loss.backward();
    optimizer.step();
    if ((epoch + 1) % 1000 == 0) {
      std::cout << "Epoch [" << epoch + 1 << "/3000], Loss: " << loss.item<float>() << std::endl;
    }
  }
  std::cout << "--- Training Complete ---" << std::endl;

  // =================================================================================
  // Part 2: Differentiate the circuit parameters using Clad
  // =================================================================================
  std::cout << "\n--- Part 2: Differentiating circuit parameters with Clad ---" << std::endl;

  // Our component choice we want to find gradients for.
  CircuitParams params = {1.0f, 33.0f};  // 1 kOhm, 33 nF

  // Our design goal: At 3kHz, we want attenuation to be exactly 0.707 (-3dB point)
  const float freq_point_kHz = 3.0f;
  const float target_attenuation = 0.707f;

  // Structures to store the resulting gradients
  CircuitParams d_params_surrogate = {0.0f, 0.0f};
  CircuitParams d_params_real = {0.0f, 0.0f};

  // Differentiate the loss function using the surrogate
  auto grad_func_surrogate = clad::gradient(loss_surrogate, "0");
  grad_func_surrogate.execute(params, freq_point_kHz, target_attenuation, &d_params_surrogate);

  // Differentiate the loss function using the true physics model for comparison
  auto grad_func_real = clad::gradient(loss_real, "0");
  grad_func_real.execute(params, freq_point_kHz, target_attenuation, &d_params_real);

  // =================================================================================
  // Part 3: Verification
  // =================================================================================
  std::cout << "\n--- Part 3: Verification ---" << std::endl;
  std::cout.precision(6);
  std::cout << "Current Parameters: R=" << params.R_kOhms << " kOhms, C=" << params.C_nF << " nF" << std::endl;

  float real_attenuation = true_rc_filter_model(freq_point_kHz, params.R_kOhms, params.C_nF);
  std::cout << "True Attenuation at " << freq_point_kHz << " kHz: " << real_attenuation << std::endl;
  float surrogate_attenuation = run_surrogate(params, freq_point_kHz);
  std::cout << "Surrogate Attenuation at " << freq_point_kHz << " kHz: " << surrogate_attenuation << std::endl;
  std::cout << "Target Attenuation: " << target_attenuation << std::endl;

  std::cout << "\n--- Gradient Comparison ---" << std::endl;
  std::cout << "dLoss/dR (via Surrogate): " << d_params_surrogate.R_kOhms << std::endl;
  std::cout << "dLoss/dR (Analytical):    " << d_params_real.R_kOhms << std::endl;
  std::cout << std::endl;
  std::cout << "dLoss/dC (via Surrogate): " << d_params_surrogate.C_nF << std::endl;
  std::cout << "dLoss/dC (Analytical):    " << d_params_real.C_nF << std::endl;

  return 0;
}
