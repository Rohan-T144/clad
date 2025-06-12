#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "clad/Differentiator/Differentiator.h"

// =================================================================================
// The "True" Simulation: A Damped Harmonic Oscillator
// This is the process our NN will learn.
// y(t) = exp(-damping * t) * cos(frequency * t)
// =================================================================================
double true_oscillator_simulation(double t, double damping, double frequency) {
  return std::exp(-damping * t) * std::cos(frequency * t);
}

// =================================================================================
// The Surrogate Model (Neural Network)
// =================================================================================
struct Net : torch::nn::Module {
  Net() {
    // 3 inputs (t, damping, frequency) -> 128 -> 128 -> 1 output (position)
    fc1 = register_module("fc1", torch::nn::Linear(3, 128));
    fc2 = register_module("fc2", torch::nn::Linear(128, 128));
    fc3 = register_module("fc3", torch::nn::Linear(128, 1));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    x = fc3->forward(x);
    return x;
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

Net surrogate_net;  // Global instance of our surrogate model

// =================================================================================
// Struct to hold our differentiable simulation meta-parameters
// =================================================================================
struct SimParams {
  double damping;
  double frequency;
};

// We need to store the normalization statistics from the training set
// to apply them later during differentiation.
struct NormalizationStats {
  torch::Tensor mean;
  torch::Tensor std;
};

NormalizationStats input_stats;  // Global stats object

namespace clad::custom_derivatives {
void call_torch_pullback(const SimParams &params, const double t, const double target_pos, double _d_y,
                         SimParams *_d_params, double *_d_t, double *_d_target_pos) {
  // We need to track gradients w.r.t the parameters' tensor.
  auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);

  // --- 1. Prepare Tensors & NORMALIZE them ---
  torch::Tensor unnormalized_input = torch::tensor({t, params.damping, params.frequency}, torch::kFloat32);
  torch::Tensor normalized_input = (unnormalized_input - input_stats.mean) / input_stats.std;

  // We need to re-attach the parts that require gradients after normalization
  torch::Tensor time_norm = normalized_input[0];
  torch::Tensor damping_norm = normalized_input[1].clone().set_requires_grad(true);
  torch::Tensor freq_norm = normalized_input[2].clone().set_requires_grad(true);

  torch::Tensor final_input = torch::stack({time_norm, damping_norm, freq_norm});

  // --- 2. Forward Pass, Loss, and Backward Pass ---
  torch::Tensor predicted_pos = surrogate_net.forward(final_input);
  torch::Tensor target_tensor = torch::tensor(target_pos, torch::kFloat32);
  torch::Tensor loss = torch::nn::functional::mse_loss(predicted_pos, target_tensor);

  loss.backward();  // {}, /*keep_graph=*/false, /*create_graph=*/false

  // The computed gradients for our meta-parameters are now in their .grad fields.
  _d_params->damping = damping_norm.grad().item<double>() * _d_y;
  _d_params->frequency = freq_norm.grad().item<double>() * _d_y;
}

}  // namespace clad::custom_derivatives

double call_torch(const SimParams &params, const double t, const double target_pos) {
  // This is the non-differentiated version, for inference.
  torch::NoGradGuard no_grad;
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor input_tensor = torch::tensor({t, params.damping, params.frequency}, options);
  double prediction = surrogate_net.forward(input_tensor).item<double>();
  return (prediction - target_pos) * (prediction - target_pos);  // MSE
}

// This is the function we want to differentiate using Clad.
double run_surrogate_and_calc_loss(const SimParams &params, const double t, const double target_pos) {
  return call_torch(params, t, target_pos);
}

int main() {
  // =================================================================================
  // Part 1: Train the surrogate model
  // =================================================================================
  std::cout << "--- Part 1: Training surrogate model ---" << std::endl;
  torch::optim::Adam optimizer(surrogate_net.parameters(), torch::optim::AdamOptions(1e-3));

  int n_samples = 4000;
  torch::Tensor train_t = torch::rand({n_samples, 1}, torch::kFloat32) * 10.0;
  torch::Tensor train_damping = torch::rand({n_samples, 1}, torch::kFloat32) * 0.5;
  torch::Tensor train_freq = torch::rand({n_samples, 1}, torch::kFloat32) * 5.0;

  torch::Tensor train_x_unnormalized = torch::cat({train_t, train_damping, train_freq}, 1);

  input_stats.mean = train_x_unnormalized.mean(/*dim=*/0);
  input_stats.std = train_x_unnormalized.std(/*dim=*/0);
  torch::Tensor train_x = (train_x_unnormalized - input_stats.mean) / input_stats.std;

  // TODO: Training code not yet working.
  // // Generate training labels
  // std::vector<double> y_vec;
  // y_vec.reserve(n_samples);
  // for (int i = 0; i < n_samples; ++i) {
  //     y_vec.push_back(true_oscillator_simulation(
  //         train_t[i].item<double>(), train_damping[i].item<double>(), train_freq[i].item<double>()
  //     ));
  // }
  // torch::Tensor train_y = torch::from_blob(y_vec.data(), {n_samples, 1}, torch::kFloat32).clone();

  // for (int epoch = 0; epoch < 4000; ++epoch) {
  //     optimizer.zero_grad();
  //     torch::Tensor prediction = surrogate_net.forward(train_x);
  //     torch::Tensor loss = torch::nn::functional::mse_loss(prediction, train_y);
  //     loss.backward();
  //     optimizer.step();
  //     if ((epoch + 1) % 1000 == 0) {
  //         std::cout << "Epoch [" << epoch + 1 << "/4000], Loss: " << loss.item<double>() << std::endl;
  //     }
  // }
  // std::cout << "--- Training Complete ---" << std::endl;

  // =================================================================================
  // Part 2: Use Clad to differentiate the loss w.r.t. META-PARAMETERS
  // =================================================================================
  std::cout << "\n--- Part 2: Differentiating meta-parameters with Clad ---" << std::endl;

  // The meta-parameters we want to find gradients for
  SimParams params = {0.1, 2.5};

  // We want to minimize the error at time t=2.0, with a target position of 0.5
  const double time_point = 2.0;
  const double target_position = 0.5;

  // Structure to store the gradients
  SimParams d_params = {0.0, 0.0};

  auto grad_func = clad::gradient(run_surrogate_and_calc_loss, "0");
  grad_func.execute(params, time_point, target_position, &d_params);

  // =================================================================================
  // Part 3: Verification
  // =================================================================================
  std::cout << "\n--- Part 3: Verification ---" << std::endl;
  std::cout.precision(6);
  std::cout << "Meta-parameters: damping=" << params.damping << ", frequency=" << params.frequency << std::endl;

  double loss_at_params = run_surrogate_and_calc_loss(params, time_point, target_position);
  std::cout << "Loss at current parameters: " << loss_at_params << std::endl;

  std::cout << "\nClad computed dLoss/d(damping):   " << d_params.damping << std::endl;
  std::cout << "Clad computed dLoss/d(frequency): " << d_params.frequency << std::endl;
  return 0;
}
