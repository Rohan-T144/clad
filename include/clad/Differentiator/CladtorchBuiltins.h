#include <clad/Differentiator/STLBuiltins.h>
#include <cladtorch/simpletorch.hpp>
// #include <cladtorch/statictorch.hpp>

namespace clad {
// specialize the zero_init function for Tensor
template <typename T> void zero_init(cladtorch::Tensor<T>& tensor) {
  // std::cerr << "==========================================================================" << ::std::endl;
  // std::cerr << "Zero initializing tensor with shape: ";
  // tensor.print();
  // std::cerr << "==========================================================================" << ::std::endl;
  tensor.fill(0);
}
// template <typename T, size_t... Dims>
// void zero_init(cladtorch::Tensor<T, Dims...>& tensor) {
//   tensor.fill(0);
// std::cerr << "Zero initializing tensor with shape: ";
// tensor.print();
// tensor.fill(0);
// Forward any additional arguments if needed
// This is a placeholder; actual implementation may vary
// }
namespace custom_derivatives {
namespace cladtorch {
// template <typename T>
// void gelu_pullback(const ::cladtorch::Tensor<T> &in, ::cladtorch::Tensor<T> _d_y, ::cladtorch::Tensor<T> *_d_in) {
//   // GELU derivative: dGELU(x) = GELU(x) + x * (1 - GELU(x)^2) * 0.5
//   // where GELU(x) = x * 0.5 * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))

//   // Compute GELU value
//   auto gelu_val = ::cladtorch::gelu(in);

//   // Compute derivative
//   auto d_gelu = gelu_val;
//   // auto d_gelu = gelu_val + in * (((gelu_val * -1) + 1.0f) * gelu_val) * 0.5f;

//   // Apply the gradient
//   *_d_in += _d_y * d_gelu;
// }

// Matrix multiplication pullback
template <typename T>
void matmul_pullback(const ::cladtorch::Tensor<T>& a, const ::cladtorch::Tensor<T>& b, ::cladtorch::Tensor<T> _d_y,
                     ::cladtorch::Tensor<T>* _d_a, ::cladtorch::Tensor<T>* _d_b) {
  // For C = matmul(A, B), the gradients are:
  // dA = matmul(dC, B^T)
  // dB = matmul(A^T, dC)

  // Handle different cases based on tensor dimensions
  if (a.ndim() == 2 && b.ndim() == 2) {
    // Case: 2D x 2D matrix multiplication
    // A: (R, C1), B: (C1, C2), C: (R, C2)
    // dA = matmul(dC, B^T) -> (R, C2) x (C2, C1) = (R, C1)
    // dB = matmul(A^T, dC) -> (C1, R) x (R, C2) = (C1, C2)
    auto b_transposed = b.transpose(0, 1);
    auto a_transposed = a.transpose(0, 1);

    auto grad_a = ::cladtorch::matmul(_d_y, b_transposed);
    auto grad_b = ::cladtorch::matmul(a_transposed, _d_y);

    *_d_a += grad_a;
    *_d_b += grad_b;
  } else if (a.ndim() == 3 && b.ndim() == 2) {
    // Case: 3D x 2D batched matrix multiplication
    // A: (B, T, C), B: (C, out_features), C: (B, T, out_features)
    // dA = matmul(dC, B^T) -> (B, T, out_features) x (out_features, C) = (B, T, C)
    // dB = matmul(A^T, dC) -> sum over batch of (C, T) x (T, out_features) = (C, out_features)

    auto b_transposed = b.transpose(0, 1);
    auto grad_a = ::cladtorch::matmul(_d_y, b_transposed);
    *_d_a += grad_a;

    // For dB, we need to sum contributions from all batch elements
    // Reshape A from (B, T, C) to (B*T, C), then transpose to (C, B*T)
    int batch_size = a.size(0), seq_len = a.size(1), channels = a.size(2);
    auto a_reshaped = a.reshape({batch_size * seq_len, channels});
    auto a_reshaped_transposed = a_reshaped.transpose(0, 1);

    // Reshape dC from (B, T, out_features) to (B*T, out_features)
    auto dy_reshaped = _d_y.reshape({batch_size * seq_len, _d_y.size(2)});

    auto grad_b = ::cladtorch::matmul(a_reshaped_transposed, dy_reshaped);
    *_d_b += grad_b;
  } else if (a.ndim() == 3 && b.ndim() == 3) {
    // Case: 3D x 3D batched matrix multiplication
    // A: (B, R, C1), B: (B, C1, C2), C: (B, R, C2)

    int B = a.size(0);
    for (int batch = 0; batch < B; ++batch) {
      // Extract batch slices - this is a simplified approach
      // In practice, you might want to implement batch-aware transpose and matmul
      // For now, we'll handle this case similarly to 2D but for each batch

      // This is a placeholder - a full implementation would need proper batch slicing
      // For now, we'll use the same logic as 2D case
      auto b_transposed = b.transpose(1, 2);
      auto a_transposed = a.transpose(1, 2);

      auto grad_a = ::cladtorch::matmul(_d_y, b_transposed);
      auto grad_b = ::cladtorch::matmul(a_transposed, _d_y);

      *_d_a += grad_a;
      *_d_b += grad_b;
    }
  } else if (a.ndim() == 4 && b.ndim() == 4) {
    // Case: 4D x 4D multi-head attention matmul
    // A: (B, H, T1, d), B: (B, H, d, T2), C: (B, H, T1, T2)

    // For 4D case, handle batch and head dimensions

    // For each batch and head, compute gradients
    // This is a simplified approach - a full implementation would be more efficient
    auto b_transposed = b.transpose(2, 3); // Transpose last two dimensions
    auto a_transposed = a.transpose(2, 3); // Transpose last two dimensions

    auto grad_a = ::cladtorch::matmul(_d_y, b_transposed);
    auto grad_b = ::cladtorch::matmul(a_transposed, _d_y);

    *_d_a += grad_a;
    *_d_b += grad_b;
  } else if (a.ndim() == 2 && b.ndim() == 1) {
    // Case: 2D x 1D matrix-vector multiplication
    // A: (R, C), B: (C,), C: (R,)
    // dA = outer_product(dC, B) -> (R,) outer (C,) = (R, C)
    // dB = matmul(A^T, dC) -> (C, R) x (R,) = (C,)

    // For dA: outer product of _d_y and b
    auto grad_a = ::cladtorch::Tensor<T>({a.size(0), a.size(1)});
    for (int r = 0; r < a.size(0); ++r)
      for (int c = 0; c < a.size(1); ++c)
        grad_a.at(r, c) = _d_y.at(r) * b.at(c);
    *_d_a += grad_a;

    // For dB: A^T * _d_y
    auto a_transposed = a.transpose(0, 1);
    auto grad_b = ::cladtorch::matmul(a_transposed, _d_y);
    *_d_b += grad_b;
  } else {
    // Unsupported case - should not happen if matmul worked
    assert(false && "Unsupported tensor dimensions for matmul pullback");
  }
}


// Softmax pullback
template <typename T>
void softmax_pullback(const ::cladtorch::Tensor<T>& input, bool is_casual, int vocab_size, ::cladtorch::Tensor<T> _d_y,
                      ::cladtorch::Tensor<T>* _d_input, bool* _d_is_casual, int* _d_vocab_size) {
  // For softmax, if y = softmax(x), then:
  // dy/dx_i = y_i * (delta_ij - y_j) where delta_ij is Kronecker delta
  // This can be written as: dy/dx = y * (grad_y - sum(grad_y * y))
  
  auto softmax_output = ::cladtorch::softmax(input, is_casual, vocab_size);
  
  // Compute sum(grad_y * y) along the last dimension
  int last_dim = input.shape().back();
  int num_vectors = input.num_elements() / last_dim;
  
  for (int vec = 0; vec < num_vectors; ++vec) {
    T sum_grad_y_times_y = 0;
    
    // Calculate the sum for this vector
    for (int i = 0; i < last_dim; ++i) {
      int idx = vec * last_dim + i;
      sum_grad_y_times_y += _d_y.data()[idx] * softmax_output.data()[idx];
    }
    
    // Compute gradient for each element in this vector
    for (int i = 0; i < last_dim; ++i) {
      int idx = vec * last_dim + i;
      T grad = softmax_output.data()[idx] * (_d_y.data()[idx] - sum_grad_y_times_y);
      _d_input->data()[idx] += grad;
    }
  }
  
  // Gradients for bool and int parameters are typically zero for softmax
  // *_d_is_casual remains unchanged (no contribution)
  // *_d_vocab_size remains unchanged (no contribution)
}

// Cross entropy loss pullback for batched version
template <typename T>
void cross_entropy_loss_pullback(const ::cladtorch::Tensor<T>& probs, const ::std::vector<int>& targets, ::cladtorch::Tensor<T> _d_y,
                                 ::cladtorch::Tensor<T>* _d_probs, ::std::vector<int>* _d_targets) {
  // For cross entropy loss L = -log(p_target), the gradient is:
  // dL/dp_i = -1/p_target if i == target, 0 otherwise
  // But since we typically use softmax + cross entropy, the combined gradient is:
  // dL/dx_i = p_i - 1 if i == target, p_i otherwise
  // However, here we only have probs, so: dL/dp_i = -1/p_target if i == target
  
  CLAD_ASSERT(probs.ndim() == 2, "Probs tensor must be 2D for batched cross entropy loss.");
  int batch_size = probs.size(0);
  int num_classes = probs.size(1);
  
  // _d_y is a scalar (the loss), so we need to broadcast its gradient
  T loss_grad = _d_y.scalar(); // Extract scalar value
  T avg_loss_grad = loss_grad / batch_size; // Since we return mean loss
  
  for (int batch = 0; batch < batch_size; ++batch) {
    int target = targets[batch];
    for (int cls = 0; cls < num_classes; ++cls) {
      int idx = batch * num_classes + cls;
      if (cls == target) {
        // Gradient is -1/p_target for the target class
        T prob_val = probs.data()[idx];
        _d_probs->data()[idx] += avg_loss_grad * (-1.0f / prob_val);
      }
      // Gradient is 0 for non-target classes (no addition needed)
    }
  }
  
  // Targets don't have gradients in typical scenarios
  // *_d_targets remains unchanged
}

// Cross entropy loss pullback for single instance version
template <typename T>
void cross_entropy_loss_pullback(const ::cladtorch::Tensor<T>& probs, int target_class, ::cladtorch::Tensor<T> _d_y,
                                 ::cladtorch::Tensor<T>* _d_probs, int* _d_target_class) {
  // For single instance cross entropy loss
  CLAD_ASSERT(probs.ndim() == 1, "Probs tensor must be 1D for single cross entropy loss.");
  int num_classes = probs.num_elements();
  
  T loss_grad = _d_y.scalar(); // Extract scalar value
  
  for (int cls = 0; cls < num_classes; ++cls) {
    if (cls == target_class) {
      // Gradient is -1/p_target for the target class
      T prob_val = probs.data()[cls];
      _d_probs->data()[cls] += loss_grad * (-1.0f / prob_val);
    }
    // Gradient is 0 for non-target classes (no addition needed)
  }
  
  // Target class doesn't have gradients in typical scenarios
  // *_d_target_class remains unchanged
}

} // namespace cladtorch
namespace class_functions {

// Custom derivatives for cladtorch Tensor operations
// template <typename T>
// Tensor& operator=(const Tensor& other) {
//   if (this != &other) {
//     delete[] _data;
//     _shape = other._shape;
//     _strides = other._strides;
//     _num_elements = other._num_elements;
//     _data = nullptr;
//     if (_num_elements > 0) {
//       _data = new T[_num_elements];
//       for (int i = 0; i < _num_elements; ++i) _data[i] = other._data[i];
//       // std::copy(other._data, other._data + _num_elements, _data);
//     }
//   }
//   return *this;
// }

void operator_plus_equal_pullback(::cladtorch::Tensor<float>* _this, const ::cladtorch::Tensor<float>& other,
                                  ::cladtorch::Tensor<float> _d_y, ::cladtorch::Tensor<float>* _d_this,
                                  ::cladtorch::Tensor<float>* _d_other) {
  // For +=, _d_y flows to _d_this
  *_d_this += _d_y;
  *_d_other += _d_y;
}

void operator_plus_pullback(const ::cladtorch::Tensor<float>* _this, const ::cladtorch::Tensor<float>& other,
                            ::cladtorch::Tensor<float> _d_y, ::cladtorch::Tensor<float>* _d_this,
                            ::cladtorch::Tensor<float>* _d_other) {
  // For +, gradient flows to both operands
  *_d_this += _d_y;
  *_d_other += _d_y;
}

// Subtraction operators
void operator_minus_equal_pullback(::cladtorch::Tensor<float>* _this, const ::cladtorch::Tensor<float>& other,
                                   ::cladtorch::Tensor<float> _d_y, ::cladtorch::Tensor<float>* _d_this,
                                   ::cladtorch::Tensor<float>* _d_other) {
  // For -=, _d_y flows to _d_this
  *_d_this += _d_y;
  *_d_other -= _d_y; // Negate gradient for _d_other
}

void operator_minus_pullback(const ::cladtorch::Tensor<float>* _this, const ::cladtorch::Tensor<float>& other,
                             ::cladtorch::Tensor<float> _d_y, ::cladtorch::Tensor<float>* _d_this,
                             ::cladtorch::Tensor<float>* _d_other) {
  // For -, gradient flows to first operand as-is
  *_d_this += _d_y;
  *_d_other -= _d_y; // Negate gradient for second operand
}

// Multiplication operators
void operator_star_equal_pullback(::cladtorch::Tensor<float>* _this, const ::cladtorch::Tensor<float>& other,
                                  ::cladtorch::Tensor<float> _d_y, ::cladtorch::Tensor<float>* _d_this,
                                  ::cladtorch::Tensor<float>* _d_other) {
  // For *=, d_this += d_y * other
  auto grad_this = _d_y * other;
  *_d_this += grad_this;
  assert(0 && "Not implemented yet");
  // For d_other, gradient is d_y * _this (before the operation)
  // Note: we need the original value of _this before the *= operation
  // This is a limitation - we'd need the original value stored
  // For now, assuming _this still contains the result after *=
  // auto current_this = *_this;
  // auto original_this = current_this / other;  // Reconstruct original value
  // auto grad_other = _d_y * original_this;
  // *_d_other += grad_other;
}

void operator_star_pullback(const ::cladtorch::Tensor<float>* _this, const ::cladtorch::Tensor<float>& other,
                            ::cladtorch::Tensor<float> _d_y, ::cladtorch::Tensor<float>* _d_this,
                            ::cladtorch::Tensor<float>* _d_other) {
  // For *, d_this += d_y * other
  auto grad_this = _d_y * other;
  *_d_this += grad_this;

  // For d_other, gradient is d_y * _this
  auto grad_other = _d_y * (*_this);
  *_d_other += grad_other;
}

// Scalar multiplication operators
void operator_star_equal_pullback(::cladtorch::Tensor<float>* _this, float scalar, ::cladtorch::Tensor<float> _d_y,
                                  ::cladtorch::Tensor<float>* _d_this, float* _d_scalar) {
  // For tensor *= scalar
  auto grad_this = _d_y * scalar;
  *_d_this += grad_this;

  // For scalar gradient, sum all elements of (_d_y * original_this)
  auto current_this = *_this;
  auto original_this = current_this / scalar; // Reconstruct original value
  auto grad_scalar_tensor = _d_y * original_this;

  float grad_scalar_sum = 0;
  for (int i = 0; i < grad_scalar_tensor.num_elements(); ++i)
    grad_scalar_sum += grad_scalar_tensor.data()[i];
  *_d_scalar += grad_scalar_sum;
}

void operator_star_pullback(const ::cladtorch::Tensor<float>* _this, float scalar, ::cladtorch::Tensor<float> _d_y,
                            ::cladtorch::Tensor<float>* _d_this, float* _d_scalar) {
  // For tensor * scalar
  auto grad_this = _d_y * scalar;
  *_d_this += grad_this;

  // For scalar gradient, sum all elements of (_d_y * _this)
  auto grad_scalar_tensor = _d_y * (*_this);

  float grad_scalar_sum = 0;
  for (int i = 0; i < grad_scalar_tensor.num_elements(); ++i)
    grad_scalar_sum += grad_scalar_tensor.data()[i];
  *_d_scalar += grad_scalar_sum;
}

// Division operators
void operator_divide_equal_pullback(::cladtorch::Tensor<float>* _this, float scalar, ::cladtorch::Tensor<float> _d_y,
                                    ::cladtorch::Tensor<float>* _d_this, float* _d_scalar) {
  // For tensor /= scalar
  auto grad_this = _d_y / scalar;
  *_d_this += grad_this;

  // For scalar gradient: d_scalar = -sum(_d_y * original_this) / (scalar^2)
  auto current_this = *_this;
  auto original_this = current_this * scalar; // Reconstruct original value
  auto grad_scalar_tensor = _d_y * original_this;

  float grad_scalar_sum = 0;
  for (int i = 0; i < grad_scalar_tensor.num_elements(); ++i)
    grad_scalar_sum += grad_scalar_tensor.data()[i];
  *_d_scalar += -grad_scalar_sum / (scalar * scalar);
}

void operator_divide_pullback(const ::cladtorch::Tensor<float>* _this, float scalar, ::cladtorch::Tensor<float> _d_y,
                              ::cladtorch::Tensor<float>* _d_this, float* _d_scalar) {
  // For tensor / scalar
  auto grad_this = _d_y / scalar;
  *_d_this += grad_this;

  // For scalar gradient: d_scalar = -sum(_d_y * _this) / (scalar^2)
  auto grad_scalar_tensor = _d_y * (*_this);

  float grad_scalar_sum = 0;
  for (int i = 0; i < grad_scalar_tensor.num_elements(); ++i)
    grad_scalar_sum += grad_scalar_tensor.data()[i];
  *_d_scalar += -grad_scalar_sum / (scalar * scalar);
}

template <typename T>
clad::ValueAndPushforward<::cladtorch::Tensor<T>&, ::cladtorch::Tensor<T>&>
operator_equal_pushforward(::cladtorch::Tensor<T>* a, const ::cladtorch::Tensor<T>& param, ::cladtorch::Tensor<T>* d_a,
                           const ::cladtorch::Tensor<T>& d_param) {
  if (a != &param) {
    delete[] a->_data; // Free existing data
    a->_shape = param._shape;
    a->_strides = param._strides;
    a->_num_elements = param._num_elements;
    a->_data = nullptr;
    if (a->_num_elements > 0) {
      a->_data = new T[a->_num_elements];
      ::std::copy(param._data, param._data + a->_num_elements, a->_data);
    }
  }
  if (d_a != &d_param) {
    delete[] d_a->_data; // Free existing data
    d_a->_shape = d_param._shape;
    d_a->_strides = d_param._strides;
    d_a->_num_elements = d_param._num_elements;
    d_a->_data = nullptr;
    if (d_a->_num_elements > 0) {
      d_a->_data = new T[d_a->_num_elements];
      ::std::copy(d_param._data, d_param._data + d_a->_num_elements, d_a->_data);
    }
  }
  return {*a, *d_a};
}

template <typename T>
clad::ValueAndPushforward<::cladtorch::Tensor<T>, ::cladtorch::Tensor<T>>
constructor_pushforward(ConstructorPushforwardTag<::cladtorch::Tensor<T>>, const ::cladtorch::Tensor<T>& p,
                        const ::cladtorch::Tensor<T>& d_p) {
  ::cladtorch::Tensor<T> v(p);
  ::cladtorch::Tensor<T> d_v(d_p);
  return {v, d_v};
}

template <typename T>
void constructor_pullback(const ::cladtorch::Tensor<T>& other, ::cladtorch::Tensor<T>* _d_this,
                          ::cladtorch::Tensor<T>* _d_other) {
  *_d_other += *_d_this;
  _d_this->fill(0);
}
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad