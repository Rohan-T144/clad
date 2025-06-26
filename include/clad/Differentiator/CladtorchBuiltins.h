#include <clad/Differentiator/STLBuiltins.h>
#include <cladtorch/simpletorch.hpp>
// #include <cladtorch/statictorch.hpp>

namespace clad {
// specialize the zero_init function for Tensor
template <typename T> void zero_init(cladtorch::Tensor<T>& tensor) {
  std::cerr << "==========================================================================" << ::std::endl;
  std::cerr << "Zero initializing tensor with shape: ";
  tensor.print();
  std::cerr << "==========================================================================" << ::std::endl;
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

void operator_plus_equal_pullback(::cladtorch::Tensor<float>* _this, const ::cladtorch::Tensor<float>& other, ::cladtorch::Tensor<float> _d_y,
                                  ::cladtorch::Tensor<float>* _d_this, ::cladtorch::Tensor<float>* _d_other) {
  // For +=, _d_y flows to _d_this
  *_d_this += _d_y;
    *_d_other += _d_y;

}

void operator_plus_pullback(const ::cladtorch::Tensor<float>* _this, const ::cladtorch::Tensor<float> &other, ::cladtorch::Tensor<float> _d_y, ::cladtorch::Tensor<float> *_d_this, ::cladtorch::Tensor<float> *_d_other) {
  // For +, gradient flows to both operands
  *_d_this += _d_y;
    *_d_other += _d_y;

}

// Subtraction operators
void operator_minus_equal_pullback(::cladtorch::Tensor<float>* _this, const ::cladtorch::Tensor<float>& other, ::cladtorch::Tensor<float> _d_y,
                                   ::cladtorch::Tensor<float>* _d_this, ::cladtorch::Tensor<float>* _d_other) {
  // For -=, _d_y flows to _d_this
  *_d_this += _d_y;
  *_d_other -= _d_y; // Negate gradient for _d_other
}

void operator_minus_pullback(const ::cladtorch::Tensor<float>* _this, const ::cladtorch::Tensor<float> &other, ::cladtorch::Tensor<float> _d_y, ::cladtorch::Tensor<float> *_d_this, ::cladtorch::Tensor<float> *_d_other) {
  // For -, gradient flows to first operand as-is
  *_d_this += _d_y;
  *_d_other -= _d_y; // Negate gradient for second operand
}

// Multiplication operators  
void operator_star_equal_pullback(::cladtorch::Tensor<float>* _this, const ::cladtorch::Tensor<float>& other, ::cladtorch::Tensor<float> _d_y,
                                      ::cladtorch::Tensor<float>* _d_this, ::cladtorch::Tensor<float>* _d_other) {
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

void operator_star_pullback(const ::cladtorch::Tensor<float>* _this, const ::cladtorch::Tensor<float> &other, ::cladtorch::Tensor<float> _d_y, ::cladtorch::Tensor<float> *_d_this, ::cladtorch::Tensor<float> *_d_other) {
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
  auto original_this = current_this / scalar;  // Reconstruct original value
  auto grad_scalar_tensor = _d_y * original_this;
  
  float grad_scalar_sum = 0;
  for (int i = 0; i < grad_scalar_tensor.num_elements(); ++i) {
    grad_scalar_sum += grad_scalar_tensor.data()[i];
  }
  *_d_scalar += grad_scalar_sum;
}

void operator_star_pullback(const ::cladtorch::Tensor<float>* _this, float scalar, ::cladtorch::Tensor<float> _d_y, ::cladtorch::Tensor<float> *_d_this, float *_d_scalar) {
  // For tensor * scalar
  auto grad_this = _d_y * scalar;
  *_d_this += grad_this;
  
  // For scalar gradient, sum all elements of (_d_y * _this)
  auto grad_scalar_tensor = _d_y * (*_this);
  
  float grad_scalar_sum = 0;
  for (int i = 0; i < grad_scalar_tensor.num_elements(); ++i) {
    grad_scalar_sum += grad_scalar_tensor.data()[i];
  }
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
  auto original_this = current_this * scalar;  // Reconstruct original value
  auto grad_scalar_tensor = _d_y * original_this;
  
  float grad_scalar_sum = 0;
  for (int i = 0; i < grad_scalar_tensor.num_elements(); ++i) {
    grad_scalar_sum += grad_scalar_tensor.data()[i];
  }
  *_d_scalar += -grad_scalar_sum / (scalar * scalar);
}

void operator_divide_pullback(const ::cladtorch::Tensor<float>* _this, float scalar, ::cladtorch::Tensor<float> _d_y, ::cladtorch::Tensor<float> *_d_this, float *_d_scalar) {
  // For tensor / scalar
  auto grad_this = _d_y / scalar;
  *_d_this += grad_this;
  
  // For scalar gradient: d_scalar = -sum(_d_y * _this) / (scalar^2)
  auto grad_scalar_tensor = _d_y * (*_this);
  
  float grad_scalar_sum = 0;
  for (int i = 0; i < grad_scalar_tensor.num_elements(); ++i) {
    grad_scalar_sum += grad_scalar_tensor.data()[i];
  }
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
  *_d_other += *_d_this; // Assuming we want to accumulate the adjoint
  _d_this->fill(0);      // Assuming we want to zero out the adjoint for this tensor
}
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad