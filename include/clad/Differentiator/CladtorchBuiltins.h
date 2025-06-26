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
  *_d_other += *_d_this;
}
void operator_plus_pullback(const ::cladtorch::Tensor<float>* _this, const ::cladtorch::Tensor<float> &other, ::cladtorch::Tensor<float> _d_y, ::cladtorch::Tensor<float> *_d_this, ::cladtorch::Tensor<float> *_d_other) {
  *_d_this += _d_y;
  *_d_other += _d_y;
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