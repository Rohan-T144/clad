#ifndef CLAD_TENSOR_HPP_DYNAMIC
#define CLAD_TENSOR_HPP_DYNAMIC

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list> // For at()
#include <iostream>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Simple assertion macro
#define CLAD_ASSERT(condition, message) assert((condition) && message)

namespace cladtorch {

// -------------------- Kernel Functions (Operating on raw pointers) --------------------
// These functions are low-level, high-performance routines that operate on raw C-style
// arrays. They remain unchanged as their performance and interface are independent of
// how the Tensor class manages its data.
namespace kernels {

inline float gelu_kernel(float x) {
  return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

inline void softmax_kernel(const float* logits, float* probs, int size) {
  if (size <= 0)
    return;
  float max_logit = logits[0];
  for (int i = 1; i < size; ++i)
    if (logits[i] > max_logit)
      max_logit = logits[i];

  float sum_exp = 0.0f;
  for (int i = 0; i < size; ++i) {
    probs[i] = std::exp(logits[i] - max_logit);
    sum_exp += probs[i];
  }

  if (sum_exp == 0.0f)
    sum_exp = 1e-9f;
  for (int i = 0; i < size; ++i)
    probs[i] /= sum_exp;
}

inline float cross_entropy_loss_kernel(const float* probs, int target_class, int size) {
  if (target_class < 0 || target_class >= size)
    return -std::log(1e-9f);
  float prob_at_target = probs[target_class];
  return -std::log(std::max(prob_at_target, 1e-9f));
}

inline void mat_vec_mul_kernel(const float* mat, const float* vec, float* result, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    result[i] = 0.0f;
    for (int j = 0; j < cols; ++j)
      result[i] += mat[i * cols + j] * vec[j];
  }
}

inline void mat_mul_kernel(const float* a_data, const float* b_data, float* result_data, size_t R, size_t C1,
                           size_t C2) {
  for (size_t i = 0; i < R; ++i) {
    for (size_t j = 0; j < C2; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < C1; ++k)
        sum += a_data[i * C1 + k] * b_data[k * C2 + j];
      result_data[i * C2 + j] = sum;
    }
  }
}

inline void batched_mat_mul_kernel(const float* a_data, const float* b_data, float* result_data, size_t batch_size,
                                   size_t R, size_t C1, size_t C2) {
  size_t a_batch_stride = R * C1;
  size_t b_batch_stride = C1 * C2;
  size_t result_batch_stride = R * C2;

  for (size_t batch = 0; batch < batch_size; ++batch) {
    const float* a_batch = a_data + batch * a_batch_stride;
    const float* b_batch = b_data + batch * b_batch_stride;
    float* result_batch = result_data + batch * result_batch_stride;
    mat_mul_kernel(a_batch, b_batch, result_batch, R, C1, C2);
  }
}

inline void element_wise_add_kernel(const float* a, const float* b, float* r, size_t n) {
  for (size_t i = 0; i < n; ++i)
    r[i] = a[i] + b[i];
}
inline void element_wise_sub_kernel(const float* a, const float* b, float* r, size_t n) {
  for (size_t i = 0; i < n; ++i)
    r[i] = a[i] - b[i];
}
inline void element_wise_mul_kernel(const float* a, const float* b, float* r, size_t n) {
  for (size_t i = 0; i < n; ++i)
    r[i] = a[i] * b[i];
}
inline void scalar_mul_kernel(const float* in, float s, float* r, size_t n) {
  for (size_t i = 0; i < n; ++i)
    r[i] = in[i] * s;
}
inline void scalar_div_kernel(const float* in, float s, float* r, size_t n) {
  CLAD_ASSERT(s != 0.0f, "Division by zero.");
  for (size_t i = 0; i < n; ++i)
    r[i] = in[i] / s;
}

template <typename T>
inline void lookup_kernel(const T* src_data, const int* indices, T* dst_data, 
                         size_t num_indices, size_t src_first_dim, size_t slice_size) {
  for (size_t i = 0; i < num_indices; ++i) {
    int idx = indices[i];
    CLAD_ASSERT(idx >= 0 && idx < (int)src_first_dim, "Index out of bounds in lookup.");
    
    const T* src_slice = src_data + idx * slice_size;
    T* dst_slice = dst_data + i * slice_size;
    
    for (size_t j = 0; j < slice_size; ++j) {
      dst_slice[j] = src_slice[j];
    }
  }
}

inline float vec_mean_kernel(size_t vec_size, const float* src) {
  float sum = 0.0f;
  for (size_t i = 0; i < vec_size; i++) {
    sum += src[i];
  }
  return sum / vec_size;
}

inline float vec_rstd_kernel(size_t vec_size, const float* src, float mean) {
  float eps = 1e-5f;
  float sum = 0.0f;
  for (size_t i = 0; i < vec_size; i++) {
    float diff = src[i] - mean;
    sum += diff * diff;
  }
  float var = sum / vec_size;
  return 1.0f / std::sqrt(var + eps);
}

inline void norm_kernel(const float* src_data, float* dst_data, size_t num_vectors, size_t vec_size) {
  for (size_t idx = 0; idx < num_vectors; ++idx) {
    const float* vec = src_data + idx * vec_size;
    float* out = dst_data + idx * vec_size;
    
    // Calculate the mean and the rstd (without bias correction)
    float mean = vec_mean_kernel(vec_size, vec);
    float rstd = vec_rstd_kernel(vec_size, vec, mean);
    
    for (size_t i = 0; i < vec_size; i++) {
      out[i] = (vec[i] - mean) * rstd;
    }
  }
}

} // namespace kernels

// -------------------- Dynamic-Shape Tensor Class --------------------
template <typename T> class Tensor {
public:
  std::vector<size_t> _shape;
  std::vector<size_t> _strides;
  size_t _num_elements = 0;
  T* _data = nullptr;

private:
  // Private helper to initialize tensor metadata and allocate memory
  void init_from_shape(const std::vector<size_t>& shape) {
    _shape = shape;
    _num_elements = 1;
    bool has_zero_dim = false;
    for (size_t dim : _shape) {
      if (dim == 0)
        has_zero_dim = true;
      _num_elements *= dim;
    }
    if (has_zero_dim)
      _num_elements = 0;

    if (_shape.empty()) {
      _strides.clear();
    } else {
      _strides.resize(_shape.size());
      _strides.back() = 1;
      for (long i = _shape.size() - 2; i >= 0; --i)
        _strides[i] = _strides[i + 1] * _shape[i + 1];
    }
    _data = _num_elements > 0 ? new T[_num_elements] : nullptr;
  }

public:
  // --- Constructors, Destructor, Assignment ---

  // Default constructor: creates an empty tensor.
  Tensor() : _num_elements(0), _data(nullptr) {}

  // Shape constructor: creates a tensor with a given shape, zero-initialized.
  explicit Tensor(const std::vector<size_t>& shape) {
    init_from_shape(shape);
    if (_data)
      std::fill(_data, _data + _num_elements, T{});
  }

  explicit Tensor(const std::vector<size_t>& shape, const T* data) {
    init_from_shape(shape);
    if (!_data)
      std::fill(_data, _data + _num_elements, T{});
    else
      std::copy(data, data + _num_elements, _data);
  }

  // Shape and value constructor: creates a tensor filled with a scalar value.
  Tensor(const std::vector<size_t>& shape, T val) {
    init_from_shape(shape);
    fill(val);
  }

  // Scalar constructor: creates a 0-dimensional tensor.
  static Tensor<T> new_scalar(T scalar_val) {
    Tensor<T> tensor;
    tensor._shape = {};
    tensor._strides = {};
    tensor._num_elements = 1;
    tensor._data = new T[1];
    tensor._data[0] = scalar_val;
    return tensor;
  }
  // explicit Tensor(T scalar_val) : _num_elements(1), _data(new T[1]) {
  //     // _shape and _strides are empty for a scalar.
  //     _data[0] = scalar_val;
  // }

  ~Tensor() {
    if (_data) {
      delete[] _data;
      _data = nullptr;
    }
  }

  // Copy constructor
  Tensor(const Tensor& other)
      : _shape(other._shape), _strides(other._strides), _num_elements(other._num_elements), _data(nullptr) {
    if (_num_elements > 0) {
      _data = new T[_num_elements];
      std::copy(other._data, other._data + _num_elements, _data);
    }
  }

  // Copy assignment
  Tensor& operator=(const Tensor& other) {
    if (this != &other) {
      delete[] _data;
      _shape = other._shape;
      _strides = other._strides;
      _num_elements = other._num_elements;
      _data = nullptr;
      if (_num_elements > 0) {
        _data = new T[_num_elements];
        std::copy(other._data, other._data + _num_elements, _data);
      }
    }
    return *this;
  }

  // Move constructor
  Tensor(Tensor&& other) noexcept
      : _shape(std::move(other._shape)), _strides(std::move(other._strides)), _num_elements(other._num_elements),
        _data(other._data) {
    other._num_elements = 0;
    other._data = nullptr;
  }

  // Move assignment
  Tensor& operator=(Tensor&& other) noexcept {
    if (this != &other) {
      delete[] _data;
      _shape = std::move(other._shape);
      _strides = std::move(other._strides);
      _num_elements = other._num_elements;
      _data = other._data;
      other._num_elements = 0;
      other._data = nullptr;
    }
    return *this;
  }

  // --- Accessors & Utilities ---
  const std::vector<size_t>& shape() const { return _shape; }
  size_t ndim() const { return _shape.size(); }
  size_t num_elements() const { return _num_elements; }
  size_t size(size_t dim) const {
    CLAD_ASSERT(dim < _shape.size(), "Dimension index out of range.");
    return _shape[dim];
  }
  T* data() { return _data; }
  const T* data() const { return _data; }

  template <typename... IdxTypes> T& at(IdxTypes... idx_values) {
    CLAD_ASSERT(sizeof...(idx_values) == ndim(), "Number of indices must match tensor dimensions.");
    if (ndim() == 0) {
      CLAD_ASSERT(sizeof...(idx_values) == 0, "Do not provide indices for a scalar tensor.");
      return _data[0];
    }

    size_t indices[] = {static_cast<size_t>(idx_values)...};
    size_t flat_index = 0;
    for (size_t i = 0; i < ndim(); ++i) {
      CLAD_ASSERT(indices[i] < _shape[i], "Index out of bounds.");
      flat_index += indices[i] * _strides[i];
    }
    return _data[flat_index];
  }

  template <typename... IdxTypes> const T& at(IdxTypes... idx_values) const {
    return const_cast<Tensor*>(this)->at(idx_values...);
  }

  // Convenience for scalar tensors
  T& scalar() {
    CLAD_ASSERT(ndim() == 0, "scalar() is only for 0-dimension tensors.");
    CLAD_ASSERT(_data != nullptr, "Accessing null data in scalar tensor.");
    return _data[0];
  }

  const T& scalar() const {
    CLAD_ASSERT(ndim() == 0, "scalar() is only for 0-dimension tensors.");
    CLAD_ASSERT(_data != nullptr, "Accessing null data in scalar tensor.");
    return _data[0];
  }

  void fill(T value) {
    if (_num_elements > 0) {
      CLAD_ASSERT(_data != nullptr, "Filling null data tensor.");
      std::fill(_data, _data + _num_elements, value);
    }
  }

  void print(const std::string& title = "") const {
    if (!title.empty())
      std::cout << title;
    std::cout << " (Shape: [";
    for (size_t i = 0; i < _shape.size(); ++i)
      std::cout << _shape[i] << (i == _shape.size() - 1 ? "" : ", ");
    std::cout << "], NumElements: " << _num_elements << ")" << std::endl;

    if (_num_elements == 0) {
      std::cout << "(Empty)\n";
      return;
    }
    CLAD_ASSERT(_data, "Printing null data tensor.");

    if (ndim() == 0) {
      std::cout << _data[0] << std::endl;
    } else if (ndim() == 1) {
      for (size_t i = 0; i < size(0); ++i)
        std::cout << at(i) << " ";
      std::cout << "\n";
    } else if (ndim() == 2) {
      for (size_t i = 0; i < size(0); ++i) {
        for (size_t j = 0; j < size(1); ++j)
          std::cout << at(i, j) << " ";
        std::cout << "\n";
      }
    } else {
      std::cout << "[";
      for (size_t i = 0; i < std::min((size_t)10, _num_elements); ++i)
        std::cout << _data[i] << (i < 9 && i < _num_elements - 1 ? ", " : "");
      if (_num_elements > 10)
        std::cout << "...";
      std::cout << "]\n";
    }
  }

  // Lookup operation: select slices from this tensor using indices
  Tensor<T> lookup(const Tensor<int>& indices) const {
    CLAD_ASSERT(ndim() > 0, "Cannot lookup from a scalar tensor.");
    CLAD_ASSERT(_data != nullptr, "Cannot lookup from null data tensor.");
    
    // Calculate the size of each slice (everything after the first dimension)
    size_t slice_size = 1;
    for (size_t i = 1; i < ndim(); ++i) {
      slice_size *= _shape[i];
    }
    
    // Create result shape: [indices.num_elements(), remaining dimensions...]
    std::vector<size_t> result_shape;
    result_shape.push_back(indices.num_elements());
    for (size_t i = 1; i < ndim(); ++i) {
      result_shape.push_back(_shape[i]);
    }
    
    Tensor<T> result(result_shape);
    
    if (indices.num_elements() > 0) {
      kernels::lookup_kernel(_data, indices.data(), result.data(), 
                           indices.num_elements(), _shape[0], slice_size);
    }
    
    return result;
  }

  // Layer normalization: normalizes along the last dimension
  Tensor<T> norm() const {
    static_assert(std::is_same_v<T, float>, "norm() is only supported for float tensors.");
    CLAD_ASSERT(ndim() > 0, "Cannot normalize a scalar tensor.");
    CLAD_ASSERT(_data != nullptr, "Cannot normalize null data tensor.");
    
    Tensor<T> result(_shape);
    
    if (_num_elements == 0) {
      return result;
    }
    
    // Calculate number of vectors and vector size
    size_t vec_size = _shape.back(); // Last dimension
    size_t num_vectors = _num_elements / vec_size;
    
    kernels::norm_kernel(_data, result.data(), num_vectors, vec_size);
    
    return result;
  }

  // --- Operator Overloads ---
  Tensor& operator+=(const Tensor& other) {
    CLAD_ASSERT(_shape == other._shape, "Element-wise addition requires identical shapes.");
    kernels::element_wise_add_kernel(_data, other._data, _data, _num_elements);
    return *this;
  }
  Tensor operator+(const Tensor& other) const { return Tensor(*this) += other; }

  Tensor& operator-=(const Tensor& other) {
    CLAD_ASSERT(_shape == other._shape, "Element-wise subtraction requires identical shapes.");
    kernels::element_wise_sub_kernel(_data, other._data, _data, _num_elements);
    return *this;
  }
  Tensor operator-(const Tensor& other) const { return Tensor(*this) -= other; }

  Tensor& operator*=(const Tensor& other) {
    CLAD_ASSERT(_shape == other._shape, "Element-wise multiplication requires identical shapes.");
    kernels::element_wise_mul_kernel(_data, other._data, _data, _num_elements);
    return *this;
  }
  Tensor operator*(const Tensor& other) const { return Tensor(*this) *= other; }

  Tensor& operator*=(T scalar) {
    kernels::scalar_mul_kernel(_data, scalar, _data, _num_elements);
    return *this;
  }
  Tensor operator*(T scalar) const { return Tensor(*this) *= scalar; }

  Tensor& operator/=(T scalar) {
    kernels::scalar_div_kernel(_data, scalar, _data, _num_elements);
    return *this;
  }
  Tensor operator/(T scalar) const { return Tensor(*this) /= scalar; }
};

// -------------------- Tensor Operations (Wrappers around Kernels) --------------------

template <typename T> Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
  // Case 1: Batched Matrix Multiplication (3D x 3D)
  if (a.ndim() == 3 && b.ndim() == 3) {
    CLAD_ASSERT(a.size(0) == b.size(0), "Batch dimension must be the same for batched matmul.");
    CLAD_ASSERT(a.size(2) == b.size(1), "Inner dimensions must match for batched matmul (a.shape[2] == b.shape[1]).");
    size_t B = a.size(0), R = a.size(1), C1 = a.size(2), C2 = b.size(2);
    Tensor<T> result({B, R, C2});
    kernels::batched_mat_mul_kernel(a.data(), b.data(), result.data(), B, R, C1, C2);
    return result;
  }
  // Case 2: Matrix-Matrix Multiplication (2D x 2D)
  if (a.ndim() == 2 && b.ndim() == 2) {
    CLAD_ASSERT(a.size(1) == b.size(0), "Inner dimensions must match for matmul (a.shape[1] == b.shape[0]).");
    size_t R = a.size(0), C1 = a.size(1), C2 = b.size(1);
    Tensor<T> result({R, C2});
    kernels::mat_mul_kernel(a.data(), b.data(), result.data(), R, C1, C2);
    return result;
  }
  // Case 3: Matrix-Vector Multiplication (2D x 1D)
  if (a.ndim() == 2 && b.ndim() == 1) {
    CLAD_ASSERT(a.size(1) == b.size(0), "Inner dimensions must match for mat-vec mul (a.shape[1] == b.shape[0]).");
    size_t R = a.size(0), C = a.size(1);
    Tensor<T> result({R});
    kernels::mat_vec_mul_kernel(a.data(), b.data(), result.data(), R, C);
    return result;
  }

  CLAD_ASSERT(false, "Unsupported shapes for matmul operation.");
  return Tensor<T>(); // Should not be reached
}

template <typename T> Tensor<T> softmax(const Tensor<T>& input) {
  CLAD_ASSERT(input.ndim() > 0, "Softmax requires at least one dimension.");
  Tensor<T> result(input.shape());

  size_t last_dim = input.shape().back();
  size_t num_vectors = input.num_elements() / last_dim;

  for (size_t i = 0; i < num_vectors; ++i) {
    const T* logits_slice = input.data() + i * last_dim;
    T* probs_slice = result.data() + i * last_dim;
    kernels::softmax_kernel(logits_slice, probs_slice, last_dim);
  }
  return result;
}

// Batched cross-entropy loss
template <typename T> Tensor<T> cross_entropy_loss(const Tensor<T>& probs, const std::vector<int>& targets) {
  CLAD_ASSERT(probs.ndim() == 2, "Probs tensor must be 2D for batched cross entropy loss.");
  size_t batch_size = probs.size(0);
  size_t num_classes = probs.size(1);
  CLAD_ASSERT(batch_size == targets.size(), "Batch size of probs and targets must match.");

  float total_loss = 0.0f;
  for (size_t i = 0; i < batch_size; ++i) {
    const T* prob_slice = probs.data() + i * num_classes;
    total_loss += kernels::cross_entropy_loss_kernel(prob_slice, targets[i], num_classes);
  }
  return Tensor<T>(total_loss / batch_size); // Return mean loss as a scalar tensor
}

// Single-instance cross-entropy loss
template <typename T> Tensor<T> cross_entropy_loss(const Tensor<T>& probs, int target_class) {
  CLAD_ASSERT(probs.ndim() == 1, "Probs tensor must be 1D for single cross entropy loss.");
  float loss_val = kernels::cross_entropy_loss_kernel(probs.data(), target_class, probs.num_elements());
  return Tensor<T>::new_scalar(loss_val); // Return loss as a scalar tensor
}

template <typename T> Tensor<T> gelu(const Tensor<T>& in) {
  Tensor<T> r(in.shape());
  for (size_t i = 0; i < in.num_elements(); ++i)
    r.data()[i] = kernels::gelu_kernel(in.data()[i]);
  return r;
}

template <typename T> Tensor<T> lookup(const Tensor<T>& src, const Tensor<int>& indices) {
  return src.lookup(indices);
}

template <typename T> Tensor<T> norm(const Tensor<T>& input) {
  return input.norm();
}

} // namespace cladtorch

#endif // CLAD_TENSOR_HPP_DYNAMIC
