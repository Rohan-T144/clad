#ifndef CLAD_TENSOR_HPP_STATIC
#define CLAD_TENSOR_HPP_STATIC

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#define CLAD_ASSERT(condition, message) assert((condition) && message)

namespace cladtorch {

// -------------------- Kernel Functions (Operating on raw pointers) --------------------
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

// --- Matrix Multiplication Kernel ---
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

// --- Batched Matrix Multiplication Kernel ---
inline void batched_mat_mul_kernel(const float* a_data, const float* b_data, float* result_data, 
                                   size_t batch_size, size_t R, size_t C1, size_t C2) {
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

// --- Element-wise and Scalar Kernels ---
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
  CLAD_ASSERT(s != 0.0f, "Div by zero");
  for (size_t i = 0; i < n; ++i)
    r[i] = in[i] / s;
}

} // namespace kernels

// -------------------- Compile-Time Tensor Class --------------------
template <typename T, size_t... Dims> class Tensor {
public:
  static constexpr size_t NDim = sizeof...(Dims);
  static_assert(NDim == 0 || ((Dims > 0) && ...), "All tensor dimensions must be positive.");

private:
  static constexpr size_t calculate_num_elements() { return (NDim == 0) ? 1 : (Dims * ... * 1); }

public:
  static constexpr size_t NumElements = calculate_num_elements();
  static constexpr std::array<size_t, NDim> Shape = {Dims...};

private:
  static constexpr std::array<size_t, NDim> calculate_strides() {
    if constexpr (NDim == 0)
      return {};
    else {
      std::array<size_t, NDim> s{};
      s[NDim - 1] = 1;
      for (long i = NDim - 2; i >= 0; --i)
        s[i] = s[i + 1] * Shape[i + 1];
      return s;
    }
  }

public:
  static constexpr std::array<size_t, NDim> Strides = calculate_strides();
  T* _data;

  // --- Constructors, Destructor, Assignment ---
  Tensor() : _data(NumElements > 0 ? new T[NumElements]{} : nullptr) {}
  explicit Tensor(T val) : _data(NumElements > 0 ? new T[NumElements] : nullptr) {
    if (_data)
      std::fill(_data, _data + NumElements, val);
  }
  explicit Tensor(const T* data) : _data(NumElements > 0 ? new T[NumElements] : nullptr) {
    if (_data)
      std::copy(data, data + NumElements, _data);
  }
  template <typename U = T, typename = std::enable_if_t<std::is_same_v<U, T> && NDim == 1>>
  Tensor(std::initializer_list<T> il) : _data(NumElements > 0 ? new T[NumElements] : nullptr) {
    CLAD_ASSERT(il.size() == Shape[0], "Initializer list size mismatch.");
    if (_data)
      std::copy(il.begin(), il.end(), _data);
  }
  ~Tensor() { delete[] _data; }
  Tensor(const Tensor& other) : _data(NumElements > 0 ? new T[NumElements] : nullptr) {
    if (_data)
      std::copy(other._data, other._data + NumElements, _data);
  }
  Tensor& operator=(const Tensor& other) {
    if (this != &other) {
      if (!_data)
        _data = new T[NumElements];
      std::copy(other._data, other._data + NumElements, _data);
    }
    return *this;
  }
  Tensor(Tensor&& other) noexcept : _data(other._data) { other._data = nullptr; }
  Tensor& operator=(Tensor&& other) noexcept {
    if (this != &other) {
      delete[] _data;
      _data = other._data;
      other._data = nullptr;
    }
    return *this;
  }

  // --- Accessors & Utilities ---
  // (at, scalar, fill, print, etc. remain the same)
  template <typename... IdxTypes> T& at(IdxTypes... idx_values);             // Declaration
  template <typename... IdxTypes> const T& at(IdxTypes... idx_values) const; // Declaration
  void print(const std::string& title = "") const;                           // Declaration
  
  // Convenience for scalar tensors
  T& scalar() {
    static_assert(NDim == 0, "scalar() method is only for 0-dimension (scalar) tensors.");
    CLAD_ASSERT(_data != nullptr, "Accessing null data in scalar tensor via scalar().");
    return _data[0];
  }

  const T& scalar() const {
    static_assert(NDim == 0, "scalar() method is only for 0-dimension (scalar) tensors.");
    CLAD_ASSERT(_data != nullptr, "Accessing null data in scalar tensor via scalar() const.");
    return _data[0];
  }

  // Fill tensor with a scalar value
  void fill(T value) {
    if constexpr (NumElements > 0) {
      CLAD_ASSERT(_data != nullptr, "Filling null data tensor.");
      std::fill(_data, _data + NumElements, value);
    }
  }
  
  // --- Operator Overloads ---
  Tensor& operator+=(const Tensor& other) {
    kernels::element_wise_add_kernel(_data, other._data, _data, NumElements);
    return *this;
  }
  Tensor operator+(const Tensor& other) const { return Tensor(*this) += other; }
  Tensor& operator-=(const Tensor& other) {
    kernels::element_wise_sub_kernel(_data, other._data, _data, NumElements);
    return *this;
  }
  Tensor operator-(const Tensor& other) const { return Tensor(*this) -= other; }
  Tensor& operator*=(T scalar) {
    kernels::scalar_mul_kernel(_data, scalar, _data, NumElements);
    return *this;
  }
  Tensor operator*(T scalar) const { return Tensor(*this) *= scalar; }
  Tensor& operator/=(T scalar) {
    kernels::scalar_div_kernel(_data, scalar, _data, NumElements);
    return *this;
  }
  Tensor operator/(T scalar) const { return Tensor(*this) /= scalar; }
};

// Implementations for at() and print() outside class body for brevity
template <typename T, size_t... Dims>
template <typename... IdxTypes>
T& Tensor<T, Dims...>::at(IdxTypes... idx_values) {
  static_assert((NDim == 0 && sizeof...(IdxTypes) == 0) || (NDim > 0 && sizeof...(IdxTypes) == NDim),
                "Number of indices must match tensor dimensions.");
  if constexpr (NDim == 0)
    return _data[0];
  else {
    size_t indices[] = {static_cast<size_t>(idx_values)...}, flat_index = 0;
    for (size_t i = 0; i < NDim; ++i)
      flat_index += indices[i] * Strides[i];
    return _data[flat_index];
  }
}
template <typename T, size_t... Dims>
template <typename... IdxTypes>
const T& Tensor<T, Dims...>::at(IdxTypes... idx_values) const {
  return const_cast<Tensor*>(this)->at(idx_values...);
}

template <typename T, size_t... Dims>
void Tensor<T, Dims...>::print(const std::string& title) const { /* Full implementation from previous step */
  if (!title.empty())
    std::cout << title;
  std::cout << " (Shape: [";
  if constexpr (NDim > 0)
    for (size_t i = 0; i < NDim; ++i)
      std::cout << Shape[i] << (i == NDim - 1 ? "" : ", ");
  std::cout << "], NumElements: " << NumElements << ")" << std::endl;
  if constexpr (NumElements == 0) {
    std::cout << "(Empty)\n";
    return;
  }
  CLAD_ASSERT(_data, "Printing null data tensor.");
  if constexpr (NDim == 0) {
    std::cout << _data[0] << std::endl;
  } else if constexpr (NDim == 1) {
    for (size_t i = 0; i < Shape[0]; ++i)
      std::cout << at(i) << " ";
    std::cout << "\n";
  } else if constexpr (NDim == 2) {
    for (size_t i = 0; i < Shape[0]; ++i) {
      for (size_t j = 0; j < Shape[1]; ++j)
        std::cout << at(i, j) << " ";
      std::cout << "\n";
    }
  } else {
    std::cout << "[";
    for (size_t i = 0; i < std::min((size_t)10, NumElements); ++i)
      std::cout << _data[i] << (i < 9 && i < NumElements - 1 ? ", " : "");
    if (NumElements > 10)
      std::cout << "...";
    std::cout << "]\n";
  }
}

// -------------------- Tensor Operations (Wrappers around Kernels) --------------------

// --- Matrix Multiplication (mat1 @ mat2) ---
// 2D x 2D matrix multiplication
template <typename T, size_t R, size_t C1, size_t C2>
Tensor<T, R, C2> matmul(const Tensor<T, R, C1>& a, const Tensor<T, C1, C2>& b) {
  Tensor<T, R, C2> result;
  kernels::mat_mul_kernel(a._data, b._data, result._data, R, C1, C2);
  return result;
}

// Matrix-vector multiplication (2D x 1D -> 1D)
template <typename T, size_t R, size_t C>
Tensor<T, R> matmul(const Tensor<T, R, C>& mat, const Tensor<T, C>& vec) {
  Tensor<T, R> result;
  kernels::mat_vec_mul_kernel(mat._data, vec._data, result._data, R, C);
  return result;
}

// Batched matrix multiplication (3D x 3D -> 3D)
template <typename T, size_t B, size_t R, size_t C1, size_t C2>
Tensor<T, B, R, C2> matmul(const Tensor<T, B, R, C1>& a, const Tensor<T, B, C1, C2>& b) {
  Tensor<T, B, R, C2> result;
  kernels::batched_mat_mul_kernel(a._data, b._data, result._data, B, R, C1, C2);
  return result;
}

// --- Softmax (applied to the last dimension) ---
template <typename T, size_t... Dims> Tensor<T, Dims...> softmax(const Tensor<T, Dims...>& input) {
  static_assert(sizeof...(Dims) > 0, "Softmax requires at least one dimension.");
  Tensor<T, Dims...> result;

  // Get the size of the last dimension using a fold expression
  constexpr size_t LastDim = (std::get<sizeof...(Dims) - 1>(std::make_tuple(Dims...)));
  // Number of vectors to apply softmax to
  constexpr size_t NumVectors = Tensor<T, Dims...>::NumElements / LastDim;

  for (size_t i = 0; i < NumVectors; ++i) {
    const T* logits_slice = input._data + i * LastDim;
    T* probs_slice = result._data + i * LastDim;
    kernels::softmax_kernel(logits_slice, probs_slice, LastDim);
  }
  return result;
}

// Calculates the mean loss over a batch of predictions.
// Probs: A tensor of probability distributions, e.g., shape (BatchSize, NumClasses).
// Targets: A standard array of correct class indices for the batch.
template <typename T, size_t BatchSize, size_t NumClasses>
Tensor<T> cross_entropy_loss(const Tensor<T, BatchSize, NumClasses>& probs, const std::array<int, BatchSize>& targets) {
  float total_loss = 0.0f;
  for (size_t i = 0; i < BatchSize; ++i) {
    const T* prob_slice = probs._data + i * NumClasses;
    total_loss += kernels::cross_entropy_loss_kernel(prob_slice, targets[i], NumClasses);
  }
  // Return the mean loss as a scalar tensor
  return Tensor<T>(total_loss / BatchSize);
}

// Overload for a single prediction (batch size of 1)
template <typename T, size_t NumClasses>
Tensor<T> cross_entropy_loss(const Tensor<T, NumClasses>& probs, int target_class) {
  float loss_val = kernels::cross_entropy_loss_kernel(probs._data, target_class, NumClasses);
  return Tensor<T>(loss_val);
}

template <typename T, size_t... Dims> Tensor<T, Dims...> gelu(const Tensor<T, Dims...>& in) {
  Tensor<T, Dims...> r;
  for (size_t i = 0; i < in.NumElements; ++i)
    r._data[i] = kernels::gelu_kernel(in._data[i]);
  return r;
}

} // namespace cladtorch

#endif // CLAD_TENSOR_HPP_STATIC
