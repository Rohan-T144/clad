#ifndef CLAD_TENSOR_HPP_STATIC
#define CLAD_TENSOR_HPP_STATIC

#include "kernels.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <type_traits>

#define CLAD_ASSERT(condition, message) assert((condition) && message)

namespace cladtorch {

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
  template <typename... IdxTypes> T& at(IdxTypes... idx_values);             // Declaration
  template <typename... IdxTypes> const T& at(IdxTypes... idx_values) const; // Declaration
  void print(const std::string& title = "") const;                           // Declaration

  // Convenience for scalar tensors
  T& scalar() {
    static_assert(NDim == 0, "scalar() method is only for 0-dimension (scalar) tensors.");
    // CLAD_ASSERT(_data != nullptr, "Accessing null data in scalar tensor via scalar().");
    return _data[0];
  }

  const T& scalar() const {
    static_assert(NDim == 0, "scalar() method is only for 0-dimension (scalar) tensors.");
    // CLAD_ASSERT(_data != nullptr, "Accessing null data in scalar tensor via scalar() const.");
    return _data[0];
  }

  // Fill tensor with a scalar value
  void fill(T value) {
    if constexpr (NumElements > 0) {
      CLAD_ASSERT(_data != nullptr, "Filling null data tensor.");
      std::fill(_data, _data + NumElements, value);
    }
  }

  // Data access
  T* data() { return _data; }
  const T* data() const { return _data; }

  // Shape utilities
  static constexpr size_t ndim() { return NDim; }
  static constexpr size_t num_elements() { return NumElements; }
  template <size_t Dim> static constexpr size_t size() {
    static_assert(Dim < NDim, "Dimension index out of bounds.");
    return Shape[Dim];
  }

  // --- Tensor Operations ---

  // Lookup operation (embedding/indexing)
  template <size_t NumIndices, size_t... RestDims>
  Tensor<T, NumIndices, RestDims...> lookup(const std::array<int, NumIndices>& indices) const {
    static_assert(NDim >= 1, "Lookup requires at least 1 dimension.");
    constexpr size_t first_dim = Shape[0];
    constexpr size_t slice_size = NumElements / first_dim;

    Tensor<T, NumIndices, RestDims...> result;
    kernels::lookup_kernel(_data, indices.data(), result._data, NumIndices, first_dim, slice_size);
    return result;
  }

  // Norm operation (LayerNorm)
  Tensor<T, Dims...> norm() const {
    static_assert(NDim >= 1, "Norm requires at least 1 dimension.");
    constexpr size_t last_dim = Shape[NDim - 1];
    constexpr size_t num_vectors = NumElements / last_dim;

    Tensor<T, Dims...> result;
    kernels::norm_kernel(_data, result._data, num_vectors, last_dim);
    return result;
  }

  // View operation (slice along first dimension)
  template <size_t NewFirstDim, size_t... RestDims> Tensor<T, NewFirstDim, RestDims...> view(size_t offset) const {
    static_assert(NDim >= 1, "View requires at least 1 dimension.");
    static_assert(NewFirstDim <= Shape[0], "New first dimension cannot exceed original.");

    constexpr size_t slice_size = NumElements / Shape[0];
    CLAD_ASSERT(offset + NewFirstDim <= Shape[0], "View offset out of bounds.");

    Tensor<T, NewFirstDim, RestDims...> result;

    // Simple copy for contiguous slice
    const T* src_start = _data + offset * slice_size;
    std::copy(src_start, src_start + result.NumElements, result._data);
    return result;
  }

  // Transpose operation (simplified for 2D case)
  // Tensor<T, Shape[1], Shape[0]> transpose() const {
  //   static_assert(NDim == 2, "Transpose currently only supports 2D tensors.");

  //   Tensor<T, Shape[1], Shape[0]> result;

  //   for (size_t i = 0; i < Shape[0]; ++i)
  //     for (size_t j = 0; j < Shape[1]; ++j)
  //       result._data[j * Shape[0] + i] = _data[i * Shape[1] + j];
  //   return result;
  // }

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

  Tensor& operator*=(const Tensor& other) {
    kernels::element_wise_mul_kernel(_data, other._data, _data, NumElements);
    return *this;
  }
  Tensor operator*(const Tensor& other) const { return Tensor(*this) *= other; }

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

// Implementations for at() and print() outside class body
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
template <typename T, size_t R, size_t C> Tensor<T, R> matmul(const Tensor<T, R, C>& mat, const Tensor<T, C>& vec) {
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

// --- Linear Layer (Fused Matrix Multiplication + Bias) ---
template <typename T, size_t BatchSeq, size_t InFeatures, size_t OutFeatures>
Tensor<T, BatchSeq, OutFeatures> linear(const Tensor<T, BatchSeq, InFeatures>& input,
                                        const Tensor<T, OutFeatures, InFeatures>& weight,
                                        const Tensor<T, OutFeatures>& bias) {
  Tensor<T, BatchSeq, OutFeatures> result;
  kernels::linear_kernel(input._data, weight._data, bias._data, result._data, BatchSeq, InFeatures, OutFeatures);
  return result;
}

// --- Softmax (applied to the last dimension) ---
template <typename T, size_t... Dims> Tensor<T, Dims...> softmax(const Tensor<T, Dims...>& input) {
  static_assert(sizeof...(Dims) > 0, "Softmax requires at least one dimension.");
  Tensor<T, Dims...> result;

  constexpr size_t LastDim = (std::get<sizeof...(Dims) - 1>(std::make_tuple(Dims...)));
  constexpr size_t NumVectors = Tensor<T, Dims...>::NumElements / LastDim;

  for (size_t i = 0; i < NumVectors; ++i) {
    const T* logits_slice = input._data + i * LastDim;
    T* probs_slice = result._data + i * LastDim;
    kernels::softmax_kernel(logits_slice, probs_slice, LastDim, LastDim, LastDim);
  }
  return result;
}

// Causal softmax with masking
template <typename T, size_t... Dims>
Tensor<T, Dims...> causal_softmax(const Tensor<T, Dims...>& input, size_t end_pos) {
  static_assert(sizeof...(Dims) > 0, "Causal softmax requires at least one dimension.");
  Tensor<T, Dims...> result;

  constexpr size_t LastDim = (std::get<sizeof...(Dims) - 1>(std::make_tuple(Dims...)));
  constexpr size_t NumVectors = Tensor<T, Dims...>::NumElements / LastDim;

  for (size_t i = 0; i < NumVectors; ++i) {
    const T* logits_slice = input._data + i * LastDim;
    T* probs_slice = result._data + i * LastDim;
    kernels::softmax_kernel(logits_slice, probs_slice, LastDim, end_pos, LastDim);
  }
  return result;
}

// --- Cross Entropy Loss ---
// Calculates the mean loss over a batch of predictions.
template <typename T, size_t BatchSize, size_t NumClasses>
Tensor<T> cross_entropy_loss(const Tensor<T, BatchSize, NumClasses>& probs, const std::array<int, BatchSize>& targets) {
  float total_loss = 0.0f;
  for (size_t i = 0; i < BatchSize; ++i) {
    const T* prob_slice = probs._data + i * NumClasses;
    total_loss += kernels::cross_entropy_loss_kernel(prob_slice, targets[i], NumClasses);
  }
  return Tensor<T>(total_loss / BatchSize);
}

// Overload for a single prediction (batch size of 1)
template <typename T, size_t NumClasses>
Tensor<T> cross_entropy_loss(const Tensor<T, NumClasses>& probs, int target_class) {
  float loss_val = kernels::cross_entropy_loss_kernel(probs._data, target_class, NumClasses);
  return Tensor<T>(loss_val);
}

// --- GELU Activation ---
template <typename T, size_t... Dims> Tensor<T, Dims...> gelu(const Tensor<T, Dims...>& input) {
  Tensor<T, Dims...> result;
  for (size_t i = 0; i < input.NumElements; ++i)
    result._data[i] = kernels::gelu_kernel(input._data[i]);
  return result;
}

// --- Utility Functions ---
template <typename T, size_t... Dims> Tensor<T, Dims...> norm(const Tensor<T, Dims...>& tensor) {
  return tensor.norm();
}

// template <typename T, size_t R, size_t C> Tensor<T, C, R> transpose(const Tensor<T, R, C>& tensor) {
//   return tensor.transpose();
// }

} // namespace cladtorch

#endif // CLAD_TENSOR_HPP_STATIC