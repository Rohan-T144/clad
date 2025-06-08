#ifndef CLAD_TENSOR_HPP_STATIC
#define CLAD_TENSOR_HPP_STATIC

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#define CLAD_ASSERT(condition, message) assert((condition) && message)

namespace cladtorch {

// -------------------- Kernel Functions (Operating on raw pointers) --------------------
// These remain largely unchanged as they are low-level, operating on float pointers.
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

  if (sum_exp == 0.0f) // Avoid division by zero if all exps are tiny
    sum_exp = 1e-9f;
  for (int i = 0; i < size; ++i)
    probs[i] /= sum_exp;
}

inline float cross_entropy_loss_kernel(const float* probs, int target_class, int size) {
  if (target_class < 0 || target_class >= size)
    return -std::log(1e-9f); // Return large loss for invalid target
  // Ensure probability is not zero or negative to avoid log(0) or log(<0)
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

inline void vec_add_kernel(const float* vec1, const float* vec2, float* result, int size) {
  for (int i = 0; i < size; ++i)
    result[i] = vec1[i] + vec2[i];
}

inline void element_wise_add_kernel(const float* a_data, const float* b_data, float* result_data, size_t num_elements) {
  for (size_t i = 0; i < num_elements; ++i)
    result_data[i] = a_data[i] + b_data[i];
}

inline void element_wise_mul_kernel(const float* a_data, const float* b_data, float* result_data, size_t num_elements) {
  for (size_t i = 0; i < num_elements; ++i)
    result_data[i] = a_data[i] * b_data[i];
}

} // namespace kernels

// -------------------- Compile-Time Tensor Class --------------------
template <typename T, size_t... Dims> class Tensor {
public:
  static constexpr size_t NDim = sizeof...(Dims);

  // Ensure all dimensions are positive at compile time (for non-scalar tensors)
  static_assert(NDim == 0 || ((Dims > 0) && ...), "All tensor dimensions must be positive.");

private:
  // Compile-time calculation for total number of elements
  static constexpr size_t calculate_num_elements() {
    if constexpr (NDim == 0) // Scalar tensor
      return 1;
    else
      return (Dims * ... * 1); // C++17 fold expression
  }

public:
  static constexpr size_t NumElements = calculate_num_elements();
  static constexpr std::array<size_t, NDim> Shape = {Dims...};

private:
  // Compile-time calculation for strides (row-major)
  static constexpr std::array<size_t, NDim> calculate_strides() {
    if constexpr (NDim == 0) {
      return std::array<size_t, 0>{};
    } else {
      std::array<size_t, NDim> s{};
      s[NDim - 1] = 1;                     // Stride of the last dimension is 1
      for (long i = NDim - 2; i >= 0; --i) // Iterate from NDim-2 down to 0
        s[i] = s[i + 1] * Shape[i + 1];
      return s;
    }
  }

public:
  static constexpr std::array<size_t, NDim> Strides = calculate_strides();

public:
  T* _data;

public:
  // Default constructor: initializes elements to zero
  Tensor() : _data(nullptr) {
    if constexpr (NumElements > 0) {
      _data = new T[NumElements];
      std::fill(_data, _data + NumElements, T{});
    }
    // If NumElements is 0 (e.g. from a zero dimension, though static_assert prevents this for Dims), _data remains
    // nullptr
  }

  // Constructor to fill with a single value
  explicit Tensor(T fill_value) : _data(nullptr) {
    if constexpr (NumElements > 0) {
      _data = new T[NumElements];
      std::fill(_data, _data + NumElements, fill_value);
    }
  }

  // Constructor from a C-style array (expects NumElements items)
  explicit Tensor(const T* initial_data) : _data(nullptr) {
    CLAD_ASSERT(initial_data != nullptr || NumElements == 0,
                "Initial data pointer cannot be null for non-empty tensor.");
    if constexpr (NumElements > 0) {
      _data = new T[NumElements];
      for (size_t i = 0; i < NumElements; ++i)
        _data[i] = initial_data[i];
    }
  }

  // Constructor for 1D tensor using initializer_list (only if NDim == 1)
  template <typename U = T, typename = std::enable_if_t<std::is_same_v<U, T> && NDim == 1>>
  Tensor(std::initializer_list<T> il) : _data(nullptr) {
    CLAD_ASSERT(il.size() == Shape[0], "Initializer list size must match tensor's dimension for 1D tensor.");
    if constexpr (NumElements > 0) { // NumElements == Shape[0] for 1D
      _data = new T[NumElements];
      std::copy(il.begin(), il.end(), _data);
    }
  }

  // Destructor
  ~Tensor() {
    delete[] _data;
    _data = nullptr;
  }

  // Copy constructor
  Tensor(const Tensor& other) : _data(nullptr) {
    if constexpr (NumElements > 0) {
      _data = new T[NumElements];
      CLAD_ASSERT(other._data != nullptr, "Source data for copy is null."); // other._data could be null if
      // other.NumElements is 0
      if (other._data) // Only copy if source data exists
        for (size_t i = 0; i < NumElements; ++i)
          _data[i] = other._data[i];
      else // If other._data is null but NumElements > 0, initialize to default.
        std::fill(_data, _data + NumElements, T{});
    }
  }

  // Copy assignment operator
  Tensor& operator=(const Tensor& other) {
    if (this == &other)
      return *this;
    if constexpr (NumElements > 0) {
      if (!_data) // Allocate if current _data is null (e.g., moved-from state)
        _data = new T[NumElements];
      CLAD_ASSERT(other._data != nullptr, "Source data for copy assignment is null.");
      if (other._data)
        for (size_t i = 0; i < NumElements; ++i)
          _data[i] = other._data[i];
      else
        std::fill(_data, _data + NumElements, T{});
    }
    // If NumElements is 0, _data is nullptr, nothing to do.
    return *this;
  }

  // Move constructor
  Tensor(Tensor&& other) noexcept : _data(other._data) { other._data = nullptr; }

  // Move assignment operator
  Tensor& operator=(Tensor&& other) noexcept {
    if (this == &other)
      return *this;
    delete[] _data;
    _data = other._data;
    other._data = nullptr;
    return *this;
  }

  // --- Accessors ---
  static constexpr size_t ndim() { return NDim; }
  static constexpr size_t numel() { return NumElements; }

  static constexpr size_t shape(size_t dim_idx) {
    CLAD_ASSERT(dim_idx < NDim, "Dimension index out of bounds.");
    return Shape[dim_idx];
  }
  static constexpr const std::array<size_t, NDim>& get_shape_array() { return Shape; }
  static constexpr const std::array<size_t, NDim>& get_strides_array() { return Strides; }

  // Element access using variadic indices (e.g., tensor.at(i, j, k))
  template <typename... IdxTypes> T& at(IdxTypes... idx_values) {
    static_assert((NDim == 0 && sizeof...(IdxTypes) == 0) || (NDim > 0 && sizeof...(IdxTypes) == NDim),
                  "Number of indices must match tensor dimensions (or zero indices for scalar).");
    static_assert(((std::is_integral_v<IdxTypes>) && ...), "Indices must be integral types.");

    if constexpr (NDim == 0) { // Scalar case
      CLAD_ASSERT(_data != nullptr, "Accessing null data in scalar tensor.");
      return _data[0];
    } else {
      size_t indices[] = {static_cast<size_t>(idx_values)...};
      size_t flat_index = 0;
      for (size_t i = 0; i < NDim; ++i) {
        CLAD_ASSERT(indices[i] < Shape[i], "Index out of bounds.");
        flat_index += indices[i] * Strides[i];
      }
      CLAD_ASSERT(flat_index < NumElements, "Calculated flat index out of bounds.");
      CLAD_ASSERT(_data != nullptr, "Accessing null data.");
      return _data[flat_index];
    }
  }

  template <typename... IdxTypes> const T& at(IdxTypes... idx_values) const {
    static_assert((NDim == 0 && sizeof...(IdxTypes) == 0) || (NDim > 0 && sizeof...(IdxTypes) == NDim),
                  "Number of indices must match tensor dimensions (or zero indices for scalar).");
    static_assert(((std::is_integral_v<IdxTypes>) && ...), "Indices must be integral types.");

    if constexpr (NDim == 0) {
      CLAD_ASSERT(_data != nullptr, "Accessing null data in scalar tensor const.");
      return _data[0];
    } else {
      size_t indices[] = {static_cast<size_t>(idx_values)...};
      size_t flat_index = 0;
      for (size_t i = 0; i < NDim; ++i) {
        CLAD_ASSERT(indices[i] < Shape[i], "Index out of bounds.");
        flat_index += indices[i] * Strides[i];
      }
      CLAD_ASSERT(flat_index < NumElements, "Calculated flat index out of bounds.");
      CLAD_ASSERT(_data != nullptr, "Accessing null data const.");
      return _data[flat_index];
    }
  }

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

  // Utility to print the tensor (basic)
  void print(const std::string& title = "") const {
    if (!title.empty())
      std::cout << title;
    std::cout << " (Shape: [";
    if constexpr (NDim > 0)
      for (size_t i = 0; i < NDim; ++i)
        std::cout << Shape[i] << (i == NDim - 1 ? "" : ", ");
    std::cout << "], NumElements: " << NumElements << ")" << std::endl;

    if constexpr (NumElements == 0) {
      std::cout << "(Tensor with zero elements)" << std::endl;
      return;
    }
    CLAD_ASSERT(_data != nullptr, "Printing null data tensor.");

    if constexpr (NDim == 0) { // Scalar
      std::cout << _data[0] << std::endl;
    } else if constexpr (NDim == 1) {
      for (size_t i = 0; i < Shape[0]; ++i)
        std::cout << this->at(i) << " "; // Use this->at for clarity
      std::cout << std::endl;
    } else if constexpr (NDim == 2) {
      for (size_t i = 0; i < Shape[0]; ++i) {
        for (size_t j = 0; j < Shape[1]; ++j)
          std::cout << this->at(i, j) << " ";
        std::cout << std::endl;
      }
    } else { // Print first few elements for NDim > 2
      std::cout << "[";
      for (size_t i = 0; i < std::min((size_t)10, NumElements); ++i)
        std::cout << _data[i] << (i == std::min((size_t)10, NumElements) - 1 || i == NumElements - 1 ? "" : ", ");
      if (NumElements > 10)
        std::cout << " ...";
      std::cout << "]" << std::endl;
    }
  }
};

// -------------------- Tensor Operations (Wrappers around Kernels) --------------------
// Hardcoded to T=float for current problem scope, but templated on T.

template <typename T, size_t... Dims> Tensor<T, Dims...> add(const Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b) {
  Tensor<T, Dims...> result;
  kernels::element_wise_add_kernel(a._data, b._data, result._data, Tensor<T, Dims...>::NumElements);
  return result;
}

template <typename T, size_t... Dims>
Tensor<T, Dims...> mul(const Tensor<T, Dims...>& a, const Tensor<T, Dims...>& b) { // Element-wise
  Tensor<T, Dims...> result;
  kernels::element_wise_mul_kernel(a._data, b._data, result._data, Tensor<T, Dims...>::NumElements);
  return result;
}

// Matrix-vector multiply: mat (RxC) * vec (C) -> result (R)
template <typename T, size_t R, size_t C>
Tensor<T, R> mat_vec_mul(const Tensor<T, R, C>& mat, const Tensor<T, C>& vec) {
  Tensor<T, R> result;
  kernels::mat_vec_mul_kernel(mat._data, vec._data, result._data, R, C);
  return result;
}

// Vector-vector add (element-wise, specialized for 1D Tensors)
// This is essentially the same as the general 'add' when Dims... has only one dimension.
// Kept for API consistency if desired, but 'add' works for 1D too.
template <typename T, size_t Size> Tensor<T, Size> vec_add(const Tensor<T, Size>& a, const Tensor<T, Size>& b) {
  Tensor<T, Size> result;
  kernels::vec_add_kernel(a._data, b._data, result._data, Size);
  return result;
}

template <typename T, size_t... Dims> Tensor<T, Dims...> apply_gelu(const Tensor<T, Dims...>& input) {
  Tensor<T, Dims...> result;
  T* out_data = result._data;
  const T* in_data = input._data;
  constexpr size_t num_elements = Tensor<T, Dims...>::NumElements;

  if constexpr (num_elements > 0) { // Ensure data pointers are valid if operating
    CLAD_ASSERT(out_data != nullptr && in_data != nullptr, "Data pointers are null in apply_gelu.");
    for (size_t i = 0; i < num_elements; ++i)
      out_data[i] = kernels::gelu_kernel(in_data[i]);
  }
  return result;
}

template <typename T, size_t Size> Tensor<T, Size> softmax(const Tensor<T, Size>& logits) {
  Tensor<T, Size> result;
  kernels::softmax_kernel(logits._data, result._data, Size);
  return result;
}

template <typename T, size_t ProbSize>
Tensor<T> cross_entropy_loss(const Tensor<T, ProbSize>& probs, int target_class) {
  float loss_val = kernels::cross_entropy_loss_kernel(probs._data, target_class, ProbSize);
  Tensor<T> loss_tensor{loss_val};
  return loss_tensor;
}

} // namespace cladtorch

#endif // CLAD_TENSOR_HPP_STATIC