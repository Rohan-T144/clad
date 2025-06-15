#ifndef CLAD_TENSOR_HPP_DYNAMIC
#define CLAD_TENSOR_HPP_DYNAMIC

#include "kernels.hpp" // Include kernel functions for operations
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

namespace cladtorch {
// -------------------- Dynamic-Shape Tensor Class --------------------
template <typename T> class Tensor {
public:
  std::vector<int> _shape;
  std::vector<int> _strides;
  int _num_elements = 0;
  T* _data = nullptr;

private:
  // Private helper to initialize tensor metadata and allocate memory
  void init_from_shape(const std::vector<int>& shape) {
    _shape = shape;
    _num_elements = 1;
    bool has_zero_dim = false;
    for (int dim : _shape) {
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
  explicit Tensor(const std::vector<int>& shape) {
    init_from_shape(shape);
    if (_data)
      std::fill(_data, _data + _num_elements, T{});
  }

  explicit Tensor(const std::vector<int>& shape, const T* data) {
    init_from_shape(shape);
    if (!_data) {
      // If _data is null, tensor is empty, nothing to copy
    } else if (data) {
      std::copy(data, data + _num_elements, _data);
    } else {
      std::fill(_data, _data + _num_elements, T{});
    }
  }

  // Shape and value constructor: creates a tensor filled with a scalar value.
  Tensor(const std::vector<int>& shape, T val) {
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
  const std::vector<int>& shape() const { return _shape; }
  int ndim() const { return _shape.size(); }
  int num_elements() const { return _num_elements; }
  int size(int dim) const {
    CLAD_ASSERT(dim < _shape.size(), "Dimension index out of range.");
    return _shape[dim];
  }
  T* data() { return _data; }
  const T* data() const { return _data; }

  // Broadcasting utilities
  static bool can_broadcast_to(const Tensor<T>& from, const Tensor<T>& to) {
    const auto& from_shape = from.shape();
    const auto& to_shape = to.shape();
    
    if (from_shape.size() > to_shape.size()) return false;
    
    int offset = to_shape.size() - from_shape.size();
    for (int i = 0; i < from_shape.size(); ++i) {
      int from_dim = from_shape[i];
      int to_dim = to_shape[i + offset];
      if (from_dim != 1 && from_dim != to_dim) {
        return false;
      }
    }
    return true;
  }

  // Compute the broadcast shape of two tensors
  static std::vector<int> broadcast_shape(const Tensor<T>& a, const Tensor<T>& b) {
    const auto& shape_a = a.shape();
    const auto& shape_b = b.shape();
    
    int max_ndim = std::max(shape_a.size(), shape_b.size());
    std::vector<int> result_shape(max_ndim);
    
    for (int i = 0; i < max_ndim; ++i) {
      int dim_a = (i < shape_a.size()) ? shape_a[shape_a.size() - 1 - i] : 1;
      int dim_b = (i < shape_b.size()) ? shape_b[shape_b.size() - 1 - i] : 1;
      
      if (dim_a == 1) {
        result_shape[max_ndim - 1 - i] = dim_b;
      } else if (dim_b == 1) {
        result_shape[max_ndim - 1 - i] = dim_a;
      } else if (dim_a == dim_b) {
        result_shape[max_ndim - 1 - i] = dim_a;
      } else {
        CLAD_ASSERT(false, "Shapes are not compatible for broadcasting.");
      }
    }
    
    return result_shape;
  }

  template <typename... IdxTypes> T& at(IdxTypes... idx_values) {
    CLAD_ASSERT(sizeof...(idx_values) == ndim(), "Number of indices must match tensor dimensions.");
    if (ndim() == 0) {
      CLAD_ASSERT(sizeof...(idx_values) == 0, "Do not provide indices for a scalar tensor.");
      return _data[0];
    }

    int indices[] = {static_cast<int>(idx_values)...};
    int flat_index = 0;
    for (int i = 0; i < ndim(); ++i) {
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
    for (int i = 0; i < _shape.size(); ++i)
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
      for (int i = 0; i < size(0); ++i)
        std::cout << at(i) << " ";
      std::cout << "\n";
    } else if (ndim() == 2) {
      for (int i = 0; i < size(0); ++i) {
        for (int j = 0; j < size(1); ++j)
          std::cout << at(i, j) << " ";
        std::cout << "\n";
      }
    } else {
      std::cout << "[";
      for (int i = 0; i < std::min((int)10, _num_elements); ++i)
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
    int slice_size = 1;
    for (int i = 1; i < ndim(); ++i)
      slice_size *= _shape[i];

    // Create result shape: [indices.num_elements(), remaining dimensions...]
    std::vector<int> result_shape;
    result_shape.push_back(indices.num_elements());
    for (int i = 1; i < ndim(); ++i)
      result_shape.push_back(_shape[i]);

    Tensor<T> result(result_shape);

    if (indices.num_elements() > 0)
      kernels::lookup_kernel(_data, indices.data(), result.data(), indices.num_elements(), _shape[0], slice_size);

    return result;
  }

  // Layer normalization: normalizes along the last dimension
  Tensor<T> norm() const {
    static_assert(std::is_same_v<T, float>, "norm() is only supported for float tensors.");
    CLAD_ASSERT(ndim() > 0, "Cannot normalize a scalar tensor.");
    CLAD_ASSERT(_data != nullptr, "Cannot normalize null data tensor.");

    Tensor<T> result(_shape);

    if (_num_elements == 0)
      return result;

    // Calculate number of vectors and vector size
    int vec_size = _shape.back(); // Last dimension
    int num_vectors = _num_elements / vec_size;

    kernels::norm_kernel(_data, result.data(), num_vectors, vec_size);

    return result;
  }

  // Create a view (slice) of the tensor along specified axis at given offset
  Tensor<T> view(const std::vector<int>& new_shape, int split_no = 0, int split_axis = 0) const {
    CLAD_ASSERT(split_axis < ndim(), "Split axis out of bounds.");
    CLAD_ASSERT(new_shape.size() == ndim(), "View shape must have same number of dimensions.");
    CLAD_ASSERT(_data != nullptr, "Cannot create view of null data tensor.");

    // Calculate offset for this split
    int split_size = new_shape[split_axis];
    int offset = split_no * split_size;
    CLAD_ASSERT(offset + split_size <= _shape[split_axis], "View extends beyond tensor bounds.");

    Tensor<T> result(new_shape);

    if (result._num_elements == 0)
      return result;

    kernels::view_kernel(_data, result.data(), _shape, _strides, new_shape, result._strides, split_axis, offset);

    return result;
  }

  // Split tensor along specified axis into chunks of given size
  std::vector<Tensor<T>> split(int size, int axis) const {
    CLAD_ASSERT(axis < ndim(), "Split axis out of bounds.");
    CLAD_ASSERT(_shape[axis] % size == 0, "Dimension size must be divisible by split size.");
    CLAD_ASSERT(_data != nullptr, "Cannot split null data tensor.");

    std::vector<Tensor<T>> tensors;

    if (_shape[axis] == size) {
      // If split size equals dimension size, return copy of this tensor
      tensors.push_back(*this);
      return tensors;
    }

    // Create new shape for each split
    std::vector<int> split_shape = _shape;
    split_shape[axis] = size;

    int num_splits = _shape[axis] / size;
    for (int i = 0; i < num_splits; ++i)
      tensors.push_back(view(split_shape, i, axis));

    return tensors;
  }

  // Transpose two dimensions of the tensor
  Tensor<T> transpose(int dim0, int dim1) const {
    CLAD_ASSERT(dim0 < ndim() && dim1 < ndim(), "Transpose dimensions out of bounds.");
    CLAD_ASSERT(_data != nullptr, "Cannot transpose null data tensor.");

    if (dim0 == dim1) {
      // No-op transpose, return copy
      return *this;
    }

    // Create transposed shape
    std::vector<int> new_shape = _shape;
    std::swap(new_shape[dim0], new_shape[dim1]);

    Tensor<T> result(new_shape);

    if (_num_elements == 0)
      return result;

    kernels::transpose_kernel(_data, result.data(), _shape, _strides, result._strides, dim0, dim1);

    return result;
  }

  // Broadcast this tensor to the given shape
  Tensor<T> broadcast_to(const std::vector<int>& target_shape) const {
    CLAD_ASSERT(_data != nullptr || _num_elements == 0, "Cannot broadcast null data tensor.");
    
    // Check if broadcasting is possible
    if (_shape.size() > target_shape.size()) {
      CLAD_ASSERT(false, "Cannot broadcast to smaller number of dimensions.");
    }
    
    int offset = target_shape.size() - _shape.size();
    for (int i = 0; i < _shape.size(); ++i) {
      int src_dim = _shape[i];
      int target_dim = target_shape[i + offset];
      if (src_dim != 1 && src_dim != target_dim) {
        CLAD_ASSERT(false, "Incompatible shapes for broadcasting.");
      }
    }
    
    // If shapes are already the same, return a copy
    if (_shape == target_shape) {
      return *this;
    }
    
    Tensor<T> result(target_shape);
    
    if (result._num_elements == 0 || _num_elements == 0) {
      return result;
    }
    
    kernels::broadcast_kernel(_data, result.data(), _shape, _strides, target_shape, result._strides);
    
    return result;
  }

  // Convenience method for 2D matrix transpose (swap dimensions 0 and 1)
  // Tensor<T> T() const {
  //   CLAD_ASSERT(ndim() >= 2, "Matrix transpose requires at least 2 dimensions.");
  //   return transpose(ndim() - 2, ndim() - 1);  // Transpose last two dimensions
  // }

  // --- Operator Overloads ---
  Tensor& operator+=(const Tensor& other) {
    if (_shape == other._shape) {
      // Same shape, use optimized kernel
      kernels::element_wise_add_kernel(_data, other._data, _data, _num_elements);
    } else {
      // Different shapes, need broadcasting
      std::vector<int> result_shape = broadcast_shape(*this, other);
      CLAD_ASSERT(result_shape == _shape, "In-place addition requires this tensor to have the broadcast result shape.");
      
      Tensor<T> other_broadcast = other.broadcast_to(_shape);
      kernels::element_wise_add_kernel(_data, other_broadcast._data, _data, _num_elements);
    }
    return *this;
  }
  
  Tensor operator+(const Tensor& other) const {
    if (_shape == other._shape) {
      // Same shape, use optimized path
      return Tensor(*this) += other;
    } else {
      // Different shapes, need broadcasting
      std::vector<int> result_shape = broadcast_shape(*this, other);
      Tensor<T> result(result_shape);
      
      Tensor<T> a_broadcast = this->broadcast_to(result_shape);
      Tensor<T> b_broadcast = other.broadcast_to(result_shape);
      
      kernels::element_wise_add_kernel(a_broadcast._data, b_broadcast._data, result._data, result._num_elements);
      return result;
    }
  }

  Tensor& operator-=(const Tensor& other) {
    if (_shape == other._shape) {
      // Same shape, use optimized kernel
      kernels::element_wise_sub_kernel(_data, other._data, _data, _num_elements);
    } else {
      // Different shapes, need broadcasting
      std::vector<int> result_shape = broadcast_shape(*this, other);
      CLAD_ASSERT(result_shape == _shape, "In-place subtraction requires this tensor to have the broadcast result shape.");
      
      Tensor<T> other_broadcast = other.broadcast_to(_shape);
      kernels::element_wise_sub_kernel(_data, other_broadcast._data, _data, _num_elements);
    }
    return *this;
  }
  
  Tensor operator-(const Tensor& other) const {
    if (_shape == other._shape) {
      // Same shape, use optimized path
      return Tensor(*this) -= other;
    } else {
      // Different shapes, need broadcasting
      std::vector<int> result_shape = broadcast_shape(*this, other);
      Tensor<T> result(result_shape);
      
      Tensor<T> a_broadcast = this->broadcast_to(result_shape);
      Tensor<T> b_broadcast = other.broadcast_to(result_shape);
      
      kernels::element_wise_sub_kernel(a_broadcast._data, b_broadcast._data, result._data, result._num_elements);
      return result;
    }
  }

  Tensor& operator*=(const Tensor& other) {
    if (_shape == other._shape) {
      // Same shape, use optimized kernel
      kernels::element_wise_mul_kernel(_data, other._data, _data, _num_elements);
    } else {
      // Different shapes, need broadcasting
      std::vector<int> result_shape = broadcast_shape(*this, other);
      CLAD_ASSERT(result_shape == _shape, "In-place multiplication requires this tensor to have the broadcast result shape.");
      
      Tensor<T> other_broadcast = other.broadcast_to(_shape);
      kernels::element_wise_mul_kernel(_data, other_broadcast._data, _data, _num_elements);
    }
    return *this;
  }
  
  Tensor operator*(const Tensor& other) const {
    if (_shape == other._shape) {
      // Same shape, use optimized path
      return Tensor(*this) *= other;
    } else {
      // Different shapes, need broadcasting
      std::vector<int> result_shape = broadcast_shape(*this, other);
      Tensor<T> result(result_shape);
      
      Tensor<T> a_broadcast = this->broadcast_to(result_shape);
      Tensor<T> b_broadcast = other.broadcast_to(result_shape);
      
      kernels::element_wise_mul_kernel(a_broadcast._data, b_broadcast._data, result._data, result._num_elements);
      return result;
    }
  }

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
  const Tensor<T>* a_ptr = &a;
  const Tensor<T>* b_ptr = &b;
  
  // Check if we need broadcasting for batched operations
  if (a.ndim() == 3 && b.ndim() == 3) {
    // Both are 3D, check if batch dimensions are compatible
    if (a.size(0) != b.size(0)) {
      // Try broadcasting
      if (a.size(0) == 1) {
        // Broadcast a's batch dimension
        std::vector<int> new_shape = a.shape();
        new_shape[0] = b.size(0);
        static Tensor<T> a_broadcast = a.broadcast_to(new_shape);
        a_ptr = &a_broadcast;
      } else if (b.size(0) == 1) {
        // Broadcast b's batch dimension
        std::vector<int> new_shape = b.shape();
        new_shape[0] = a.size(0);
        static Tensor<T> b_broadcast = b.broadcast_to(new_shape);
        b_ptr = &b_broadcast;
      } else {
        CLAD_ASSERT(false, "Incompatible batch dimensions for matmul.");
      }
    }
    
    CLAD_ASSERT(a_ptr->size(2) == b_ptr->size(1), "Inner dimensions must match for batched matmul (a.shape[2] == b.shape[1]).");
    int B = a_ptr->size(0), R = a_ptr->size(1), C1 = a_ptr->size(2), C2 = b_ptr->size(2);
    Tensor<T> result({B, R, C2});
    kernels::batched_mat_mul_kernel(a_ptr->data(), b_ptr->data(), result.data(), B, R, C1, C2);
    return result;
  }
  
  // Case 2: Matrix-Matrix Multiplication (2D x 2D)
  if (a.ndim() == 2 && b.ndim() == 2) {
    CLAD_ASSERT(a.size(1) == b.size(0), "Inner dimensions must match for matmul (a.shape[1] == b.shape[0]).");
    int R = a.size(0), C1 = a.size(1), C2 = b.size(1);
    Tensor<T> result({R, C2});
    kernels::mat_mul_kernel(a.data(), b.data(), result.data(), R, C1, C2);
    return result;
  }
  
  // Case 3: Matrix-Vector Multiplication (2D x 1D)
  if (a.ndim() == 2 && b.ndim() == 1) {
    CLAD_ASSERT(a.size(1) == b.size(0), "Inner dimensions must match for mat-vec mul (a.shape[1] == b.shape[0]).");
    int R = a.size(0), C = a.size(1);
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

  int last_dim = input.shape().back();
  int num_vectors = input.num_elements() / last_dim;

  for (int i = 0; i < num_vectors; ++i) {
    const T* logits_slice = input.data() + i * last_dim;
    T* probs_slice = result.data() + i * last_dim;
    kernels::softmax_kernel(logits_slice, probs_slice, last_dim);
  }
  return result;
}

// Batched cross-entropy loss
template <typename T> Tensor<T> cross_entropy_loss(const Tensor<T>& probs, const std::vector<int>& targets) {
  CLAD_ASSERT(probs.ndim() == 2, "Probs tensor must be 2D for batched cross entropy loss.");
  int batch_size = probs.size(0);
  int num_classes = probs.size(1);
  CLAD_ASSERT(batch_size == targets.size(), "Batch size of probs and targets must match.");

  float total_loss = 0.0f;
  for (int i = 0; i < batch_size; ++i) {
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
  for (int i = 0; i < in.num_elements(); ++i)
    r.data()[i] = kernels::gelu_kernel(in.data()[i]);
  return r;
}

template <typename T> Tensor<T> lookup(const Tensor<T>& src, const Tensor<int>& indices) { return src.lookup(indices); }

template <typename T> Tensor<T> norm(const Tensor<T>& input) { return input.norm(); }

template <typename T> std::vector<Tensor<T>> split(const Tensor<T>& input, int size, int axis) {
  return input.split(size, axis);
}

template <typename T> Tensor<T> transpose(const Tensor<T>& input, int dim0, int dim1) {
  return input.transpose(dim0, dim1);
}

template <typename T> Tensor<T> broadcast_to(const Tensor<T>& input, const std::vector<int>& target_shape) {
  return input.broadcast_to(target_shape);
}

} // namespace cladtorch

#endif // CLAD_TENSOR_HPP_DYNAMIC
