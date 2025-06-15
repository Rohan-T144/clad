#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#define CLAD_ASSERT(condition, message) assert((condition) && message)

// -------------------- Kernel Functions (Operating on raw pointers) --------------------
// These functions are low-level, high-performance routines that operate on raw C-style
// arrays. They remain unchanged as their performance and interface are independent of
// how the Tensor class manages its data.
namespace cladtorch::kernels {
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
inline void lookup_kernel(const T* src_data, const int* indices, T* dst_data, size_t num_indices, size_t src_first_dim,
                          size_t slice_size) {
  for (size_t i = 0; i < num_indices; ++i) {
    int idx = indices[i];
    CLAD_ASSERT(idx >= 0 && idx < (int)src_first_dim, "Index out of bounds in lookup.");

    const T* src_slice = src_data + idx * slice_size;
    T* dst_slice = dst_data + i * slice_size;

    for (size_t j = 0; j < slice_size; ++j)
      dst_slice[j] = src_slice[j];
  }
}

template <typename T>
inline void view_kernel(const T* src_data, T* dst_data, const std::vector<int>& src_shape,
                        const std::vector<int>& src_strides, const std::vector<int>& dst_shape,
                        const std::vector<int>& dst_strides, size_t split_axis, size_t offset) {
  size_t dst_elements = 1;
  for (size_t dim : dst_shape)
    dst_elements *= dim;

  for (size_t dst_idx = 0; dst_idx < dst_elements; ++dst_idx) {
    // Convert flat index to multi-dimensional indices for dst
    std::vector<int> dst_coords(dst_shape.size());
    size_t temp_idx = dst_idx;
    for (long i = dst_shape.size() - 1; i >= 0; --i) {
      dst_coords[i] = temp_idx % dst_shape[i];
      temp_idx /= dst_shape[i];
    }

    // Map dst coordinates to src coordinates
    std::vector<int> src_coords = dst_coords;
    src_coords[split_axis] += offset;

    // Convert src coordinates to flat index
    size_t src_idx = 0;
    for (size_t i = 0; i < src_coords.size(); ++i)
      src_idx += src_coords[i] * src_strides[i];

    dst_data[dst_idx] = src_data[src_idx];
  }
}

template <typename T>
inline void transpose_kernel(const T* src_data, T* dst_data, const std::vector<int>& src_shape,
                             const std::vector<int>& src_strides, const std::vector<int>& dst_strides,
                             size_t dim0, size_t dim1) {
  size_t total_elements = 1;
  for (size_t dim : src_shape)
    total_elements *= dim;

  for (size_t src_idx = 0; src_idx < total_elements; ++src_idx) {
    // Convert flat index to multi-dimensional coordinates
    std::vector<int> coords(src_shape.size());
    size_t temp_idx = src_idx;
    for (long i = src_shape.size() - 1; i >= 0; --i) {
      coords[i] = temp_idx % src_shape[i];
      temp_idx /= src_shape[i];
    }

    // Swap the dimensions for transpose
    std::swap(coords[dim0], coords[dim1]);

    // Calculate destination flat index
    size_t dst_idx = 0;
    for (size_t i = 0; i < coords.size(); ++i)
      dst_idx += coords[i] * dst_strides[i];

    dst_data[dst_idx] = src_data[src_idx];
  }
}

inline float vec_mean_kernel(size_t vec_size, const float* src) {
  float sum = 0.0f;
  for (size_t i = 0; i < vec_size; i++)
    sum += src[i];
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

    for (size_t i = 0; i < vec_size; i++)
      out[i] = (vec[i] - mean) * rstd;
  }
}

template <typename T>
inline void broadcast_kernel(const T* src_data, T* dst_data, 
                           const std::vector<int>& src_shape, const std::vector<int>& src_strides,
                           const std::vector<int>& dst_shape, const std::vector<int>& dst_strides) {
  size_t total_elements = 1;
  for (int dim : dst_shape)
    total_elements *= dim;
    
  for (size_t dst_idx = 0; dst_idx < total_elements; ++dst_idx) {
    // Convert flat index to multi-dimensional coordinates for dst
    std::vector<int> dst_coords(dst_shape.size());
    size_t temp_idx = dst_idx;
    for (long i = dst_shape.size() - 1; i >= 0; --i) {
      dst_coords[i] = temp_idx % dst_shape[i];
      temp_idx /= dst_shape[i];
    }
    
    // Map dst coordinates to src coordinates with broadcasting
    std::vector<int> src_coords(src_shape.size(), 0);
    int src_dim = src_shape.size() - 1;
    for (int dst_dim = dst_shape.size() - 1; dst_dim >= 0 && src_dim >= 0; --dst_dim, --src_dim) {
      if (src_shape[src_dim] == 1) {
        src_coords[src_dim] = 0;  // Broadcast dimension
      } else {
        src_coords[src_dim] = dst_coords[dst_dim];
      }
    }
    
    // Convert src coordinates to flat index
    size_t src_idx = 0;
    for (size_t i = 0; i < src_coords.size(); ++i)
      src_idx += src_coords[i] * src_strides[i];
      
    dst_data[dst_idx] = src_data[src_idx];
  }
}

template <typename T>
inline void broadcast_add_kernel(const T* a_data, const T* b_data, T* result_data,
                               const std::vector<int>& a_shape, const std::vector<int>& a_strides,
                               const std::vector<int>& b_shape, const std::vector<int>& b_strides,
                               const std::vector<int>& result_shape, const std::vector<int>& result_strides) {
  size_t total_elements = 1;
  for (int dim : result_shape)
    total_elements *= dim;
    
  for (size_t result_idx = 0; result_idx < total_elements; ++result_idx) {
    // Convert flat index to multi-dimensional coordinates
    std::vector<int> coords(result_shape.size());
    size_t temp_idx = result_idx;
    for (long i = result_shape.size() - 1; i >= 0; --i) {
      coords[i] = temp_idx % result_shape[i];
      temp_idx /= result_shape[i];
    }
    
    // Get indices for a and b with broadcasting
    size_t a_idx = 0, b_idx = 0;
    
    // Map to a coordinates
    int a_dim = a_shape.size() - 1;
    for (int coord_dim = coords.size() - 1; coord_dim >= 0 && a_dim >= 0; --coord_dim, --a_dim) {
      int a_coord = (a_shape[a_dim] == 1) ? 0 : coords[coord_dim];
      a_idx += a_coord * a_strides[a_dim];
    }
    
    // Map to b coordinates  
    int b_dim = b_shape.size() - 1;
    for (int coord_dim = coords.size() - 1; coord_dim >= 0 && b_dim >= 0; --coord_dim, --b_dim) {
      int b_coord = (b_shape[b_dim] == 1) ? 0 : coords[coord_dim];
      b_idx += b_coord * b_strides[b_dim];
    }
    
    result_data[result_idx] = a_data[a_idx] + b_data[b_idx];
  }
}

} // namespace cladtorch::kernels
