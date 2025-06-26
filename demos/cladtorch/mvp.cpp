#include <clad/Differentiator/Differentiator.h>
#include <clad/Differentiator/STLBuiltins.h>
#include <clad/Differentiator/CladtorchBuiltins.h>
#include <cladtorch/simpletorch.hpp>
using namespace cladtorch;
using FTensor = Tensor<float>;

struct layer {
  // FTensor weight;
  FTensor bias;

  layer(int in_features, int out_features) {
    // weight = FTensor({out_features, in_features}, 0.1f);
    bias = FTensor({out_features}, 0.3f);
  }

  FTensor forward(const FTensor &input) const {
    auto res = input + bias;
    auto res2 = gelu(res);
    return res2;
  }
};

float add(const layer &l, FTensor a, FTensor b) {
  auto da = a * 0.3;
  auto db = b * 0.7;
  auto res = da + db;
  auto res2 = l.forward(res);
  return res2._data[0];
}

int main() {
  FTensor a({1}, 3), b({1}, 4), d_a({1}, 0.f), d_b({1}, 0.f);
  layer l(1, 1), d_l(1, 1);
  d_l.bias = FTensor({1}, 0.f);  // Initialize the gradient for bias
  auto grad = clad::gradient(add);
  grad.dump();
  grad.execute(l, a, b, &d_l, &d_a, &d_b);
  std::cout << "a: " << a._data[0] << ", b: " << b._data[0] << std::endl;
  std::cout << "d_a: " << d_a._data[0] << ", d_b: " << d_b._data[0] << std::endl;
  std::cout << "d_l.bias: " << d_l.bias._data[0] << std::endl;
}