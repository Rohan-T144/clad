// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oFunctors.out 2>&1 | %filecheck %s
// RUN: ./Functors.out | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -oFunctors.out
// RUN: ./Functors.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

struct Experiment {
  mutable double x, y;
  Experiment(double p_x = 0, double p_y = 0) : x(p_x), y(p_y) {}
  double operator()(double i, double j) {
    return x*i*i*j + y*i*j*j;
  }
  void setX(double val) {
    x = val;
  }

  // CHECK: void operator_call_hessian(double i, double j, double *hessianMatrix) {
  // CHECK-NEXT:     Experiment _d_this;
  // CHECK-NEXT:     this->operator_call_darg0_grad(i, j, &_d_this, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
  // CHECK-NEXT:     Experiment _d_this0;
  // CHECK-NEXT:     this->operator_call_darg1_grad(i, j, &_d_this0, hessianMatrix + {{2U|2UL|2ULL}}, hessianMatrix + {{3U|3UL|3ULL}});
  // CHECK-NEXT: }
};

struct ExperimentConst {
  mutable double x, y;
  ExperimentConst(double p_x = 0, double p_y = 0) : x(p_x), y(p_y) {}
  double operator()(double i, double j) const {
    return x*i*i*j + y*i*j*j;
  }
  void setX(double val) const {
    x = val;
  }

  // CHECK: void operator_call_hessian(double i, double j, double *hessianMatrix) const {
  // CHECK-NEXT:     ExperimentConst _d_this;
  // CHECK-NEXT:     this->operator_call_darg0_grad(i, j, &_d_this, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
  // CHECK-NEXT:     ExperimentConst _d_this0;
  // CHECK-NEXT:     this->operator_call_darg1_grad(i, j, &_d_this0, hessianMatrix + {{2U|2UL|2ULL}}, hessianMatrix + {{3U|3UL|3ULL}});
  // CHECK-NEXT: }
};

struct ExperimentVolatile {
  mutable double x, y;
  ExperimentVolatile(double p_x = 0, double p_y = 0) : x(p_x), y(p_y) {}
  double operator()(double i, double j) volatile {
    return x*i*i*j + y*i*j*j;
  }
  void setX(double val) volatile {
    x = val;
  }

  // CHECK: void operator_call_hessian(double i, double j, double *hessianMatrix) volatile {
  // CHECK-NEXT:     volatile ExperimentVolatile _d_this;
  // CHECK-NEXT:     this->operator_call_darg0_grad(i, j, &_d_this, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
  // CHECK-NEXT:     volatile ExperimentVolatile _d_this0;
  // CHECK-NEXT:     this->operator_call_darg1_grad(i, j, &_d_this0, hessianMatrix + {{2U|2UL|2ULL}}, hessianMatrix + {{3U|3UL|3ULL}});
  // CHECK-NEXT: }
};

struct ExperimentConstVolatile {
  mutable double x, y;
  ExperimentConstVolatile(double p_x = 0, double p_y = 0) : x(p_x), y(p_y) {}
  double operator()(double i, double j) const volatile {
    return x*i*i*j + y*i*j*j;
  }
  void setX(double val) const volatile {
    x = val;
  }

  // CHECK: void operator_call_hessian(double i, double j, double *hessianMatrix) const volatile {
  // CHECK-NEXT:     volatile ExperimentConstVolatile _d_this;
  // CHECK-NEXT:     this->operator_call_darg0_grad(i, j, &_d_this, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
  // CHECK-NEXT:     volatile ExperimentConstVolatile _d_this0;
  // CHECK-NEXT:     this->operator_call_darg1_grad(i, j, &_d_this0, hessianMatrix + {{2U|2UL|2ULL}}, hessianMatrix + {{3U|3UL|3ULL}});
  // CHECK-NEXT: }
};

namespace outer {
  namespace inner {
    struct ExperimentNNS {
      mutable double x, y;
      ExperimentNNS(double p_x = 0, double p_y = 0) : x(p_x), y(p_y) {}
      double operator()(double i, double j) {
        return x*i*i*j + y*i*j*j;
      }
      void setX(double val) {
        x = val;
      }

      // CHECK: void operator_call_hessian(double i, double j, double *hessianMatrix) {
      // CHECK-NEXT:     outer::inner::ExperimentNNS _d_this;
      // CHECK-NEXT:     this->operator_call_darg0_grad(i, j, &_d_this, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
      // CHECK-NEXT:     outer::inner::ExperimentNNS _d_this0;
      // CHECK-NEXT:     this->operator_call_darg1_grad(i, j, &_d_this0, hessianMatrix + {{2U|2UL|2ULL}}, hessianMatrix + {{3U|3UL|3ULL}});
      // CHECK-NEXT: }
    };

    auto lambdaNNS = [](double i, double j) {
      return i*i*j*j;
    };

    // CHECK: inline void operator_call_hessian(double i, double j, double *hessianMatrix) const {
    // CHECK-NEXT:     this->operator_call_darg0_grad(i, j, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
    // CHECK-NEXT:     this->operator_call_darg1_grad(i, j, hessianMatrix + {{2U|2UL|2ULL}}, hessianMatrix + {{3U|3UL|3ULL}});
    // CHECK-NEXT: }
  }
}

#define INIT(E)                   \
  auto d_##E = clad::hessian(&E); \
  auto d_##E##Ref = clad::hessian(E);

#define TEST(E)                                                         \
  result[0] = result[1] = result[2] = result[3] = 0;                    \
  d_##E.execute(7, 9, result);                                          \
  printf("{%.2f, %.2f, %.2f, %.2f}, ", result[0], result[1], result[2], \
         result[3]);                                                    \
  result[0] = result[1] = result[2] = result[3] = 0;                    \
  d_##E##Ref.execute(7, 9, result);                                     \
  printf("{%.2f, %.2f, %.2f, %.2f}\n", result[0], result[1], result[2], \
         result[3]);

double x = 3;
double y = 5;
int main() {
  double result[4];
  Experiment E(3, 5);
  auto E_Again = E;
  const ExperimentConst E_Const(3, 5);
  volatile ExperimentVolatile E_Volatile(3, 5);
  const volatile ExperimentConstVolatile E_ConstVolatile(3, 5);
  outer::inner::ExperimentNNS E_NNS(3, 5);
  auto E_NNS_Again = E_NNS;
  auto lambda = [](double i, double j) {
    return i*i*j*j;
  };

  // CHECK: inline void operator_call_hessian(double i, double j, double *hessianMatrix) const {
  // CHECK-NEXT:     this->operator_call_darg0_grad(i, j, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
  // CHECK-NEXT:     this->operator_call_darg1_grad(i, j, hessianMatrix + {{2U|2UL|2ULL}}, hessianMatrix + {{3U|3UL|3ULL}});
  // CHECK-NEXT: }

  auto lambdaWithCapture = [&](double i, double jj) {
    return x*i*i*jj + y*i*jj*jj;
  };

  // CHECK: inline void operator_call_hessian(double i, double jj, double *hessianMatrix) const {
  // CHECK-NEXT:     this->operator_call_darg0_grad(i, jj, hessianMatrix + {{0U|0UL|0ULL}}, hessianMatrix + {{1U|1UL|1ULL}});
  // CHECK-NEXT:     this->operator_call_darg1_grad(i, jj, hessianMatrix + {{2U|2UL|2ULL}}, hessianMatrix + {{3U|3UL|3ULL}});
  // CHECK-NEXT: }

  auto lambdaNNS = outer::inner::lambdaNNS;

  INIT(E);
  INIT(E_Again);
  INIT(E_Const);
  INIT(E_Volatile);
  INIT(E_ConstVolatile);
  INIT(E_NNS);
  INIT(E_NNS_Again);
  INIT(lambdaNNS);
  INIT(lambda);
  INIT(lambdaWithCapture);

  TEST(E);                  // CHECK-EXEC: {54.00, 132.00, 132.00, 70.00}, {54.00, 132.00, 132.00, 70.00}
  TEST(E_Again);            // CHECK-EXEC: {54.00, 132.00, 132.00, 70.00}, {54.00, 132.00, 132.00, 70.00}
  TEST(E_Const);            // CHECK-EXEC: {54.00, 132.00, 132.00, 70.00}, {54.00, 132.00, 132.00, 70.00}
  TEST(E_Volatile);         // CHECK-EXEC: {54.00, 132.00, 132.00, 70.00}, {54.00, 132.00, 132.00, 70.00}
  TEST(E_ConstVolatile);    // CHECK-EXEC: {54.00, 132.00, 132.00, 70.00}, {54.00, 132.00, 132.00, 70.00}
  TEST(E_NNS);              // CHECK-EXEC: {54.00, 132.00, 132.00, 70.00}, {54.00, 132.00, 132.00, 70.00}
  TEST(E_NNS_Again);        // CHECK-EXEC: {54.00, 132.00, 132.00, 70.00}, {54.00, 132.00, 132.00, 70.00}
  TEST(lambdaWithCapture);  // CHECK-EXEC: {54.00, 132.00, 132.00, 70.00}, {54.00, 132.00, 132.00, 70.00}
  TEST(lambda);             // CHECK-EXEC: {162.00, 252.00, 252.00, 98.00}, {162.00, 252.00, 252.00, 98.00}
  TEST(lambdaNNS);          // CHECK-EXEC: {162.00, 252.00, 252.00, 98.00}, {162.00, 252.00, 252.00, 98.00}

  E.setX(6);
  E_Again.setX(6);
  E_Const.setX(6);
  E_Volatile.setX(6);
  E_ConstVolatile.setX(6);
  E_NNS.setX(6);
  E_NNS_Again.setX(6);
  x = 6;

  TEST(E);                  // CHECK-EXEC: {108.00, 174.00, 174.00, 70.00}, {108.00, 174.00, 174.00, 70.00}
  TEST(E_Again);            // CHECK-EXEC: {108.00, 174.00, 174.00, 70.00}, {108.00, 174.00, 174.00, 70.00}
  TEST(E_Const);            // CHECK-EXEC: {108.00, 174.00, 174.00, 70.00}, {108.00, 174.00, 174.00, 70.00}
  TEST(E_Volatile);         // CHECK-EXEC: {108.00, 174.00, 174.00, 70.00}, {108.00, 174.00, 174.00, 70.00}
  TEST(E_ConstVolatile);    // CHECK-EXEC: {108.00, 174.00, 174.00, 70.00}, {108.00, 174.00, 174.00, 70.00}
  TEST(E_NNS);              // CHECK-EXEC: {108.00, 174.00, 174.00, 70.00}, {108.00, 174.00, 174.00, 70.00}
  TEST(E_NNS_Again);        // CHECK-EXEC: {108.00, 174.00, 174.00, 70.00}, {108.00, 174.00, 174.00, 70.00}
  TEST(lambdaWithCapture);  // CHECK-EXEC: {108.00, 174.00, 174.00, 70.00}, {108.00, 174.00, 174.00, 70.00}
}
