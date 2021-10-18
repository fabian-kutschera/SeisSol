#include <cxxtest/TestSuite.h>

#include <vector>
#include <math.h>

#include <Numerical_aux/BrentsMethod.h>
#include <Kernels/precision.hpp>



namespace seissol {
  namespace unit_test {
    class BrentsMethodTestSuite;
  }
}

class seissol::unit_test::BrentsMethodTestSuite : public CxxTest::TestSuite {
private:
  static constexpr double epsilon = 10 * std::numeric_limits<real>::epsilon();

  /*
   * Evaluate a polynomial with the Horner scheme
   * $p(x) = \sum_i^N c_i x^i$
   * @param coefficients vector of the coefficients c_i
   * @param x point at which the polynomial shall be evaluated
   */
  real evaluatePolynomial(std::vector<real> coefficients, real x) {
    real result = coefficients.back();
    for (int i = coefficients.size()-2; i >= 0; i--) {
      result *= x;
      result += coefficients[i];
    }
    return result;
  }

public:
  void testRootsPolynomial() {
    // find roots for some polynomials x + x^3 + x^5 + ... x^(2n+1)
    const real root = 0;
    std::vector<real> coefficients = {0, 1};
    for (int i = 0; i < 100; ++i) {
      real calculatedRoot1 =  seissol::rootfinding::findRootsByBrent([&](real x) { return evaluatePolynomial(coefficients, x); }, -1.0, 1.0, epsilon);
      TS_ASSERT_DELTA(root, calculatedRoot1, epsilon);
      coefficients.push_back(0);
      coefficients.push_back(1);
    }
  }

  void testRootsSmoothFunctions() {
    // find roots of some smooth functions
    const real root = 0;

    auto function1 = [](real x) { return std::exp(x) * x; };
    real calculatedRoot1 =  seissol::rootfinding::findRootsByBrent(function1, -1.0, 1.0, epsilon);
    TS_ASSERT_DELTA(root, calculatedRoot1, epsilon);

    auto function2 = [](real x) { return std::asinh(x); };
    real calculatedRoot2 =  seissol::rootfinding::findRootsByBrent(function2, -1.0, 1.0, epsilon);
    TS_ASSERT_DELTA(root, calculatedRoot2, epsilon);

    auto function3 = [](real x) { return std::sin(x); };
    real calculatedRoot3 =  seissol::rootfinding::findRootsByBrent(function3, -1.0, 1.0, epsilon);
    TS_ASSERT_DELTA(root, calculatedRoot3, epsilon);

    auto function4 = [](real x) { return std::sin(x); };
    real calculatedRoot4 =  seissol::rootfinding::findRootsByBrent(function4, -1.0, 1.0, epsilon);
    TS_ASSERT_DELTA(root, calculatedRoot4, epsilon);

    auto function5 = [](real x) { return std::tan(x); };
    real calculatedRoot5 =  seissol::rootfinding::findRootsByBrent(function5, -1.0, 1.0, epsilon);
    TS_ASSERT_DELTA(root, calculatedRoot5, epsilon);

    auto function6 = [](real x) { return std::sinh(x); };
    real calculatedRoot6 =  seissol::rootfinding::findRootsByBrent(function6, -1.0, 1.0, epsilon);
    TS_ASSERT_DELTA(root, calculatedRoot6, epsilon);

    auto function7 = [](real x) { return std::log(0.5*x+1); };
    real calculatedRoot7 =  seissol::rootfinding::findRootsByBrent(function7, -1.0, 1.0, epsilon);
    TS_ASSERT_DELTA(root, calculatedRoot7, epsilon);
  }

  void testRootsNonSmoothFunctions() {
    real root = 0;

    auto function1 = [](real x) -> real { if(x < -0.5) return -0.5; return x; };
    real calculatedRoot1 =  seissol::rootfinding::findRootsByBrent(function1, -1.0, 1.0, epsilon);
    TS_ASSERT_DELTA(root, calculatedRoot1, epsilon);

    auto function2 = [](real x) -> real { if(x < -0.5) return 0.5; if(x > 0.5) return -0.5; return x; };
    real calculatedRoot2 =  seissol::rootfinding::findRootsByBrent(function2, -2.0, 2.0, epsilon);
    TS_ASSERT_DELTA(root, calculatedRoot2, epsilon);
  }

};

