#ifndef SEISSOL_BRENTSMETHOD_H
#define SEISSOL_BRENTSMETHOD_H

/*
 * We implement Brent's method for root finding:
 * https://en.wikipedia.org/wiki/Brent%27s_method
 */

#include <utils/logger.h>

namespace seissol::rootfinding {

  template<typename Func, typename T>
  T findRootsByBrent(Func f, T a, T b, T epsilon=1e-14) {
  T fEvalAtA = f(a);
  T fEvalAtB = f(b);
  if(fEvalAtA * fEvalAtB > 0) {
    // return NaN
    return 0.0/0.0;
  }
  if(std::abs(fEvalAtA) < std::abs(fEvalAtB)) {
    std::swap(a, b);
    std::swap(fEvalAtA, fEvalAtB);
  }
  T c = a;
  T d = (T)0;
  bool mflag = true;
  while (std::abs(fEvalAtB) > epsilon && std::abs(b - a) > epsilon) {
    T fEvalAtC = f(c);
    T s;
    if(std::abs(fEvalAtA - fEvalAtC) > epsilon && std::abs(fEvalAtB - fEvalAtC) > 0) {
      s = a*fEvalAtB*fEvalAtC / (fEvalAtA -fEvalAtB) / (fEvalAtA - fEvalAtC) +
          b*fEvalAtA*fEvalAtC / (fEvalAtB -fEvalAtA) / (fEvalAtB - fEvalAtC) +
          c*fEvalAtA*fEvalAtB / (fEvalAtC -fEvalAtA) / (fEvalAtC - fEvalAtB);
    } else {
      s = b - fEvalAtB * (b-a) / (fEvalAtB - fEvalAtA);
    }
    bool condition1 = s < 0.25*(3*a+b) || s > b;
    bool condition2 = mflag && std::abs(s-b) >= 0.5 * std::abs(b-c);
    bool condition3 = !mflag && std::abs(s-b) >= 0.5 * std::abs(c-d);
    bool condition4 = mflag && std::abs(b-c) < std::abs(epsilon);
    bool condition5 = !mflag && std::abs(c-d) < std::abs(epsilon);
    if(condition1 || condition2 || condition3 || condition4 || condition5) {
      s = 0.5 * (a+b);
      mflag = true;
    } else {
      mflag = false;
    }
    T fevalAtS = f(s);
    d = c;
    c = b;
    fEvalAtC = fEvalAtB;
    if(fEvalAtA*fevalAtS < 0) {
      b = s;
      fEvalAtB = fevalAtS;
    } else {
      a = s;
      fEvalAtA = fevalAtS;
    }
    if(std::abs(fEvalAtA) < std::abs(fEvalAtB)) {
      std::swap(a, b);
      std::swap(fEvalAtA, fEvalAtB);
    }
  }
  return b;
}

}


#endif //SEISSOL_BRENTSMETHOD_H
