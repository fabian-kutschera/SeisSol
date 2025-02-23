// SPDX-FileCopyrightText: 2022-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause
// SPDX-LicenseComments: Full text under /LICENSE and /LICENSES/
//
// SPDX-FileContributor: Author lists in /AUTHORS and /CITATION.cff

#ifndef SEISSOL_SRC_EQUATIONS_POROELASTIC_MODEL_POROELASTICSETUP_H_
#define SEISSOL_SRC_EQUATIONS_POROELASTIC_MODEL_POROELASTICSETUP_H_

#include <cassert>

#include <Eigen/Dense>
#include <yateto.h>

#include "Kernels/Common.h"
#include "Model/Common.h"
#include "Numerical/Eigenvalues.h"
#include "Numerical/Transformation.h"
#include "generated_code/init.h"

namespace seissol::model {
struct AdditionalPoroelasticParameters {
  Eigen::Matrix<double, 6, 1> alpha;
  double KBar;
  double M;
  double m;
  Eigen::Matrix<double, 6, 6> cBar;
  double rhoBar;
  double rho1;
  double rho2;
  double beta1;
  double beta2;
};

inline AdditionalPoroelasticParameters
    getAdditionalParameters(const PoroElasticMaterial& material) {
  Eigen::Matrix<double, 6, 1> alpha;
  alpha << 1 - (3 * material.lambda + 2 * material.mu) / (3 * material.bulkSolid),
      1 - (3 * material.lambda + 2 * material.mu) / (3 * material.bulkSolid),
      1 - (3 * material.lambda + 2 * material.mu) / (3 * material.bulkSolid), -0.0, -0.0, -0.0;

  Eigen::Matrix<double, 6, 6> c;
  c << material.lambda + 2 * material.mu, material.lambda, material.lambda, 0, 0, 0,
      material.lambda, material.lambda + 2 * material.mu, material.lambda, 0, 0, 0, material.lambda,
      material.lambda, material.lambda + 2 * material.mu, 0, 0, 0, 0, 0, 0, material.mu, 0, 0, 0, 0,
      0, 0, material.mu, 0, 0, 0, 0, 0, 0, material.mu;

  double KBar = material.lambda + 2 * material.mu / 3;
  double M = material.bulkSolid / (1 - material.porosity - KBar / material.bulkSolid +
                                   material.porosity * material.bulkSolid / material.bulkFluid);
  double m = material.rhoFluid * material.tortuosity / material.porosity;

  Eigen::Matrix<double, 6, 6> cBar = c + M * alpha * alpha.transpose();

  double rhoBar = (1 - material.porosity) * material.rho + material.porosity * material.rhoFluid;
  double rho1 = rhoBar - material.rhoFluid * material.rhoFluid / m;
  double rho2 = material.rhoFluid - m * rhoBar / material.rhoFluid;
  double beta1 = material.rhoFluid / m;
  double beta2 = rhoBar / material.rhoFluid;

  return {alpha, KBar, M, m, cBar, rhoBar, rho1, rho2, beta1, beta2};
}

template <typename T>
inline void setToZero(T& AT) {
  AT.setZero();
}

template <typename T>
inline void
    getTransposedCoefficientMatrix(const PoroElasticMaterial& material, unsigned dim, T& AT) {
  setToZero<T>(AT);
  const AdditionalPoroelasticParameters params = getAdditionalParameters(material);
  switch (dim) {
  case 0:
    AT(0, 6) = -1 / params.rho1;
    AT(0, 10) = -1 / params.rho2;
    AT(3, 7) = -1 / params.rho1;
    AT(3, 11) = -1 / params.rho2;
    AT(5, 8) = -1 / params.rho1;
    AT(5, 12) = -1 / params.rho2;

    AT(6, 0) = -params.cBar(0, 0);
    AT(6, 1) = -params.cBar(1, 0);
    AT(6, 2) = -params.cBar(2, 0);
    AT(6, 3) = -params.cBar(5, 0);
    AT(6, 4) = -params.cBar(3, 0);
    AT(6, 5) = -params.cBar(4, 0);
    AT(6, 9) = params.M * params.alpha(0);

    AT(7, 0) = -params.cBar(0, 5);
    AT(7, 1) = -params.cBar(1, 5);
    AT(7, 2) = -params.cBar(2, 5);
    AT(7, 3) = -params.cBar(5, 5);
    AT(7, 4) = -params.cBar(3, 5);
    AT(7, 5) = -params.cBar(4, 5);
    AT(7, 9) = params.M * params.alpha(5);

    AT(8, 0) = -params.cBar(0, 4);
    AT(8, 1) = -params.cBar(1, 4);
    AT(8, 2) = -params.cBar(2, 4);
    AT(8, 3) = -params.cBar(5, 4);
    AT(8, 4) = -params.cBar(3, 4);
    AT(8, 5) = -params.cBar(4, 4);
    AT(8, 9) = params.M * params.alpha(4);

    AT(9, 6) = -params.beta1 / params.rho1;
    AT(9, 10) = -params.beta2 / params.rho2;

    AT(10, 0) = -params.M * params.alpha(0);
    AT(10, 1) = -params.M * params.alpha(1);
    AT(10, 2) = -params.M * params.alpha(2);
    AT(10, 3) = -params.M * params.alpha(5);
    AT(10, 4) = -params.M * params.alpha(3);
    AT(10, 5) = -params.M * params.alpha(4);
    AT(10, 9) = params.M;
    break;
  case 1:
    AT(1, 7) = -1 / params.rho1;
    AT(1, 11) = -1 / params.rho2;
    AT(3, 6) = -1 / params.rho1;
    AT(3, 10) = -1 / params.rho2;
    AT(4, 8) = -1 / params.rho1;
    AT(4, 12) = -1 / params.rho2;

    AT(6, 0) = -params.cBar(0, 5);
    AT(6, 1) = -params.cBar(1, 5);
    AT(6, 2) = -params.cBar(2, 5);
    AT(6, 3) = -params.cBar(5, 5);
    AT(6, 4) = -params.cBar(3, 5);
    AT(6, 5) = -params.cBar(4, 5);
    AT(6, 9) = params.M * params.alpha(5);

    AT(7, 0) = -params.cBar(0, 1);
    AT(7, 1) = -params.cBar(1, 1);
    AT(7, 2) = -params.cBar(2, 1);
    AT(7, 3) = -params.cBar(5, 1);
    AT(7, 4) = -params.cBar(3, 1);
    AT(7, 5) = -params.cBar(4, 1);
    AT(7, 9) = params.M * params.alpha(1);

    AT(8, 0) = -params.cBar(0, 3);
    AT(8, 1) = -params.cBar(1, 3);
    AT(8, 2) = -params.cBar(2, 3);
    AT(8, 3) = -params.cBar(5, 3);
    AT(8, 4) = -params.cBar(3, 3);
    AT(8, 5) = -params.cBar(4, 3);
    AT(8, 9) = params.M * params.alpha(3);

    AT(9, 7) = -params.beta1 / params.rho1;
    AT(9, 11) = -params.beta2 / params.rho2;

    AT(11, 0) = -params.M * params.alpha(0);
    AT(11, 1) = -params.M * params.alpha(1);
    AT(11, 2) = -params.M * params.alpha(2);
    AT(11, 3) = -params.M * params.alpha(5);
    AT(11, 4) = -params.M * params.alpha(3);
    AT(11, 5) = -params.M * params.alpha(4);
    AT(11, 9) = params.M;
    break;
  case 2:
    AT(2, 8) = -1 / params.rho1;
    AT(2, 12) = -1 / params.rho2;
    AT(4, 7) = -1 / params.rho1;
    AT(4, 11) = -1 / params.rho2;
    AT(5, 6) = -1 / params.rho1;
    AT(5, 10) = -1 / params.rho2;

    AT(6, 0) = -params.cBar(0, 4);
    AT(6, 1) = -params.cBar(1, 4);
    AT(6, 2) = -params.cBar(2, 4);
    AT(6, 3) = -params.cBar(5, 4);
    AT(6, 4) = -params.cBar(3, 4);
    AT(6, 5) = -params.cBar(4, 4);
    AT(6, 9) = params.M * params.alpha(4);

    AT(7, 0) = -params.cBar(0, 3);
    AT(7, 1) = -params.cBar(1, 3);
    AT(7, 2) = -params.cBar(2, 3);
    AT(7, 3) = -params.cBar(5, 3);
    AT(7, 4) = -params.cBar(3, 3);
    AT(7, 5) = -params.cBar(4, 3);
    AT(7, 9) = params.M * params.alpha(3);

    AT(8, 0) = -params.cBar(0, 2);
    AT(8, 1) = -params.cBar(1, 2);
    AT(8, 2) = -params.cBar(2, 2);
    AT(8, 3) = -params.cBar(5, 2);
    AT(8, 4) = -params.cBar(3, 2);
    AT(8, 5) = -params.cBar(4, 2);
    AT(8, 9) = params.M * params.alpha(2);

    AT(9, 8) = -params.beta1 / params.rho1;
    AT(9, 12) = -params.beta2 / params.rho2;

    AT(12, 0) = -params.M * params.alpha(0);
    AT(12, 1) = -params.M * params.alpha(1);
    AT(12, 2) = -params.M * params.alpha(2);
    AT(12, 3) = -params.M * params.alpha(5);
    AT(12, 4) = -params.M * params.alpha(3);
    AT(12, 5) = -params.M * params.alpha(4);
    AT(12, 9) = params.M;
    break;

  default:
    logError() << "Cannot create transposed coefficient matrix for dimension " << dim
               << ", has to be either 0, 1 or 2.";
  }
}

template <typename T>
inline void getTransposedSourceCoefficientTensor(const PoroElasticMaterial& material, T& ET) {
  const AdditionalPoroelasticParameters params = getAdditionalParameters(material);
  const double e1 = params.beta1 * material.viscosity / (params.rho1 * material.permeability);
  const double e2 = params.beta2 * material.viscosity / (params.rho2 * material.permeability);

  ET.setZero();
  ET(10, 6) = e1;
  ET(11, 7) = e1;
  ET(12, 8) = e1;

  ET(10, 10) = e2;
  ET(11, 11) = e2;
  ET(12, 12) = e2;
}

template <typename T>
void getTransposedFreeSurfaceGodunovState(MaterialType materialtype,
                                          T& QgodLocal,
                                          T& QgodNeighbor,
                                          Eigen::Matrix<double, 13, 13>& R) {
  if (materialtype != MaterialType::Poroelastic) {
    logError() << "This is only used for poroelastic materials. You should never end up here.";
  }

  constexpr size_t RelevantQuantities =
      seissol::model::MaterialT::NumQuantities - 6 * NUMBER_OF_RELAXATION_MECHANISMS;
  for (size_t i = 0; i < RelevantQuantities; i++) {
    for (size_t j = 0; j < RelevantQuantities; j++) {
      QgodNeighbor(i, j) = std::numeric_limits<double>::signaling_NaN();
    }
  }

  QgodLocal.setZero();
  using Matrix44 = Eigen::Matrix<double, 4, 4>;
  using Matrix64 = Eigen::Matrix<double, 6, 4>;

  std::array<int, 4> tractionIndices = {0, 3, 5, 9};
  std::array<int, 6> velocityIndices = {6, 7, 8, 10, 11, 12};
  std::array<int, 4> columnIndices = {0, 1, 2, 3};
  Matrix44 R11 = R(tractionIndices, columnIndices);
  Matrix64 R21 = R(velocityIndices, columnIndices);
  Matrix64 S = (-(R21 * R11.inverse())).eval();
  setBlocks(QgodLocal, S, tractionIndices, velocityIndices);
}

// zeroThreshold has default parameter 1e-7, see src/Model/Common.hpp
template <>
inline seissol::eigenvalues::Eigenpair<std::complex<double>,
                                       seissol::model::MaterialT::NumQuantities>
    getEigenDecomposition(const PoroElasticMaterial& material, double zeroThreshold) {
  std::array<std::complex<double>,
             seissol::model::MaterialT::NumQuantities * seissol::model::MaterialT::NumQuantities>
      AT;
  auto ATView = yateto::DenseTensorView<2, std::complex<double>>(
      AT.data(),
      {seissol::model::MaterialT::NumQuantities, seissol::model::MaterialT::NumQuantities});
  getTransposedCoefficientMatrix(material, 0, ATView);
  std::array<std::complex<double>,
             seissol::model::MaterialT::NumQuantities * seissol::model::MaterialT::NumQuantities>
      A;
  // transpose AT to get A
  for (std::size_t i = 0; i < seissol::model::MaterialT::NumQuantities; i++) {
    for (std::size_t j = 0; j < seissol::model::MaterialT::NumQuantities; j++) {
      A[i + seissol::model::MaterialT::NumQuantities * j] =
          AT[seissol::model::MaterialT::NumQuantities * i + j];
    }
  }
  eigenvalues::Eigenpair<std::complex<double>, seissol::model::MaterialT::NumQuantities> eigenpair;
  eigenvalues::computeEigenvaluesWithLapack(A, eigenpair);

#ifndef NDEBUG
  using CMatrix = Eigen::Matrix<std::complex<double>,
                                seissol::model::MaterialT::NumQuantities,
                                seissol::model::MaterialT::NumQuantities>;
  using CVector = Eigen::Matrix<std::complex<double>, seissol::model::MaterialT::NumQuantities, 1>;
  CMatrix eigenvectors = CMatrix(eigenpair.vectors.data());
  CVector eigenvalues = CVector(eigenpair.values.data());
  // check number of eigenvalues
  // also check that the imaginary parts are zero
  int evNeg = 0;
  int evPos = 0;
  for (std::size_t i = 0; i < seissol::model::MaterialT::NumQuantities; ++i) {
    assert(std::abs(eigenvalues(i).imag()) < zeroThreshold);
    if (eigenvalues(i).real() < -zeroThreshold) {
      ++evNeg;
    } else if (eigenvalues(i).real() > zeroThreshold) {
      ++evPos;
    }
  }
  assert(evNeg == 4);
  assert(evPos == 4);

  // check whether eigensolver is good enough
  CMatrix coeff(A.data());
  const CMatrix matrixMult = coeff * eigenvectors;
  CMatrix eigenvalueMatrix = CMatrix::Zero();
  for (size_t i = 0; i < seissol::model::MaterialT::NumQuantities; i++) {
    eigenvalueMatrix(i, i) = eigenvalues(i);
  }
  const CMatrix vectorMult = eigenvectors * eigenvalueMatrix;
  const CMatrix diff = matrixMult - vectorMult;
  const double norm = diff.norm();

  std::stringstream messageStream;
  messageStream << "Residual " << norm << " is larger than " << zeroThreshold
                << ": Eigensolver is not accurate enough";
  assert((messageStream.str().c_str(), norm < zeroThreshold));
#endif
  return eigenpair;
};

template <>
inline void getTransposedGodunovState(const PoroElasticMaterial& local,
                                      const PoroElasticMaterial& neighbor,
                                      FaceType faceType,
                                      init::QgodLocal::view::type& QgodLocal,
                                      init::QgodNeighbor::view::type& QgodNeighbor) {
  // Will be used to check, whether numbers are (numerically) zero
  constexpr auto zeroThreshold = 1e-7;
  using CMatrix = Eigen::Matrix<std::complex<double>,
                                seissol::model::MaterialT::NumQuantities,
                                seissol::model::MaterialT::NumQuantities>;
  using Matrix = Eigen::Matrix<double,
                               seissol::model::MaterialT::NumQuantities,
                               seissol::model::MaterialT::NumQuantities>;
  using CVector = Eigen::Matrix<std::complex<double>, seissol::model::MaterialT::NumQuantities, 1>;

  auto splitEigenDecomposition = [&zeroThreshold](const PoroElasticMaterial& material) {
    auto eigenpair = getEigenDecomposition(material, zeroThreshold);
    return std::pair<CVector, CMatrix>{eigenpair.getValuesAsVector(),
                                       eigenpair.getVectorsAsMatrix()};
  };

  auto [localEigenvalues, localEigenvectors] = splitEigenDecomposition(local);
  auto [neighborEigenvalues, neighborEigenvectors] = splitEigenDecomposition(neighbor);

  CMatrix chiMinus = CMatrix::Zero();
  CMatrix chiPlus = CMatrix::Zero();
  for (int i = 0; i < 13; i++) {
    if (localEigenvalues(i).real() < -zeroThreshold) {
      chiMinus(i, i) = 1.0;
    }
    if (localEigenvalues(i).real() > zeroThreshold) {
      chiPlus(i, i) = 1.0;
    }
  }
  CMatrix R = localEigenvectors * chiMinus + neighborEigenvectors * chiPlus;
  // set null space eigenvectors manually
  R(1, 4) = 1.0;
  R(2, 5) = 1.0;
  R(12, 6) = 1.0;
  R(11, 7) = 1.0;
  R(4, 8) = 1.0;
  if (faceType == FaceType::FreeSurface) {
    Matrix realR = R.real();
    getTransposedFreeSurfaceGodunovState(MaterialType::Poroelastic, QgodLocal, QgodNeighbor, realR);
  } else {
    CMatrix invR = R.inverse();
    CMatrix godunovMinus = R * chiMinus * invR;
    CMatrix godunovPlus = R * chiPlus * invR;

    for (unsigned i = 0; i < QgodLocal.shape(1); ++i) {
      for (unsigned j = 0; j < QgodLocal.shape(0); ++j) {
        QgodLocal(j, i) = godunovPlus(i, j).real();
        QgodNeighbor(j, i) = godunovMinus(i, j).real();
        assert(std::abs(godunovPlus(j, i).imag()) < zeroThreshold);
        assert(std::abs(godunovMinus(j, i).imag()) < zeroThreshold);
      }
    }
  }
}

template <typename Tview>
inline void calcZinv(yateto::DenseTensorView<2, real, unsigned>& Zinv,
                     Tview& sourceMatrix,
                     size_t quantity,
                     real timeStepWidth) {
  using Matrix = Eigen::Matrix<real, ConvergenceOrder, ConvergenceOrder>;
  using Vector = Eigen::Matrix<real, ConvergenceOrder, 1>;

  Matrix Z(init::Z::Values);
  // sourceMatrix[i,i] = 0 for i < 10
  // This is specific to poroelasticity, so change this for another equation
  // We need this check, because otherwise the lookup sourceMatrix(quantity, quantity) fails
  if (quantity >= 10) {
    Z -= timeStepWidth * sourceMatrix(quantity, quantity) * Matrix::Identity();
  }

  auto solver = Z.colPivHouseholderQr();
  for (std::size_t col = 0; col < ConvergenceOrder; col++) {
    Vector rhs = Vector::Zero();
    rhs(col) = 1.0;
    auto ZinvCol = solver.solve(rhs);
    for (std::size_t row = 0; row < ConvergenceOrder; row++) {
      // save as transposed
      Zinv(col, row) = ZinvCol(row);
    }
  }
}

// constexpr for loop since we need to instatiate the view templates
template <size_t iStart, size_t iEnd, typename Tview>
struct zInvInitializerForLoop {
  zInvInitializerForLoop(
      real ZinvData[seissol::model::MaterialT::NumQuantities][ConvergenceOrder * ConvergenceOrder],
      Tview& sourceMatrix,
      real timeStepWidth) {
    auto Zinv = init::Zinv::view<iStart>::create(ZinvData[iStart]);
    calcZinv(Zinv, sourceMatrix, iStart, timeStepWidth);
    if constexpr (iStart < iEnd - 1) {
      zInvInitializerForLoop<iStart + 1, iEnd, Tview>(ZinvData, sourceMatrix, timeStepWidth);
    }
  };
};

inline void initializeSpecificLocalData(const PoroElasticMaterial& material,
                                        real timeStepWidth,
                                        PoroelasticLocalData* localData) {
  auto sourceMatrix = init::ET::view::create(localData->sourceMatrix);
  sourceMatrix.setZero();
  getTransposedSourceCoefficientTensor(material, sourceMatrix);

  zInvInitializerForLoop<0, seissol::model::MaterialT::NumQuantities, decltype(sourceMatrix)>(
      localData->Zinv, sourceMatrix, timeStepWidth);
  std::fill(localData->G, localData->G + seissol::model::MaterialT::NumQuantities, 0.0);
  localData->G[10] = sourceMatrix(10, 6);
  localData->G[11] = sourceMatrix(11, 7);
  localData->G[12] = sourceMatrix(12, 8);

  localData->typicalTimeStepWidth = timeStepWidth;
}
} // namespace seissol::model

#endif // SEISSOL_SRC_EQUATIONS_POROELASTIC_MODEL_POROELASTICSETUP_H_
