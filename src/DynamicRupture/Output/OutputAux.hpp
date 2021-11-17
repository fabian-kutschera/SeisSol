#ifndef SEISSOL_DROUTPUTAUX_HPP
#define SEISSOL_DROUTPUTAUX_HPP

#include "DataTypes.hpp"
#include "Geometry/MeshReader.h"
#include <memory>

namespace seissol {
template <int N, typename T>
auto reshape(T* ptr) -> T (*)[N] {
  return reinterpret_cast<T(*)[N]>(ptr);
}
} // namespace seissol

namespace seissol::dr {
int getElementVertexId(int localSideId, int localFaceVertexId);

ExtTriangle getReferenceFace(int localSideId);

void computeStrikeAndDipVectors(const VrtxCoords normal, VrtxCoords strike, VrtxCoords dip);

ExtVrtxCoords getMidTrianglePoint(const ExtTriangle& triangle);

ExtVrtxCoords getMidPoint(const ExtVrtxCoords& p1, const ExtVrtxCoords& p2);

std::tuple<unsigned, std::shared_ptr<double[]>, std::shared_ptr<double[]>>
    generateTriangleQuadrature(unsigned polyDegree);

void assignNearestGaussianPoints(ReceiverPointsT& geoPoints);

std::pair<int, double> getNearestFacePoint(const double targetPoint[2],
                                           const double (*facePoints)[2],
                                           unsigned numFacePoints);

void projectPointToFace(ExtVrtxCoords& point, const ExtTriangle& face, const VrtxCoords faceNormal);

PlusMinusBasisFunctionsT getPlusMinusBasisFunctions(const VrtxCoords point,
                                                    const VrtxCoords* plusElementCoords[4],
                                                    const VrtxCoords* minusElementCoords[4]);

std::vector<double> getAllVertices(const seissol::dr::ReceiverPointsT& receiverPoints);
std::vector<unsigned int> getCellConnectivity(const seissol::dr::ReceiverPointsT& receiverPoints);

template <int Size>
std::unique_ptr<int[]> convertMaskFromBoolToInt(const std::array<bool, Size>& boolMask) {
  auto intMask = std::unique_ptr<int[]>(new int[boolMask.size()]);

  for (size_t i = 0; i < boolMask.size(); ++i) {
    intMask[i] = static_cast<int>(boolMask[i]);
  }

  return intMask;
}
} // namespace seissol::dr

#endif // SEISSOL_DROUTPUTAUX_HPP
