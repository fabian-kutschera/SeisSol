#ifndef SEISSOL_TRIPLEFAULTFACEREFINER_HPP
#define SEISSOL_TRIPLEFAULTFACEREFINER_HPP

#include "FaultRefiners.hpp"
#include "DynamicRupture/Output/OutputAux.hpp"
#include "utils/logger.h"
#include <memory>

namespace seissol::dr::output::refiner {
RefinerType convertToType(int strategy) {
  switch (strategy) {
  case 1:
    return RefinerType::Triple;
  case 2:
    return RefinerType::Quad;
  default:
    logError() << "Unknown refinement strategy for Fault Face Refiner";
  }
}

std::unique_ptr<FaultRefiner> get(RefinerType strategy) {
  switch (strategy) {
  case RefinerType::Triple:
    return std::unique_ptr<FaultRefiner>(new TripleFaultFaceRefiner);
  case RefinerType::Quad:
    return std::unique_ptr<FaultRefiner>(new QuadFaultFaceRefiner);
  }
}

void FaultRefiner::repeat(Data data, pointsPair& point1, pointsPair& point2, pointsPair& point3) {
  static constexpr size_t GLOBAL = 0;
  static constexpr size_t REFERENCE = 1;

  ExtTriangle subReferenceFace(
      std::get<REFERENCE>(point1), std::get<REFERENCE>(point2), std::get<REFERENCE>(point3));
  ExtTriangle subGlobalFace(
      std::get<GLOBAL>(point1), std::get<GLOBAL>(point2), std::get<GLOBAL>(point3));
  refineAndAccumulate({data.refinementLevel - 1, data.faultFaceIndex, data.localFaceSideId},
                      subReferenceFace,
                      subGlobalFace);
}

void TripleFaultFaceRefiner::refineAndAccumulate(Data data,
                                                 ExtTriangle referenceFace,
                                                 ExtTriangle globalFace) {

  if (data.refinementLevel == 0) {
    ReceiverPointT receiver{};
    receiver.isInside = true;
    receiver.faultFaceIndex = data.faultFaceIndex;
    receiver.localFaceSideId = data.localFaceSideId;
    receiver.globalReceiverIndex = points.size();
    receiver.global = getMidTrianglePoint(globalFace);
    receiver.reference = getMidTrianglePoint(referenceFace);
    receiver.globalTriangle = globalFace;

    points.push_back(receiver);
    return;
  }

  auto midPoint = std::make_pair(getMidTrianglePoint(globalFace), getMidTrianglePoint(globalFace));
  std::array<pointsPair, 3> points{};
  for (size_t i = 0; i < 3; ++i) {
    points[i] = std::make_pair(globalFace[0], referenceFace[0]);
  }

  repeat(data, points[0], points[1], midPoint);
  repeat(data, midPoint, points[1], points[2]);
  repeat(data, points[0], midPoint, points[2]);
}

void QuadFaultFaceRefiner::refineAndAccumulate(Data data,
                                               ExtTriangle referenceFace,
                                               ExtTriangle globalFace) {

  if (data.refinementLevel == 0) {
    ReceiverPointT receiver{};

    receiver.isInside = true;
    receiver.faultFaceIndex = data.faultFaceIndex;
    receiver.localFaceSideId = data.localFaceSideId;
    receiver.globalReceiverIndex = points.size();
    receiver.global = getMidTrianglePoint(globalFace);
    receiver.reference = getMidTrianglePoint(referenceFace);
    receiver.globalTriangle = globalFace;

    points.push_back(receiver);
    return;
  }

  auto split = [&globalFace, &referenceFace](size_t pointIndex1, size_t pointIndex2) {
    return std::make_pair(getMidPoint(globalFace[pointIndex1], globalFace[pointIndex2]),
                          getMidPoint(referenceFace[pointIndex1], referenceFace[pointIndex2]));
  };

  auto midPoint1 = split(0, 1);
  auto midPoint2 = split(1, 2);
  auto midPoint3 = split(2, 0);

  pointsPair trianglePoint = std::make_pair(globalFace.p1, referenceFace.p1);
  repeat(data, trianglePoint, midPoint1, midPoint3);

  trianglePoint = std::make_pair(globalFace.p2, referenceFace.p2);
  repeat(data, midPoint1, trianglePoint, midPoint2);

  repeat(data, midPoint1, midPoint2, midPoint3);

  trianglePoint = std::make_pair(globalFace.p3, referenceFace.p3);
  repeat(data, midPoint3, midPoint2, trianglePoint);
}
} // namespace seissol::dr::output::refiner

#endif // SEISSOL_TRIPLEFAULTFACEREFINER_HPP
