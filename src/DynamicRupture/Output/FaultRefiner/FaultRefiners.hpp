#ifndef SEISSOL_INTERFACE_HPP
#define SEISSOL_INTERFACE_HPP

#include "DynamicRupture/Output/DataTypes.hpp"
#include <memory>
#include <tuple>

namespace seissol::dr::output::refiner {
RefinerType convertToType(int strategy);

class FaultRefiner;
std::unique_ptr<FaultRefiner> get(RefinerType strategy);

class FaultRefiner {
  public:
  struct Data {
    int refinementLevel{};
    int faultFaceIndex{};
    int localFaceSideId{};
  };

  virtual int getNumSubTriangles() = 0;
  virtual void
      refineAndAccumulate(Data data, ExtTriangle referenceFace, ExtTriangle globalFace) = 0;

  ReceiverPointsT&& moveAllReceiverPoints() { return std::move(points); }
  ReceiverPointsT getAllReceiverPoints() { return points; }

  protected:
  ReceiverPointsT points{};

  using pointsPair = std::pair<ExtVrtxCoords, ExtVrtxCoords>;
  void repeat(Data data, pointsPair& point1, pointsPair& point2, pointsPair& point3);
};

class TripleFaultFaceRefiner : public FaultRefiner {
  public:
  int getNumSubTriangles() override { return 3; }
  void refineAndAccumulate(Data data, ExtTriangle referenceFace, ExtTriangle globalFace) final;
};

class QuadFaultFaceRefiner : public FaultRefiner {
  public:
  int getNumSubTriangles() final { return 4; }
  void refineAndAccumulate(Data data, ExtTriangle referenceFace, ExtTriangle globalFace) final;
};
} // namespace seissol::dr::output::refiner
#endif // SEISSOL_INTERFACE_HPP
