#ifndef SEISSOL_DROUTOUT_DR_BASE_HPP
#define SEISSOL_DROUTOUT_DR_BASE_HPP

#include "Initializer/InputAux.hpp"
#include "Initializer/DynamicRupture.h"
#include "DynamicRupture/Output/ParametersInitializer.hpp"
#include "DynamicRupture/Output/Builders/ElementWiseBuilder.hpp"
#include "DynamicRupture/Output/Builders/PickPointBuilder.hpp"
#include <iostream>
#include <memory>

namespace seissol::dr::output {
class Base {
  public:
  virtual ~Base() = default;

  void setInputParam(const YAML::Node& inputData, const MeshReader& mesher) {
    using namespace initializers;

    ParametersInitializer reader(inputData);
    generalParams = reader.getDrGeneralParams();

    // adjust general output parameters
    generalParams.isRfTimeOn = generalParams.isRfOutputOn;
    if (generalParams.isDsOutputOn && !generalParams.isRfOutputOn) {
      generalParams.isRfOutputOn = true;
      generalParams.isRfTimeOn = true;
    }

    PickpointParamsT ppParams;
    ElementwiseFaultParamsT ewParams;

    switch (generalParams.outputPointType) {
    case OutputType::None:
      break;

    case OutputType::AtPickpoint:
      ppOutputBuilder = std::make_unique<PickPointBuilder>();
      ppOutputBuilder->setMeshReader(&mesher);
      ppOutputBuilder->setParams(reader.getPickPointParams());
      break;

    case OutputType::Elementwise:
      ewOutputBuilder = std::make_unique<ElementWiseBuilder>();
      ewOutputBuilder->setMeshReader(&mesher);
      ewOutputBuilder->setParams(reader.getElementwiseFaultParams());
      break;

    case OutputType::AtPickpointAndElementwise:
      ppOutputBuilder = std::make_unique<PickPointBuilder>();
      ppOutputBuilder->setMeshReader(&mesher);
      ppOutputBuilder->setParams(reader.getPickPointParams());

      ewOutputBuilder = std::make_unique<ElementWiseBuilder>();
      ewOutputBuilder->setMeshReader(&mesher);
      ewOutputBuilder->setParams(reader.getElementwiseFaultParams());
      break;

    default:
      throw std::runtime_error("Unknown fault output type (not 3,4,5)");
    }
  }
  void setDrData(seissol::initializers::LTSTree* userDrTree,
                 seissol::initializers::DynamicRupture* drDescription) {
    drTree = userDrTree;
    dynRup = drDescription;
  }

  void init();
  void initFaceToLtsMap();

  void writePickpointOutput(double time, double dt);
  bool isAtPickpoint(double time, double dt);
  void updateElementwiseOutput();

  virtual void tiePointers(seissol::initializers::Layer& layerData,
                           seissol::initializers::DynamicRupture* description,
                           seissol::Interoperability& e_interoperability);

  virtual void postCompute(seissol::initializers::DynamicRupture& DynRup) = 0;

  protected:
  void initElementwiseOutput();
  void initPickpointOutput();

  [[nodiscard]] std::string constructPickpointReceiverFileName(int receiverGlobalIndex) const;
  void calcFaultOutput(OutputType type, OutputData& state, double time = 0.0);

  GeneralParamsT generalParams;

  std::unique_ptr<ElementWiseBuilder> ewOutputBuilder{nullptr};
  std::unique_ptr<PickPointBuilder> ppOutputBuilder{nullptr};

  seissol::initializers::LTSTree* drTree{nullptr};
  seissol::initializers::DynamicRupture* dynRup{nullptr};

  std::vector<std::pair<seissol::initializers::Layer*, size_t>> faceToLtsMap{};
  size_t iterationStep{0};
};
} // namespace seissol::dr::output
#endif // SEISSOL_DROUTOUT_DR_BASE_HPP
