#include "SeisSolParameters.h"
#include <Initializer/Parameters/CubeGeneratorParameters.h>
#include <Initializer/Parameters/DRParameters.h>
#include <Initializer/Parameters/InitializationParameters.h>
#include <Initializer/Parameters/LtsParameters.h>
#include <Initializer/Parameters/MeshParameters.h>
#include <Initializer/Parameters/ModelParameters.h>
#include <Initializer/Parameters/OutputParameters.h>
#include <Initializer/Parameters/ParameterReader.h>
#include <Initializer/Parameters/SourceParameters.h>
#include <utils/logger.h>

namespace seissol::initializer::parameters {

SeisSolParameters readSeisSolParameters(ParameterReader* parameterReader) {
  logInfo() << "Reading SeisSol parameter file...";

  const CubeGeneratorParameters cubeGeneratorParameters =
      readCubeGeneratorParameters(parameterReader);
  const DRParameters drParameters = readDRParameters(parameterReader);
  const InitializationParameters initializationParameters =
      readInitializationParameters(parameterReader);
  const MeshParameters meshParameters = readMeshParameters(parameterReader);
  const ModelParameters modelParameters = readModelParameters(parameterReader);
  const OutputParameters outputParameters = readOutputParameters(parameterReader);
  const SourceParameters sourceParameters = readSourceParameters(parameterReader);
  const TimeSteppingParameters timeSteppingParameters = readTimeSteppingParameters(parameterReader);

  parameterReader->warnDeprecated({"boundaries",
                                   "rffile",
                                   "inflowbound",
                                   "inflowboundpwfile",
                                   "inflowbounduin",
                                   "source110",
                                   "source15",
                                   "source1618",
                                   "source17",
                                   "source19",
                                   "spongelayer",
                                   "sponges",
                                   "analysis",
                                   "analysisfields",
                                   "debugging"});

  logInfo() << "SeisSol parameter file read successfully.";

  auto printYesNo = [](bool yesno) { return yesno ? "yes" : "no"; };

  logInfo() << "Model information:";
  logInfo() << "Elastic model:" << printYesNo(isModelElastic());
  logInfo() << "Viscoelastic model:" << printYesNo(isModelViscoelastic());
  logInfo() << "Anelastic model:" << printYesNo(isModelAnelastic());
  logInfo() << "Poroelastic model:" << printYesNo(isModelPoroelastic());
  logInfo() << "Anisotropic model:" << printYesNo(isModelAnisotropic());
  logInfo() << "Plasticity:" << printYesNo(modelParameters.plasticity);

  return SeisSolParameters{cubeGeneratorParameters,
                           drParameters,
                           initializationParameters,
                           meshParameters,
                           modelParameters,
                           outputParameters,
                           sourceParameters,
                           timeSteppingParameters};
}
} // namespace seissol::initializer::parameters
