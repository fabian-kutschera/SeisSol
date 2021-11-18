#ifndef SEISSOL_PARAMETERS_H
#define SEISSOL_PARAMETERS_H

#include <yaml-cpp/yaml.h>

#include "DynamicRupture/Typedefs.hpp"
#include "Initializer/InputAux.hpp"
#include "Kernels/precision.hpp"
#include "Typedefs.hpp"

#include <Eigen/Dense>

namespace seissol::dr {
/**
 * Saves all dynamic rupture parameter read from parameter.par file
 * if values are not defined they are set to an initial value (mostly 0)
 */
struct DRParameters {
  bool isDynamicRuptureEnabled{true};
  int outputPointType{3};
  Eigen::Vector3d referencePoint;
  int slipRateOutputType{1};
  FrictionLawType frictionLawType{0};
  int backgroundType{0};
  bool isRfOutputOn{false};
  bool isDsOutputOn{false};
  bool isMagnitudeOutputOn{false};
  bool isEnergyRateOutputOn{false};
  bool isThermalPressureOn{false};
  int energyRatePrintTimeInterval{1};
  bool isInstaHealingOn{false};
  real t0{0.0};
  struct RateAndState {
    real f0{0.0};
    real a{0.0};
    real b{0.0};
    real sr0{0.0};
    real initialSlipRate1{0.0};
    real initialSlipRate2{0.0};
    real muW{0.0};
    real tpAlpha{0.0};
    real rhoC{0.0};
    real tpLambda{0.0};
    real initialTemperature{0.0};
    real initialPressure{0.0};
  } rateAndState;
  real vStar{0.0}; // Prakash-Clifton regularization parameter
  real prakashLength{0.0};
  std::string faultFileName{""};
};

inline DRParameters readParametersFromYaml(YAML::Node& params) {
  DRParameters drParameters;
  const YAML::Node& yamlParams = params["dynamicrupture"];

  if (params["dynamicrupture"]) {
    double xref = 0.0;
    initializers::updateIfExists(yamlParams, "xref", xref);
    double yref = 0.0;
    initializers::updateIfExists(yamlParams, "yref", yref);
    double zref = 0.0;
    initializers::updateIfExists(yamlParams, "zref", zref);
    drParameters.referencePoint = {xref, yref, zref};

    initializers::updateIfExists(yamlParams, "outputpointtype", drParameters.outputPointType);
    initializers::updateIfExists(yamlParams, "sliprateoutputtype", drParameters.slipRateOutputType);
    initializers::updateIfExists(yamlParams, "fl", drParameters.frictionLawType);
    initializers::updateIfExists(yamlParams, "backgroundtype", drParameters.backgroundType);
    initializers::updateIfExists(yamlParams, "rf_output_on", drParameters.isRfOutputOn);
    initializers::updateIfExists(yamlParams, "ds_output_on", drParameters.isDsOutputOn);
    initializers::updateIfExists(
        yamlParams, "magnitude_output_on", drParameters.isMagnitudeOutputOn);
    initializers::updateIfExists(
        yamlParams, "energy_rate_output_on", drParameters.isEnergyRateOutputOn);
    initializers::updateIfExists(yamlParams, "thermalpress", drParameters.isThermalPressureOn);
    initializers::updateIfExists(
        yamlParams, "energy_rate_printtimeinterval", drParameters.backgroundType);
    initializers::updateIfExists(yamlParams, "inst_healing", drParameters.isInstaHealingOn);
    initializers::updateIfExists(yamlParams, "t_0", drParameters.t0);
    initializers::updateIfExists(yamlParams, "rs_f0", drParameters.rateAndState.f0);
    initializers::updateIfExists(yamlParams, "rs_a", drParameters.rateAndState.a);
    initializers::updateIfExists(yamlParams, "rs_b", drParameters.rateAndState.b);
    initializers::updateIfExists(yamlParams, "rs_sr0", drParameters.rateAndState.sr0);
    initializers::updateIfExists(
        yamlParams, "rs_inisliprate1", drParameters.rateAndState.initialSlipRate1);
    initializers::updateIfExists(
        yamlParams, "rs_inisliprate2", drParameters.rateAndState.initialSlipRate2);
    initializers::updateIfExists(yamlParams, "mu_w", drParameters.rateAndState.muW);

    // Thermal Pressurisation parameters
    initializers::updateIfExists(yamlParams, "alpha_th", drParameters.rateAndState.tpAlpha);
    initializers::updateIfExists(yamlParams, "rho_c", drParameters.rateAndState.rhoC);
    initializers::updateIfExists(yamlParams, "tp_lambda", drParameters.rateAndState.tpLambda);
    initializers::updateIfExists(
        yamlParams, "initemp", drParameters.rateAndState.initialTemperature);
    initializers::updateIfExists(
        yamlParams, "inipressure", drParameters.rateAndState.initialPressure);

    // Prakash-Clifton regularization parameters
    initializers::updateIfExists(yamlParams, "vStar", drParameters.vStar);
    initializers::updateIfExists(yamlParams, "prakashLength", drParameters.prakashLength);

    // filename of the yaml file describing the fault parameters
    initializers::updateIfExists(yamlParams, "modelfilename", drParameters.faultFileName);
  }
  // if there is no filename given for the fault, assume that we do not use dynamic rupture
  if (drParameters.faultFileName == "") {
    drParameters.isDynamicRuptureEnabled = false;
  }

  return drParameters;
}
} // namespace seissol::dr
#endif // SEISSOL_PARAMETERS_H
