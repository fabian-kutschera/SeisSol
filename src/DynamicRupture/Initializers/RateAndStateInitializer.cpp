#include "RateAndStateInitializer.h"

namespace seissol::dr::initializers {
void RateAndStateInitializer::initializeFault(seissol::initializers::DynamicRupture* dynRup,
                                              seissol::initializers::LTSTree* dynRupTree,
                                              seissol::Interoperability* interoperability) {
  BaseDRInitializer::initializeFault(dynRup, dynRupTree, interoperability);
  auto* concreteLts = dynamic_cast<seissol::initializers::LTS_RateAndState*>(dynRup);

  for (seissol::initializers::LTSTree::leaf_iterator it =
           dynRupTree->beginLeaf(seissol::initializers::LayerMask(Ghost));
       it != dynRupTree->endLeaf();
       ++it) {

    bool(*ds)[numPaddedPoints] = it->var(concreteLts->ds);
    real* averagedSlip = it->var(concreteLts->averagedSlip);
    real(*slipRateStrike)[numPaddedPoints] = it->var(concreteLts->slipRateStrike);
    real(*slipRateDip)[numPaddedPoints] = it->var(concreteLts->slipRateDip);
    real(*mu)[numPaddedPoints] = it->var(concreteLts->mu);

    real(*stateVariable)[numPaddedPoints] = it->var(concreteLts->stateVariable);
    real(*sl0)[numPaddedPoints] = it->var(concreteLts->rs_sl0);
    real(*a)[numPaddedPoints] = it->var(concreteLts->rs_a);
    real(*initialStressInFaultCS)[numPaddedPoints][6] =
        it->var(concreteLts->initialStressInFaultCS);

    real initialSlipRate = std::sqrt(std::pow(drParameters.rateAndState.initialSlipRate1, 2) +
                                     std::pow(drParameters.rateAndState.initialSlipRate2, 2));

    for (unsigned ltsFace = 0; ltsFace < it->getNumberOfCells(); ++ltsFace) {
      for (unsigned pointIndex = 0; pointIndex < numPaddedPoints; ++pointIndex) {
        ds[ltsFace][pointIndex] = drParameters.isDsOutputOn;
        slipRateStrike[ltsFace][pointIndex] = drParameters.rateAndState.initialSlipRate1;
        slipRateDip[ltsFace][pointIndex] = drParameters.rateAndState.initialSlipRate2;
        // compute initial friction and state
        std::tie(stateVariable[ltsFace][pointIndex], mu[ltsFace][pointIndex]) =
            computeInitialStateAndFriction(initialStressInFaultCS[ltsFace][pointIndex][3],
                                           initialStressInFaultCS[ltsFace][pointIndex][5],
                                           initialStressInFaultCS[ltsFace][pointIndex][0],
                                           a[ltsFace][pointIndex],
                                           drParameters.rateAndState.b,
                                           sl0[ltsFace][pointIndex],
                                           drParameters.rateAndState.sr0,
                                           drParameters.rateAndState.f0,
                                           initialSlipRate);
      }
      averagedSlip[ltsFace] = 0.0;
    }
    // can be removed once output is in c++
    for (unsigned int ltsFace = 0; ltsFace < it->getNumberOfCells(); ++ltsFace) {
      const auto& drFaceInformation = it->var(dynRup->faceInformation);
      unsigned meshFace = static_cast<int>(drFaceInformation[ltsFace].meshFace);
      interoperability->copyFrictionOutputToFortranSpecific(
          ltsFace, meshFace, averagedSlip, slipRateStrike, slipRateDip, mu);
      interoperability->copyFrictionOutputToFortranStateVar(ltsFace, meshFace, stateVariable);
    }
  }
}

std::pair<real, real>
    RateAndStateInitializer::computeInitialStateAndFriction(real tractionXY,
                                                            real tractionXZ,
                                                            real pressure,
                                                            real a,
                                                            real b,
                                                            real sl0,
                                                            real sr0,
                                                            real f0,
                                                            real initialSlipRate) {
  real absoluteTraction = std::sqrt(std::pow(tractionXY, 2) + std::pow(tractionXZ, 2));
  real tmp = std::abs(absoluteTraction / (a * pressure));
  real stateVariable = sl0 / sr0 *
                       std::exp((a * std::log(std::exp(tmp) - std::exp(-tmp)) - f0 -
                                 a * std::log(initialSlipRate / sr0)) /
                                b);
  real tmp2 =
      initialSlipRate * 0.5 / sr0 * std::exp((f0 + b * std::log(sr0 * stateVariable / sl0)) / a);
  real mu = a * std::asinh(tmp2);
  return {stateVariable, mu};
}

void RateAndStateInitializer::addAdditionalParameters(
    std::unordered_map<std::string, real*>& parameterToStorageMap,
    seissol::initializers::DynamicRupture* dynRup,
    seissol::initializers::LTSInternalNode::leaf_iterator& it) {
  auto* concreteLts = dynamic_cast<seissol::initializers::LTS_RateAndState*>(dynRup);
  real(*sl0)[numPaddedPoints] = it->var(concreteLts->rs_sl0);
  real(*a)[numPaddedPoints] = it->var(concreteLts->rs_a);
  parameterToStorageMap.insert({"sl0", (real*)sl0});
  parameterToStorageMap.insert({"rs_a", (real*)a});
}

std::pair<real, real>
    RateAndStateFastVelocityInitializer::computeInitialStateAndFriction(real tractionXY,
                                                                        real tractionXZ,
                                                                        real pressure,
                                                                        real a,
                                                                        real b,
                                                                        real sl0,
                                                                        real sr0,
                                                                        real f0,
                                                                        real initialSlipRate) {
  real absoluteTraction = std::sqrt(std::pow(tractionXY, 2) + std::pow(tractionXZ, 2));
  real tmp = std::abs(absoluteTraction / (a * pressure));
  real stateVariable =
      a * std::log(2.0 * sr0 / initialSlipRate * (std::exp(tmp) - std::exp(-tmp)) / 2.0);
  real tmp2 = initialSlipRate * 0.5 / sr0 * std::exp(stateVariable / a);
  // asinh(x)=log(x+sqrt(x^2+1))
  real mu = a * std::log(tmp2 + std::sqrt(std::pow(tmp2, 2) + 1.0));
  return {stateVariable, mu};
}

void RateAndStateFastVelocityInitializer::addAdditionalParameters(
    std::unordered_map<std::string, real*>& parameterToStorageMap,
    seissol::initializers::DynamicRupture* dynRup,
    seissol::initializers::LTSInternalNode::leaf_iterator& it) {
  RateAndStateInitializer::addAdditionalParameters(parameterToStorageMap, dynRup, it);
  auto* concreteLts =
      dynamic_cast<seissol::initializers::LTS_RateAndStateFastVelocityWeakening*>(dynRup);
  real(*srW)[numPaddedPoints] = it->var(concreteLts->rs_srW);
  parameterToStorageMap.insert({"rs_srW", (real*)srW});
}

void RateAndStateThermalPressurisationInitializer::initializeFault(
    seissol::initializers::DynamicRupture* dynRup,
    seissol::initializers::LTSTree* dynRupTree,
    seissol::Interoperability* interoperability) {
  RateAndStateInitializer::initializeFault(dynRup, dynRupTree, interoperability);

  auto* concreteLts =
      dynamic_cast<seissol::initializers::LTS_RateAndStateThermalPressurisation*>(dynRup);

  for (seissol::initializers::LTSTree::leaf_iterator it =
           dynRupTree->beginLeaf(seissol::initializers::LayerMask(Ghost));
       it != dynRupTree->endLeaf();
       ++it) {
    real(*temperature)[numPaddedPoints] = it->var(concreteLts->temperature);
    real(*pressure)[numPaddedPoints] = it->var(concreteLts->pressure);
    real(*tpTheta)[numPaddedPoints][numberOfTPGridPoints] = it->var(concreteLts->TP_theta);
    real(*tpSigma)[numPaddedPoints][numberOfTPGridPoints] = it->var(concreteLts->TP_sigma);

    for (unsigned ltsFace = 0; ltsFace < it->getNumberOfCells(); ++ltsFace) {
      for (unsigned pointIndex = 0; pointIndex < numPaddedPoints; ++pointIndex) {
        temperature[ltsFace][pointIndex] = drParameters.rateAndState.initialTemperature;
        pressure[ltsFace][pointIndex] = drParameters.rateAndState.initialPressure;
        for (unsigned tpGridIndex = 0; tpGridIndex < numberOfTPGridPoints; ++tpGridIndex) {
          tpTheta[ltsFace][pointIndex][tpGridIndex] = 0.0;
          tpSigma[ltsFace][pointIndex][tpGridIndex] = 0.0;
        }
      }
    }
  }
}

void RateAndStateThermalPressurisationInitializer::addAdditionalParameters(
    std::unordered_map<std::string, real*>& parameterToStorageMap,
    seissol::initializers::DynamicRupture* dynRup,
    seissol::initializers::LTSInternalNode::leaf_iterator& it) {
  RateAndStateFastVelocityInitializer::addAdditionalParameters(parameterToStorageMap, dynRup, it);

  auto* concreteLts =
      dynamic_cast<seissol::initializers::LTS_RateAndStateThermalPressurisation*>(dynRup);

  real(*tpHalfWidthShearZone)[numPaddedPoints] = it->var(concreteLts->TP_halfWidthShearZone);
  real(*alphaHy)[numPaddedPoints] = it->var(concreteLts->TP_alphaHy);
  parameterToStorageMap.insert({"tpHalfWidthShearZone", (real*)tpHalfWidthShearZone});
  parameterToStorageMap.insert({"alphaHy", (real*)alphaHy});
}
} // namespace seissol::dr::initializers