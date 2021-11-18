#include "BaseFrictionLaw.h"

namespace seissol::dr::friction_law {
void BaseFrictionLaw::copyLtsTreeToLocal(seissol::initializers::Layer& layerData,
                                         seissol::initializers::DynamicRupture* dynRup,
                                         real updateTime) {
  impAndEta = layerData.var(dynRup->impAndEta);
  initialStressInFaultCS = layerData.var(dynRup->initialStressInFaultCS);
  mu = layerData.var(dynRup->mu);
  slip = layerData.var(dynRup->slip);
  slipStrike = layerData.var(dynRup->slipStrike);
  slipDip = layerData.var(dynRup->slipDip);
  slipRateMagnitude = layerData.var(dynRup->slipRateMagnitude);
  slipRateStrike = layerData.var(dynRup->slipRateStrike);
  slipRateDip = layerData.var(dynRup->slipRateDip);
  ruptureTime = layerData.var(dynRup->ruptureTime);
  ruptureFront = layerData.var(dynRup->ruptureFront);
  peakSlipRate = layerData.var(dynRup->peakSlipRate);
  tractionXY = layerData.var(dynRup->tractionXY);
  tractionXZ = layerData.var(dynRup->tractionXZ);
  imposedStatePlus = layerData.var(dynRup->imposedStatePlus);
  imposedStateMinus = layerData.var(dynRup->imposedStateMinus);
  fullUpdateTime = updateTime;
}

void BaseFrictionLaw::precomputeStressFromQInterpolated(
    FaultStresses& faultStresses,
    real qInterpolatedPlus[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
    real qInterpolatedMinus[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
    unsigned int ltsFace) {
  // this initialization of the kernel could be moved to the initializer,
  // since all inputs outside the j-loop are time independent
  // set inputParam could be extendent for this
  // the kernel then could be a class attribute (but be careful of race conditions since this is
  // computed in parallel!!)
  dynamicRupture::kernel::StressFromQInterpolated stressFromQInterpolatedKrnl;
  stressFromQInterpolatedKrnl.eta_p = impAndEta[ltsFace].etaP;
  stressFromQInterpolatedKrnl.eta_s = impAndEta[ltsFace].etaS;
  stressFromQInterpolatedKrnl.inv_Zp = impAndEta[ltsFace].inversePWaveImpedance;
  stressFromQInterpolatedKrnl.inv_Zs = impAndEta[ltsFace].inverseSWaveImpedance;
  stressFromQInterpolatedKrnl.inv_Zp_neig = impAndEta[ltsFace].inversePWaveImpedanceNeighbor;
  stressFromQInterpolatedKrnl.inv_Zs_neig = impAndEta[ltsFace].inverseSWaveImpedanceNeighbor;
  stressFromQInterpolatedKrnl.select0 = init::select0::Values;
  stressFromQInterpolatedKrnl.select3 = init::select3::Values;
  stressFromQInterpolatedKrnl.select5 = init::select5::Values;
  stressFromQInterpolatedKrnl.select6 = init::select6::Values;
  stressFromQInterpolatedKrnl.select7 = init::select7::Values;
  stressFromQInterpolatedKrnl.select8 = init::select8::Values;

  for (int j = 0; j < CONVERGENCE_ORDER; j++) {
    stressFromQInterpolatedKrnl.QInterpolatedMinus = qInterpolatedMinus[j];
    stressFromQInterpolatedKrnl.QInterpolatedPlus = qInterpolatedPlus[j];
    stressFromQInterpolatedKrnl.NorStressGP = faultStresses.normalStressGP[j];
    stressFromQInterpolatedKrnl.XYStressGP = faultStresses.stressXYGP[j];
    stressFromQInterpolatedKrnl.XZStressGP = faultStresses.stressXZGP[j];
    // Carsten Uphoff Thesis: EQ.: 4.53
    stressFromQInterpolatedKrnl.execute();
  }

  static_assert(tensor::QInterpolated::Shape[0] == tensor::resample::Shape[0],
                "Different number of quadrature points?");
}

void BaseFrictionLaw::postcomputeImposedStateFromNewStress(
    real qInterpolatedPlus[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
    real qInterpolatedMinus[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
    const FaultStresses& faultStresses,
    double timeWeights[CONVERGENCE_ORDER],
    unsigned int ltsFace) {
  // this initialization of the kernel could be moved to the initializer
  // set inputParam could be extendent for this (or create own function)
  // the kernel then could be a class attribute and following values are only set once
  //(but be careful of race conditions since this is computed in parallel for each face!!)
  dynamicRupture::kernel::ImposedStateFromNewStress imposedStateFromNewStressKrnl;
  imposedStateFromNewStressKrnl.select0 = init::select0::Values;
  imposedStateFromNewStressKrnl.select3 = init::select3::Values;
  imposedStateFromNewStressKrnl.select5 = init::select5::Values;
  imposedStateFromNewStressKrnl.select6 = init::select6::Values;
  imposedStateFromNewStressKrnl.select7 = init::select7::Values;
  imposedStateFromNewStressKrnl.select8 = init::select8::Values;
  imposedStateFromNewStressKrnl.inv_Zs = impAndEta[ltsFace].inverseSWaveImpedance;
  imposedStateFromNewStressKrnl.inv_Zs_neig = impAndEta[ltsFace].inverseSWaveImpedanceNeighbor;
  imposedStateFromNewStressKrnl.inv_Zp = impAndEta[ltsFace].inversePWaveImpedance;
  imposedStateFromNewStressKrnl.inv_Zp_neig = impAndEta[ltsFace].inversePWaveImpedanceNeighbor;

  // set imposed state to zero
  for (unsigned int i = 0; i < tensor::QInterpolated::size(); i++) {
    imposedStatePlus[ltsFace][i] = 0;
    imposedStateMinus[ltsFace][i] = 0;
  }
  imposedStateFromNewStressKrnl.imposedStatePlus = imposedStatePlus[ltsFace];
  imposedStateFromNewStressKrnl.imposedStateMinus = imposedStateMinus[ltsFace];

  for (int j = 0; j < CONVERGENCE_ORDER; j++) {
    imposedStateFromNewStressKrnl.NorStressGP = faultStresses.normalStressGP[j];
    imposedStateFromNewStressKrnl.TractionGP_XY = faultStresses.tractionXYResultGP[j];
    imposedStateFromNewStressKrnl.TractionGP_XZ = faultStresses.tractionXZResultGP[j];
    imposedStateFromNewStressKrnl.timeWeights = timeWeights[j];
    imposedStateFromNewStressKrnl.QInterpolatedMinus = qInterpolatedMinus[j];
    imposedStateFromNewStressKrnl.QInterpolatedPlus = qInterpolatedPlus[j];
    // Carsten Uphoff Thesis: EQ.: 4.60
    imposedStateFromNewStressKrnl.execute();
  }
}

/*
 * https://strike.scec.org/cvws/download/SCEC_validation_slip_law.pdf
 */
real BaseFrictionLaw::calcSmoothStepIncrement(real currentTime, real dt) {
  real gNuc;
  real prevTime;
  if (currentTime > 0.0 && currentTime <= drParameters.t0) {
    gNuc = calcSmoothStep(currentTime);
    prevTime = currentTime - dt;
    if (prevTime > 0.0) {
      gNuc = gNuc - calcSmoothStep(prevTime);
    }
  } else {
    gNuc = 0.0;
  }
  return gNuc;
}

/*
 * https://strike.scec.org/cvws/download/SCEC_validation_slip_law.pdf
 */
real BaseFrictionLaw::calcSmoothStep(real currentTime) {
  real gNuc;
  if (currentTime <= 0) {
    gNuc = 0.0;
  } else {
    if (currentTime < drParameters.t0) {
      gNuc = std::exp(std::pow(currentTime - drParameters.t0, 2) /
                      (currentTime * (currentTime - 2.0 * drParameters.t0)));
    } else {
      gNuc = 1.0;
    }
  }
  return gNuc;
}

void BaseFrictionLaw::saveRuptureFrontOutput(unsigned int ltsFace) {
  for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
    constexpr real ruptureFrontThreshold = 0.001;
    if (ruptureFront[ltsFace][pointIndex] &&
        slipRateMagnitude[ltsFace][pointIndex] > ruptureFrontThreshold) {
      ruptureTime[ltsFace][pointIndex] = fullUpdateTime;
      ruptureFront[ltsFace][pointIndex] = false;
    }
  }
}

void BaseFrictionLaw::savePeakSlipRateOutput(unsigned int ltsFace) {
  for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
    if (slipRateMagnitude[ltsFace][pointIndex] > peakSlipRate[ltsFace][pointIndex]) {
      peakSlipRate[ltsFace][pointIndex] = slipRateMagnitude[ltsFace][pointIndex];
    }
  }
}

void BaseFrictionLaw::saveAverageSlipOutput(std::array<real, numPaddedPoints>& tmpSlip,
                                            unsigned int ltsFace) {
  real sumTmpSlip = 0;
  if (drParameters.isMagnitudeOutputOn) {
    for (int pointIndex = 0; pointIndex < numberOfPoints; pointIndex++)
      sumTmpSlip += tmpSlip[pointIndex];
    averagedSlip[ltsFace] = averagedSlip[ltsFace] + sumTmpSlip / numberOfPoints;
  }
}

void BaseFrictionLaw::computeDeltaT(double* timePoints) {
  deltaT[0] = timePoints[0];
  for (int timeIndex = 1; timeIndex < CONVERGENCE_ORDER; timeIndex++) {
    deltaT[timeIndex] = timePoints[timeIndex] - timePoints[timeIndex - 1];
  }
  // to fill last segment of Gaussian integration
  deltaT[CONVERGENCE_ORDER - 1] = deltaT[CONVERGENCE_ORDER - 1] + deltaT[0];
}
} // namespace seissol::dr::friction_law
