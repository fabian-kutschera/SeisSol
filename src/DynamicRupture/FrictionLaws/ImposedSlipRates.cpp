#include "ImposedSlipRates.h"

namespace seissol::dr::friction_law {
void ImposedSlipRates::copyLtsTreeToLocal(seissol::initializers::Layer& layerData,
                                          seissol::initializers::DynamicRupture* dynRup,
                                          real fullUpdateTime) {
  // first copy all Variables from the Base Lts dynRup tree
  BaseFrictionLaw::copyLtsTreeToLocal(layerData, dynRup, fullUpdateTime);

  auto* concreteLts = dynamic_cast<seissol::initializers::LTS_ImposedSlipRates*>(dynRup);
  nucleationStressInFaultCS = layerData.var(concreteLts->nucleationStressInFaultCS);
  averagedSlip = layerData.var(concreteLts->averagedSlip);
}

void ImposedSlipRates::evaluate(
    seissol::initializers::Layer& layerData,
    seissol::initializers::DynamicRupture* dynRup,
    real (*qInterpolatedPlus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
    real (*qInterpolatedMinus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
    real fullUpdateTime,
    double timeWeights[CONVERGENCE_ORDER]) {

  copyLtsTreeToLocal(layerData, dynRup, fullUpdateTime);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (unsigned ltsFace = 0; ltsFace < layerData.getNumberOfCells(); ++ltsFace) {
    // initialize struct for in/outputs stresses
    FaultStresses faultStresses = {};

    // declare local variables
    std::array<real, numPaddedPoints> tmpSlip{0};
    real tn = fullUpdateTime;
    real gNuc = 0.0;

    // compute stresses from Qinterpolated
    precomputeStressFromQInterpolated(
        faultStresses, qInterpolatedPlus[ltsFace], qInterpolatedMinus[ltsFace], ltsFace);

    for (int timeIndex = 0; timeIndex < CONVERGENCE_ORDER; timeIndex++) { // loop over time steps
      real timeIncrement = deltaT[timeIndex];
      tn = tn + timeIncrement;
      gNuc = calcSmoothStepIncrement(tn, timeIncrement) / timeIncrement;

      for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
        //! EQN%NucleationStressInFaultCS (1 and 2) contains the slip in FaultCS
        faultStresses.tractionXYResultGP[timeIndex][pointIndex] =
            faultStresses.stressXYGP[timeIndex][pointIndex] -
            impAndEta[ltsFace].etaS * nucleationStressInFaultCS[ltsFace][pointIndex][0] * gNuc;
        faultStresses.tractionXZResultGP[timeIndex][pointIndex] =
            faultStresses.stressXZGP[timeIndex][pointIndex] -
            impAndEta[ltsFace].etaS * nucleationStressInFaultCS[ltsFace][pointIndex][1] * gNuc;
        slipRateStrike[ltsFace][pointIndex] =
            nucleationStressInFaultCS[ltsFace][pointIndex][0] * gNuc;
        slipRateDip[ltsFace][pointIndex] = nucleationStressInFaultCS[ltsFace][pointIndex][1] * gNuc;
        slipRateMagnitude[ltsFace][pointIndex] =
            std::sqrt(std::pow(slipRateStrike[ltsFace][pointIndex], 2) +
                      std::pow(slipRateDip[ltsFace][pointIndex], 2));

        //! Update slip
        slipStrike[ltsFace][pointIndex] += slipRateStrike[ltsFace][pointIndex] * timeIncrement;
        slipDip[ltsFace][pointIndex] += slipRateDip[ltsFace][pointIndex] * timeIncrement;
        slip[ltsFace][pointIndex] += slipRateMagnitude[ltsFace][pointIndex] * timeIncrement;
        tmpSlip[pointIndex] += slipRateMagnitude[ltsFace][pointIndex] * timeIncrement;

        tractionXY[ltsFace][pointIndex] = faultStresses.tractionXYResultGP[timeIndex][pointIndex];
        tractionXZ[ltsFace][pointIndex] = faultStresses.tractionXYResultGP[timeIndex][pointIndex];
      }
    }
    // output rupture front
    // outside of timeIndex loop in order to safe an 'if' in a loop
    // this way, no subtimestep resolution possible
    saveRuptureFrontOutput(ltsFace);

    // output peak slip rate
    savePeakSlipRateOutput(ltsFace);

    //---compute and store slip to determine the magnitude of an earthquake ---
    //    to this end, here the slip is computed and averaged per element
    //    in calc_seissol.f90 this value will be multiplied by the element surface
    //    and an output happened once at the end of the simulation
    saveAverageSlipOutput(tmpSlip, ltsFace);

    // save stresses in imposedState
    postcomputeImposedStateFromNewStress(qInterpolatedPlus[ltsFace],
                                         qInterpolatedMinus[ltsFace],
                                         faultStresses,
                                         timeWeights,
                                         ltsFace);
  } // End of Loop over Faces
}
} // namespace seissol::dr::friction_law
