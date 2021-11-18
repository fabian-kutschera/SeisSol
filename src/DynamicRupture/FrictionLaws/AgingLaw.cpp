#include "AgingLaw.h"
namespace seissol::dr::friction_law {
real AgingLaw::calcStateVariableHook(real stateVariable, real tmp, real timeIncrement, real sl0) {
  return stateVariable * std::exp(-tmp * timeIncrement / sl0) +
         sl0 / tmp * (1.0 - std::exp(-tmp * timeIncrement / sl0));
}

void AgingLaw::evaluate(
    initializers::Layer& layerData,
    initializers::DynamicRupture* dynRup,
    real (*qInterpolatedPlus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
    real (*qInterpolatedMinus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
    real fullUpdateTime,
    double timeWeights[CONVERGENCE_ORDER]) {
  auto* concreteLts = dynamic_cast<initializers::LTS_RateAndState*>(dynRup);
  copyLtsTreeToLocal(layerData, dynRup, fullUpdateTime);

  model::IsotropicWaveSpeeds* waveSpeedsPlus = layerData.var(concreteLts->waveSpeedsPlus);
  model::IsotropicWaveSpeeds* waveSpeedsMinus = layerData.var(concreteLts->waveSpeedsMinus);
  real(*initialStressInFaultCS)[numPaddedPoints][6] =
      layerData.var(concreteLts->initialStressInFaultCS);

  real(*a)[numPaddedPoints] = layerData.var(concreteLts->rs_a);
  real(*sl0)[numPaddedPoints] = layerData.var(concreteLts->rs_sl0);

  real(*mu)[numPaddedPoints] = layerData.var(concreteLts->mu);
  real(*slip)[numPaddedPoints] = layerData.var(concreteLts->slip);
  real(*slipStrike)[numPaddedPoints] = layerData.var(concreteLts->slipStrike);
  real(*slipDip)[numPaddedPoints] = layerData.var(concreteLts->slipDip);
  real(*slipRateStrike)[numPaddedPoints] = layerData.var(concreteLts->slipRateStrike);
  real(*slipRateDip)[numPaddedPoints] = layerData.var(concreteLts->slipRateDip);
  real(*stateVariable)[numPaddedPoints] = layerData.var(concreteLts->stateVariable);

  real(*tracXY)[numPaddedPoints] = layerData.var(concreteLts->tractionXY);
  real(*tracXZ)[numPaddedPoints] = layerData.var(concreteLts->tractionXZ);

  // loop parameter are fixed, not variable??
  constexpr unsigned int nSRupdates{5}, nSVupdates{2};

#ifdef _OPENMP
#pragma omp parallel for schedule(static) // private(qInterpolatedPlus,qInterpolatedMinus)
#endif
  for (unsigned ltsFace = 0; ltsFace < layerData.getNumberOfCells(); ++ltsFace) {

    FaultStresses faultStresses = {};

    precomputeStressFromQInterpolated(
        faultStresses, qInterpolatedPlus[ltsFace], qInterpolatedMinus[ltsFace], ltsFace);

    for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {

      // Find variables at given fault node
      real localSlip = slip[ltsFace][pointIndex];                     // Slip path
      real localSlipStrike = slipStrike[ltsFace][pointIndex];         // Slip along direction 1
      real localSlipDip = slipDip[ltsFace][pointIndex];               // Slip along direction 2
      real localSlipRateStrike = slipRateStrike[ltsFace][pointIndex]; // Slip rate along direction 1
      real localSlipRateDip = slipRateDip[ltsFace][pointIndex];       // Slip rate along direction 2
      real localStateVariable = stateVariable[ltsFace][pointIndex];   // State Variable
      real initialPressure = initialStressInFaultCS[ltsFace][pointIndex][0]; // initial pressure

      real localMu = 0;
      real localTractionXY = 0;
      real localTractionXZ = 0;

      for (int timeIndex = 0; timeIndex < CONVERGENCE_ORDER; timeIndex++) {
        real localPressure = faultStresses.normalStressGP[timeIndex][pointIndex];
        real timeIncrement = deltaT[timeIndex];

        // load traction and normal stress
        real pressure = localPressure + initialPressure;

        real stressXY = initialStressInFaultCS[ltsFace][pointIndex][3] +
                        faultStresses.stressXYGP[timeIndex][pointIndex];
        real stressXZ = initialStressInFaultCS[ltsFace][pointIndex][5] +
                        faultStresses.stressXZGP[timeIndex][pointIndex];
        real totalShearStressYZ = std::sqrt(std::pow(stressXY, 2) + std::pow(stressXZ, 2));

        // We use the regularized rate-and-state friction, after Rice & Ben-Zion (1996)
        // ( Numerical note: ASINH(X)=LOG(X+SQRT(X^2+1)) )

        // Careful, the tmpStateVariable must always be corrected using tmpStateVariable and not
        // localStateVariable!
        real tmpStateVariable = localStateVariable;

        // The following process is adapted from that described by Kaneko et al. (2008)
        slipRateMagnitude[ltsFace][pointIndex] =
            std::sqrt(std::pow(localSlipRateStrike, 2) + std::pow(localSlipRateDip, 2));
        real tmp = std::fabs(slipRateMagnitude[ltsFace][pointIndex]);

        // This loop corrects stateVariable values
        for (unsigned int j = 0; j < nSVupdates; j++) {
          slipRateMagnitude[ltsFace][pointIndex] =
              std::fabs(slipRateMagnitude[ltsFace][pointIndex]);

          // FL= 3 aging law and FL=4 slip law
          localStateVariable =
              calcStateVariableHook(tmpStateVariable, tmp, timeIncrement, sl0[ltsFace][pointIndex]);

          // Newton-Raphson algorithm to determine the value of the slip rate.
          // We wish to find SR that fulfills g(SR)=f(SR), by building up the function NR=f-g ,
          // which has
          //  a derivative dNR = d(NR)/d(SR). We can then find SR by iterating SR_{i+1}=SR_i-( NR_i
          //  / dNR_i ).
          // In our case we equalize the values of the traction for two equations:
          //             g = SR*mu/2/cs + T^G             (eq. 18 of de la Puente et al. (2009))
          //             f = (mu*P_0-|S_0|)*S_0/|S_0|     (Coulomb's model of friction)
          //               where mu=a*asinh(SR/2/SR0*exp((F0+b*log(SR0*SV/L))/a (eq. 2a of Lapusta
          //               and Rice (2003))

          // SRtest: We use as first guess the  SR value of the previous time step
          real slipRateGuess = slipRateMagnitude[ltsFace][pointIndex];

          for (unsigned int i = 0; i < nSRupdates; i++) { // This loop corrects SR values
            tmp = 0.5 / drParameters.rateAndState.sr0 *
                  std::exp((drParameters.rateAndState.f0 +
                            drParameters.rateAndState.b *
                                std::log(drParameters.rateAndState.sr0 * localStateVariable /
                                         sl0[ltsFace][pointIndex])) /
                           a[ltsFace][pointIndex]);
            real tmp2 = tmp * slipRateGuess;
            // TODO: author before me: not sure if ShTest=TotalShearStressYZ should be + or -...
            real newtonRaphson =
                -(1.0 / waveSpeedsPlus->sWaveVelocity / waveSpeedsPlus->density +
                  1.0 / waveSpeedsMinus->sWaveVelocity / waveSpeedsMinus->density) *
                    (std::fabs(pressure) * a[ltsFace][pointIndex] *
                         std::log(tmp2 + std::sqrt(std::pow(tmp2, 2) + 1.0)) -
                     totalShearStressYZ) -
                slipRateGuess;

            real newtonRaphsonDerivative =
                -(1.0 / waveSpeedsPlus->sWaveVelocity / waveSpeedsPlus->density +
                  1.0 / waveSpeedsMinus->sWaveVelocity / waveSpeedsMinus->density) *
                    (std::fabs(pressure) * a[ltsFace][pointIndex] /
                     std::sqrt(1 + std::pow(tmp2, 2)) * tmp) -
                1.0;
            // no ABS needed around newtonRaphson/newtonRaphsonDerivative at least for aging law
            real slipRateGuess = std::fabs(slipRateGuess - newtonRaphson / newtonRaphsonDerivative);
          }

          // For the next SV update, use the mean slip rate between the initial guess and the one
          // found (Kaneko 2008, step 6)
          tmp = 0.5 * (slipRateMagnitude[ltsFace][pointIndex] + std::fabs(slipRateGuess));

          slipRateMagnitude[ltsFace][pointIndex] = std::fabs(slipRateGuess);
        } // End SV-Loop

        // FL= 3 aging law and FL=4 slip law
        localStateVariable =
            calcStateVariableHook(tmpStateVariable, tmp, timeIncrement, sl0[ltsFace][pointIndex]);

        // TODO: reused calc from above -> simplify
        tmp = 0.5 * (slipRateMagnitude[ltsFace][pointIndex]) / drParameters.rateAndState.sr0 *
              std::exp((drParameters.rateAndState.f0 +
                        drParameters.rateAndState.b *
                            std::log(drParameters.rateAndState.sr0 * localStateVariable /
                                     sl0[ltsFace][pointIndex])) /
                       a[ltsFace][pointIndex]);

        localMu = a[ltsFace][pointIndex] * std::log(tmp + std::sqrt(std::pow(tmp, 2) + 1.0));

        // 2D:
        // LocTrac  = -(ABS(S_0)-LocMu*(LocP+P_0))*(S_0/ABS(S_0))
        // LocTrac  = ABS(LocTrac)*(-SignSR)  !!! line commented as it leads NOT to correct results
        // update stress change
        localTractionXY = -((initialStressInFaultCS[ltsFace][pointIndex][3] +
                             faultStresses.stressXYGP[pointIndex][timeIndex]) /
                            totalShearStressYZ) *
                          (localMu * pressure);
        localTractionXZ = -((initialStressInFaultCS[ltsFace][pointIndex][5] +
                             faultStresses.stressXZGP[pointIndex][timeIndex]) /
                            totalShearStressYZ) *
                          (localMu * pressure);
        localTractionXY = localTractionXY - initialStressInFaultCS[ltsFace][pointIndex][3];
        localTractionXZ = localTractionXZ - initialStressInFaultCS[ltsFace][pointIndex][5];

        // Compute slip
        // ABS of LocSR removed as it would be the accumulated slip that is usually not needed in
        // the solver, see linear slip weakening
        localSlip = localSlip + (slipRateMagnitude[ltsFace][pointIndex]) * timeIncrement;

        // Update slip rate (notice that LocSR(T=0)=-2c_s/mu*s_xy^{Godunov} is the slip rate caused
        // by a free surface!)
        localSlipRateStrike = -(1.0 / (waveSpeedsPlus->sWaveVelocity * waveSpeedsPlus->density) +
                                1.0 / (waveSpeedsMinus->sWaveVelocity * waveSpeedsMinus->density)) *
                              (localTractionXY - faultStresses.stressXYGP[timeIndex][pointIndex]);
        localSlipRateDip = -(1.0 / (waveSpeedsPlus->sWaveVelocity * waveSpeedsPlus->density) +
                             1.0 / (waveSpeedsMinus->sWaveVelocity * waveSpeedsMinus->density)) *
                           (localTractionXZ - faultStresses.stressXZGP[timeIndex][pointIndex]);

        localSlipStrike = localSlipStrike + localSlipRateStrike * timeIncrement;
        localSlipDip = localSlipDip + localSlipRateDip * timeIncrement;

        // Save traction for flux computation
        faultStresses.tractionXYResultGP[timeIndex][pointIndex] = localTractionXY;
        faultStresses.tractionXZResultGP[timeIndex][pointIndex] = localTractionXZ;
      } // End of timeIndex- loop

      mu[ltsFace][pointIndex] = localMu;
      slipRateStrike[ltsFace][pointIndex] = localSlipRateStrike;
      slipRateDip[ltsFace][pointIndex] = localSlipRateDip;
      slip[ltsFace][pointIndex] = localSlip;
      slipStrike[ltsFace][pointIndex] = localSlipStrike;
      slipDip[ltsFace][pointIndex] = localSlipDip;
      stateVariable[ltsFace][pointIndex] = localStateVariable;
      tracXY[ltsFace][pointIndex] = localTractionXY;
      tracXZ[ltsFace][pointIndex] = localTractionXZ;

    } // End of pointIndex-loop

    // output rupture front
    // outside of timeIndex loop in order to safe an 'if' in a loop
    // this way, no subtimestep resolution possible
    saveRuptureFrontOutput(ltsFace);

    savePeakSlipRateOutput(ltsFace);

    postcomputeImposedStateFromNewStress(qInterpolatedPlus[ltsFace],
                                         qInterpolatedMinus[ltsFace],
                                         faultStresses,
                                         timeWeights,
                                         ltsFace);
  } // end face-loop
} // end evaluate function

} // namespace seissol::dr::friction_law