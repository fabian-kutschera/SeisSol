#include "VelocityWeakening.h"

namespace seissol::dr::friction_law {

void VelocityWeakening::copyLtsTreeToLocal(seissol::initializers::Layer& layerData,
                                           seissol::initializers::DynamicRupture* dynRup,
                                           real fullUpdateTime) {
  // first copy all Variables from the Base Lts dynRup tree
  BaseFrictionLaw::copyLtsTreeToLocal(layerData, dynRup, fullUpdateTime);
  // TODO: change later to const_cast
  // seissol::initializers::DR_lts_template *concreteLts =
  // dynamic_cast<seissol::initializers::DR_lts_template *>(dynRup);

  auto concreteLts = dynamic_cast<seissol::initializers::LTS_RateAndState*>(dynRup);
  stateVar = layerData.var(concreteLts->stateVariable);
  sl0 = layerData.var(concreteLts->rs_sl0);
  a = layerData.var(concreteLts->rs_a);

  /*a
   * Add new LTS parameter specific for this
   */
}

void VelocityWeakening::evaluate(
    seissol::initializers::Layer& layerData,
    seissol::initializers::DynamicRupture* dynRup,
    real (*QInterpolatedPlus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
    real (*QInterpolatedMinus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
    real fullUpdateTime,
    double timeWeights[CONVERGENCE_ORDER]) {
  VelocityWeakening::copyLtsTreeToLocal(layerData, dynRup, fullUpdateTime);
  constexpr unsigned int nSRupdates{5}, nSVupdates{2};
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (unsigned ltsFace = 0; ltsFace < layerData.getNumberOfCells(); ++ltsFace) {
    // initialize struct for in/outputs stresses
    FaultStresses faultStresses = {};

    // compute stresses from Qinterpolated
    precomputeStressFromQInterpolated(
        faultStresses, QInterpolatedPlus[ltsFace], QInterpolatedMinus[ltsFace], ltsFace);

    for (int pointIndex = 0; pointIndex < numberOfPoints; pointIndex++) {
      // Find variables at given fault node
      real localSlip = slip[ltsFace][pointIndex];                //  Slip path
      real localSlip1 = slipStrike[ltsFace][pointIndex];         // Slip along direction 1
      real localSlip2 = slipDip[ltsFace][pointIndex];            //  Slip  along direction 2
      real localSlipRate1 = slipRateStrike[ltsFace][pointIndex]; // Slip Rate along direction 2
      real localSlipRate2 = slipRateDip[ltsFace][pointIndex];    // Slip Rate along direction 2
      real localStateVariable = stateVar[ltsFace][pointIndex];   // stateVariable
      real initialPressure = initialStressInFaultCS[ltsFace][pointIndex][0]; // initial pressure

      real localMu = 0;
      real localTractionXY = 0;
      real localTractionXZ = 0;

      for (int timeIndex = 0; timeIndex < CONVERGENCE_ORDER; timeIndex++) {
        real localPressure = faultStresses.NormalStressGP[timeIndex][pointIndex];
        real timeIncrement = deltaT[timeIndex];

        // load traction and normal stress
        real pressure = localPressure + initialPressure;
        real stressXY = initialStressInFaultCS[ltsFace][pointIndex][3] +
                        faultStresses.XYStressGP[timeIndex][pointIndex];
        real stressXZ = initialStressInFaultCS[ltsFace][pointIndex][5] +
                        faultStresses.XZStressGP[timeIndex][pointIndex];
        real totalShearStressYZ = std::sqrt(std::pow(stressXY, 2) + std::pow(stressXZ, 2));

        // We use the regularized rate-and-state friction, after Rice & Ben-Zion (1996)
        // ( Numerical note: ASINH(X)=LOG(X+SQRT(X^2+1)) )

        real stateVariable = localStateVariable; // Careful, the SV must always be corrected using
                                                 // stateVariable and not localStateVariable!

        // The following process is adapted from that described by Kaneko et al. (2008)
        slipRateMagnitude[ltsFace][pointIndex] =
            std::sqrt(std::pow(localSlipRate1, 2) + std::pow(localSlipRate2, 2));

        real characteristicTime = sl0[ltsFace][pointIndex] / drParameters.rs_sr0;
        real coeft = exp(-timeIncrement / characteristicTime);

        real tmp = 0;
        for (unsigned int j = 0; j < nSVupdates; j++) { //! This loop corrects SV values
          slipRateMagnitude[ltsFace][pointIndex] = std::abs(slipRateMagnitude[ltsFace][pointIndex]);
          // exact integration assuming constant V in this loop
          localStateVariable =
              characteristicTime * slipRateMagnitude[ltsFace][pointIndex] * (1.0 - coeft) +
              coeft * stateVariable;
          //! Newton-Raphson algorithm to determine the value of the slip rate.
          //! We wish to find SR that fulfills g(SR)=f(SR), by building up the function NR=f-g ,
          //! which has
          //!  a derivative dNR = d(NR)/d(SR). We can then find SR by iterating SR_{i+1}=SR_i-( NR_i
          //!  / dNR_i ).
          //! In our case we equalize the values of the traction for two equations:
          //!             g = SR*mu/2/cs + T^G             (eq. 18 of de la Puente et al. (2009))
          //!             f = (mu*initialPressure-|S_0|)*S_0/|S_0|     (Coulomb's model of friction)
          //!             where mu = mu_s + a V/(V+Vc) - b SV/(SV + Vc)
          //!
          real slipRateGuess =
              slipRateMagnitude[ltsFace][pointIndex]; // We use as first guess the SR value of the
                                                      // previous time ste

          for (unsigned int i = 0; i < nSRupdates; i++) {
            real tmp =
                drParameters.rs_f0 +
                a[ltsFace][pointIndex] * slipRateGuess / (slipRateGuess + drParameters.rs_sr0) -
                drParameters.rs_b * localStateVariable /
                    (localStateVariable + sl0[ltsFace][pointIndex]);
            real NR =
                -impAndEta[ltsFace].inv_eta_s * (std::abs(pressure) * tmp - totalShearStressYZ) -
                slipRateGuess;
            real dNR = -impAndEta[ltsFace].inv_eta_s *
                           (abs(pressure) *
                            (a[ltsFace][pointIndex] / (slipRateGuess + drParameters.rs_sr0) -
                             a[ltsFace][pointIndex] * slipRateGuess /
                                 std::pow(slipRateGuess + drParameters.rs_sr0, 2))) -
                       1.0;
            slipRateGuess = slipRateGuess - NR / dNR;
          } // End numberSlipRateUpdates-Loop
          // For the next SV update, use the mean slip rate between the
          // initial guess and the one found (Kaneko 2008, step 6)
          tmp = 0.5 * (slipRateMagnitude[ltsFace][pointIndex] + std::abs(slipRateGuess));
          slipRateMagnitude[ltsFace][pointIndex] = abs(slipRateGuess);
        } // End nSVupdates-Loop -  This loop corrects SV values

        localStateVariable = characteristicTime * tmp * (1 - coeft) + coeft * stateVariable;
        tmp = 0.5 * (slipRateMagnitude[ltsFace][pointIndex]) / drParameters.rs_sr0 *
              exp((drParameters.rs_f0 +
                   drParameters.rs_b *
                       log(drParameters.rs_sr0 * localStateVariable / sl0[ltsFace][pointIndex])) /
                  a[ltsFace][pointIndex]);

        //! Ampuero and Ben-Zion 2008 (eq. 1):
        // localMu = friction coefficient (mu_f)
        // rs_f0 = static coefficient (mu_s)
        // rs_a = positive coefficient, quantifying  the direct effect (alpha)
        // localSlipRate = slip rate (V)
        // rs_sr0 = characteristic velocity scale (V_c)
        // rs_b = positive coefficient, quantifying  the evolution effect (beta)
        // sl0 = characteristic velocity scale (V_c)
        localMu = drParameters.rs_f0 +
                  a[ltsFace][pointIndex] * slipRateMagnitude[ltsFace][pointIndex] /
                      (slipRateMagnitude[ltsFace][pointIndex] + drParameters.rs_sr0) -
                  drParameters.rs_b * localStateVariable /
                      (localStateVariable + sl0[ltsFace][pointIndex]);

        // update stress change
        real localTractionXY = -((initialStressInFaultCS[ltsFace][pointIndex][3] +
                                  faultStresses.XYStressGP[timeIndex][pointIndex]) /
                                 totalShearStressYZ) *
                               localMu * pressure;
        real localTractionXZ = -((initialStressInFaultCS[ltsFace][pointIndex][5] +
                                  faultStresses.XZStressGP[timeIndex][pointIndex]) /
                                 totalShearStressYZ) *
                               localMu * pressure;
        localTractionXY = localTractionXY - initialStressInFaultCS[ltsFace][pointIndex][3];
        localTractionXZ = localTractionXZ - initialStressInFaultCS[ltsFace][pointIndex][5];

        // Compute slip
        // ABS of localSlipRate removed as it would be the accumulated slip that is
        // usually not needed in the solver, see linear slip weakening
        localSlip = localSlip + slipRateMagnitude[ltsFace][pointIndex] * timeIncrement;

        // Update slip rate (notice that localSlipRate(T=0)=-2c_s/mu*s_xy^{Godunov} is the slip rate
        // caused by a free surface!)
        localSlipRate1 = -impAndEta[ltsFace].inv_eta_s *
                         (localTractionXY - faultStresses.XYStressGP[timeIndex][pointIndex]);
        localSlipRate2 = -impAndEta[ltsFace].inv_eta_s *
                         (localTractionXZ - faultStresses.XZStressGP[timeIndex][pointIndex]);

        localSlip1 = localSlip1 + (localSlipRate1)*timeIncrement;
        localSlip2 = localSlip2 + (localSlipRate2)*timeIncrement;

        // Save traction for flux computation
        faultStresses.XYTractionResultGP[timeIndex][pointIndex] = localTractionXY;
        faultStresses.XZTractionResultGP[timeIndex][pointIndex] = localTractionXZ;
      } // End of timeIndex loop

      mu[ltsFace][pointIndex] = localMu;
      slipRateStrike[ltsFace][pointIndex] = localSlipRate1;
      slipRateDip[ltsFace][pointIndex] = localSlipRate2;
      slip[ltsFace][pointIndex] = localSlip;
      slipStrike[ltsFace][pointIndex] = localSlip1;
      slipDip[ltsFace][pointIndex] = localSlip2;
      stateVar[ltsFace][pointIndex] = localStateVariable;
      tractionXY[ltsFace][pointIndex] = localTractionXY;
      tractionXZ[ltsFace][pointIndex] = localTractionXZ;
    } // End of pointIndex-loop

    // output rupture front
    // outside of timeIndex loop in order to safe an 'if' in a loop
    // this way, no subtimestep resolution possible
    saveRuptureFrontOutput(ltsFace);

    // output peak slip rate
    savePeakSlipRateOutput(ltsFace);

    // save stresses in imposedState
    postcomputeImposedStateFromNewStress(QInterpolatedPlus[ltsFace],
                                         QInterpolatedMinus[ltsFace],
                                         faultStresses,
                                         timeWeights,
                                         ltsFace);

  } // End of Loop over Faces

} // End of Function evaluate

} // namespace seissol::dr::friction_law