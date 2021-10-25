#include "LinearSlipWeakening.h"
namespace seissol::dr::friction_law {

  void LinearSlipWeakeningLawFL2::evaluate(seissol::initializers::Layer& layerData,
               seissol::initializers::DynamicRupture* dynRup,
               real (*QInterpolatedPlus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
               real (*QInterpolatedMinus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
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
      dynamicRupture::kernel::resampleParameter resampleKrnl;
      resampleKrnl.resampleM = init::resample::Values;

      std::array<real, numPaddedPoints> outputSlip{0};
      setTimeHook(ltsFace);

      precomputeStressFromQInterpolated(
          faultStresses, QInterpolatedPlus[ltsFace], QInterpolatedMinus[ltsFace], ltsFace);

      for (int timeIndex = 0; timeIndex < CONVERGENCE_ORDER; timeIndex++) { // loop over time steps
        for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
          // calculate total shear stress in Y and Z direction
          real stressXY = initialStressInFaultCS[ltsFace][pointIndex][3] +
                  faultStresses.XYStressGP[timeIndex][pointIndex];
          real stressXZ = initialStressInFaultCS[ltsFace][pointIndex][5] +
                  faultStresses.XZStressGP[timeIndex][pointIndex];
          real totalShearStressYZ = std::sqrt(std::pow(stressXY, 2) + std::pow(stressXZ, 2));
          // calculate normal stress
          real normalStress = initialStressInFaultCS[ltsFace][pointIndex][0] + faultStresses.NormalStressGP[timeIndex][pointIndex];
          //compute friction and slip rate
          std::tie(mu[ltsFace][pointIndex], slipRateMagnitude[ltsFace][pointIndex]) = 
            invertFrictionAndSlipRate(totalShearStressYZ, normalStress, ltsFace, pointIndex);

          slipRateStrike[ltsFace][pointIndex] = slipRateMagnitude[ltsFace][pointIndex] *
            (initialStressInFaultCS[ltsFace][pointIndex][3] +
             faultStresses.XYStressGP[timeIndex][pointIndex]) /
            totalShearStressYZ;
          slipRateDip[ltsFace][pointIndex] = slipRateMagnitude[ltsFace][pointIndex] *
            (initialStressInFaultCS[ltsFace][pointIndex][5] +
             faultStresses.XZStressGP[timeIndex][pointIndex]) /
            totalShearStressYZ;

          // calculate traction
          faultStresses.XYTractionResultGP[timeIndex][pointIndex] =
            faultStresses.XYStressGP[timeIndex][pointIndex] -
            impAndEta[ltsFace].eta_s * slipRateStrike[ltsFace][pointIndex];
          faultStresses.XZTractionResultGP[timeIndex][pointIndex] =
            faultStresses.XZStressGP[timeIndex][pointIndex] -
            impAndEta[ltsFace].eta_s * slipRateDip[ltsFace][pointIndex];
          tractionXY[ltsFace][pointIndex] = faultStresses.XYTractionResultGP[timeIndex][pointIndex];
          tractionXZ[ltsFace][pointIndex] = faultStresses.XYTractionResultGP[timeIndex][pointIndex];

          // update directional slip
          slipStrike[ltsFace][pointIndex] += slipRateStrike[ltsFace][pointIndex] * deltaT[timeIndex];
          slipDip[ltsFace][pointIndex] += slipRateDip[ltsFace][pointIndex] * deltaT[timeIndex];
        }

        // function g, output: stateVariablePsi & outputSlip
        real resampledSlipRate[numPaddedPoints];
        resampleKrnl.resamplePar = slipRateMagnitude[ltsFace];
        // output after execute
        resampleKrnl.resampledPar = resampledSlipRate; 

        // Resample slip-rate, such that the state (slip) lies in the same polynomial space as the degrees
        // of freedom resampleMatrix first projects LocSR on the two-dimensional basis on the reference
        // triangle with degree less or equal than CONVERGENCE_ORDER-1, and then evaluates the polynomial
        // at the quadrature points
        resampleKrnl.execute();

        for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
          //-------------------------------------
          // integrate Sliprate To Get Slip = State Variable
          slip[ltsFace][pointIndex] =
            slip[ltsFace][pointIndex] + resampledSlipRate[pointIndex] * deltaT[timeIndex];
          outputSlip[pointIndex] =
            outputSlip[pointIndex] + slipRateMagnitude[ltsFace][pointIndex] * deltaT[timeIndex];

          // Modification T. Ulrich: generalisation of tpv16/17 to 30/31
          // actually slip is already the stateVariable for this FL, but to simplify the next equations we
          // divide it here by d_C
          stateVariable[ltsFace][pointIndex] = std::min(
              std::fabs(slip[ltsFace][pointIndex]) / d_c[ltsFace][pointIndex], static_cast<real>(1.0));
        }

        // instantaneous healing option Reset Mu and Slip
        if (m_Params->IsInstaHealingOn == true) {
          instantaneousHealing(ltsFace);
        }
      } // End of timeIndex-Loop

      // output rupture front
      saveRuptureFrontOutput(ltsFace);

      // output time when shear stress is equal to the dynamic stress after rupture arrived
      // currently only for linear slip weakening
      saveDynamicStressOutput(ltsFace);

      // output peak slip rate
      savePeakSlipRateOutput(ltsFace);

      //---compute and store slip to determine the magnitude of an earthquake ---
      //    to this end, here the slip is computed and averaged per element
      //    in calc_seissol.f90 this value will be multiplied by the element surface
      //    and an output happened once at the end of the simulation
      saveAverageSlipOutput(outputSlip, ltsFace);

      postcomputeImposedStateFromNewStress(QInterpolatedPlus[ltsFace],
                                           QInterpolatedMinus[ltsFace],
                                           faultStresses,
                                           timeWeights,
                                           ltsFace);
    } // End of Loop over Faces
  }   // End of Function evaluate

  std::pair<real, real> LinearSlipWeakeningLawFL2::invertFrictionAndSlipRate(real totalShearStressYZ, real normalStress, unsigned int ltsFace, unsigned int pointIndex) {
    // function f, output: calculated mu
    real mu = mu_S[ltsFace][pointIndex] -
      (mu_S[ltsFace][pointIndex] - mu_D[ltsFace][pointIndex]) * stateVariable[ltsFace][pointIndex];
    // calculate fault strength
    // (Uphoff eq 2.44) with addition cohesion term
    real strength = cohesion[ltsFace][pointIndex] -
      mu * std::min(normalStress, static_cast<real>(0.0));

    // calculate slip rate
    real slipRate = std::max(static_cast<real>(0.0),
        (totalShearStressYZ - strength) * impAndEta[ltsFace].inv_eta_s);
    return {mu, slipRate};
  }


void LinearSlipWeakeningLawFL2::calcStrengthHook(std::array<real, numPaddedPoints>& Strength,
                                                 FaultStresses& faultStresses,
                                                 unsigned int timeIndex,
                                                 unsigned int ltsFace) {
}

void LinearSlipWeakeningLawFL2::calcStateVariableHook(
    std::array<real, numPaddedPoints>& stateVariablePsi,
    std::array<real, numPaddedPoints>& outputSlip,
    dynamicRupture::kernel::resampleParameter& resampleKrnl,
    unsigned int timeIndex,
    unsigned int ltsFace) {
}

void LinearSlipWeakeningLawFL16::copyLtsTreeToLocal(seissol::initializers::Layer& layerData,
                                                    seissol::initializers::DynamicRupture* dynRup,
                                                    real fullUpdateTime) {
  // first copy all Variables from the Base Lts dynRup tree
  LinearSlipWeakeningLawFL2::copyLtsTreeToLocal(layerData, dynRup, fullUpdateTime);
  // maybe change later to const_cast?
  seissol::initializers::LTS_LinearSlipWeakeningFL16* ConcreteLts =
      dynamic_cast<seissol::initializers::LTS_LinearSlipWeakeningFL16*>(dynRup);
  forced_rupture_time = layerData.var(ConcreteLts->forced_rupture_time);
  tn = layerData.var(ConcreteLts->tn);
}

void LinearSlipWeakeningLawFL16::setTimeHook(unsigned int ltsFace) {
  tn[ltsFace] = m_fullUpdateTime;
}

void LinearSlipWeakeningLawFL16::calcStateVariableHook(
    std::array<real, numPaddedPoints>& stateVariablePsi,
    std::array<real, numPaddedPoints>& outputSlip,
    dynamicRupture::kernel::resampleParameter& resampleKrnl,
    unsigned int timeIndex,
    unsigned int ltsFace) {
  LinearSlipWeakeningLawFL2::calcStateVariableHook(
      stateVariablePsi, outputSlip, resampleKrnl, timeIndex, ltsFace);
  tn[ltsFace] += deltaT[timeIndex];

  for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
    real f2 = 0.0;
    if (m_Params->t_0 == 0) {
      if (tn[ltsFace] >= forced_rupture_time[ltsFace][pointIndex]) {
        f2 = 1.0;
      } else {
        f2 = 0.0;
      }
    } else {
      f2 = std::max(
          static_cast<real>(0.0),
          std::min(static_cast<real>(1.0),
                   (m_fullUpdateTime - forced_rupture_time[ltsFace][pointIndex]) / m_Params->t_0));
    }
    stateVariablePsi[pointIndex] = std::max(stateVariablePsi[pointIndex], f2);
  }
}

void LinearSlipWeakeningLawBimaterialFL6::calcStrengthHook(
    std::array<real, numPaddedPoints>& Strength,
    FaultStresses& faultStresses,
    unsigned int timeIndex,
    unsigned int ltsFace) {
  std::array<real, numPaddedPoints> LocSlipRate;
  std::array<real, numPaddedPoints> sigma;

  for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
    //  modify strength according to prakash clifton
    // literature e.g.: Pelties - Verification of an ADER-DG method for complex dynamic rupture
    // problems
    LocSlipRate[pointIndex] =
        std::sqrt(slipRateStrike[ltsFace][pointIndex] * slipRateStrike[ltsFace][pointIndex] +
                  slipRateDip[ltsFace][pointIndex] * slipRateDip[ltsFace][pointIndex]);
    sigma[pointIndex] = faultStresses.NormalStressGP[timeIndex][pointIndex] +
                        initialStressInFaultCS[ltsFace][pointIndex][0];
    prak_clif_mod(strengthData[ltsFace][pointIndex],
                  sigma[pointIndex],
                  LocSlipRate[pointIndex],
                  mu[ltsFace][pointIndex],
                  deltaT[timeIndex]);

    // TODO: add this line to make the FL6 actually functional: (this line is also missing in the
    // master branch) Strength[pointIndex] = strengthData[ltsFace][pointIndex];
  }
}

void LinearSlipWeakeningLawBimaterialFL6::calcStateVariableHook(
    std::array<real, numPaddedPoints>& stateVariablePsi,
    std::array<real, numPaddedPoints>& outputSlip,
    dynamicRupture::kernel::resampleParameter& resampleKrnl,
    unsigned int timeIndex,
    unsigned int ltsFace) {
  for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
    slip[ltsFace][pointIndex] =
        slip[ltsFace][pointIndex] + slipRateMagnitude[ltsFace][pointIndex] * deltaT[timeIndex];
    outputSlip[pointIndex] = slip[ltsFace][pointIndex];

    //-------------------------------------
    // Modif T. Ulrich-> generalisation of tpv16/17 to 30/31
    // actually slip is already the stateVariable for this FL, but to simplify the next equations we
    // divide it here by d_C
    stateVariablePsi[pointIndex] = std::min(
        std::fabs(slip[ltsFace][pointIndex]) / d_c[ltsFace][pointIndex], static_cast<real>(1.0));
  }
}

void LinearSlipWeakeningLawBimaterialFL6::copyLtsTreeToLocal(
    seissol::initializers::Layer& layerData,
    seissol::initializers::DynamicRupture* dynRup,
    real fullUpdateTime) {
  // first copy all Variables from the Base Lts dynRup tree
  LinearSlipWeakeningLaw::copyLtsTreeToLocal(layerData, dynRup, fullUpdateTime);
  // maybe change later to const_cast?
  seissol::initializers::LTS_LinearBimaterialFL6* ConcreteLts =
      dynamic_cast<seissol::initializers::LTS_LinearBimaterialFL6*>(dynRup);
  strengthData = layerData.var(ConcreteLts->strengthData);
}

/*
 * calculates strength
 */
void LinearSlipWeakeningLawBimaterialFL6::prak_clif_mod(
    real& strength, real& sigma, real& LocSlipRate, real& mu, real& dt) {
  real expterm;
  expterm = std::exp(-(std::abs(LocSlipRate) + m_Params->v_star) * dt / m_Params->prakash_length);
  strength = strength * expterm - std::max(static_cast<real>(0.0), -mu * sigma) * (expterm - 1.0);
}
} // namespace seissol::dr::friction_law
