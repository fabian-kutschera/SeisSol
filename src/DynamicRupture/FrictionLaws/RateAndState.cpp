#include "RateAndState.h"

namespace seissol::dr::friction_law {
void RateAndStateFastVelocityWeakeningLaw::copyLtsTreeToLocalRS(
    seissol::initializers::Layer& layerData,
    seissol::initializers::DynamicRupture* dynRup,
    real fullUpdateTime) {
  // first copy all Variables from the Base Lts dynRup tree
  BaseFrictionLaw::copyLtsTreeToLocal(layerData, dynRup, fullUpdateTime);
  // maybe change later to const_cast?
  auto* concreteLts =
      dynamic_cast<seissol::initializers::LTS_RateAndStateFastVelocityWeakening*>(dynRup);
  nucleationStressInFaultCS = layerData.var(concreteLts->nucleationStressInFaultCS);

  sl0 = layerData.var(concreteLts->rs_sl0);
  a = layerData.var(concreteLts->rs_a);
  srW = layerData.var(concreteLts->rs_srW);
  ds = layerData.var(concreteLts->ds);
  averagedSlip = layerData.var(concreteLts->averagedSlip);
  stateVariable = layerData.var(concreteLts->stateVariable);
  dynStressTime = layerData.var(concreteLts->dynStressTime);
}

void RateAndStateFastVelocityWeakeningLaw::preCalcTime() {
  dt = 0;
  for (int timeIndex = 0; timeIndex < CONVERGENCE_ORDER; timeIndex++) {
    dt += deltaT[timeIndex];
  }
  if (fullUpdateTime <= drParameters.t0) {
    gNuc = calcSmoothStepIncrement(fullUpdateTime, dt);
  }
}

void RateAndStateFastVelocityWeakeningLaw::setInitialValues(
    std::array<real, numPaddedPoints>& localStateVariable, unsigned int ltsFace) {
  if (fullUpdateTime <= drParameters.t0) {
    for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
      for (int i = 0; i < 6; i++) {
        initialStressInFaultCS[ltsFace][pointIndex][i] +=
            nucleationStressInFaultCS[ltsFace][pointIndex][i] * gNuc;
      }
    }
  } // end If-Tnuc

  for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
    localStateVariable[pointIndex] = stateVariable[ltsFace][pointIndex];
  }
}

void RateAndStateFastVelocityWeakeningLaw::calcInitialSlipRate(
    std::array<real, numPaddedPoints>& totalShearStressYZ,
    FaultStresses& faultStresses,
    std::array<real, numPaddedPoints>& stateVarZero,
    std::array<real, numPaddedPoints>& localStateVariable,
    std::array<real, numPaddedPoints>& temporarySlipRate,
    unsigned int timeIndex,
    unsigned int ltsFace) {

  for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {

    // friction develops as                    mu = a * arcsinh[ V/(2*V0) * exp(SV/a) ]
    // state variable SV develops as          dSV / dt = -(V - L) * (SV - SV_ss)
    //                                        SV_ss = a * ln[ 2*V0/V * sinh(mu_ss/a) ]
    //                                        mu_ss = mu_w + [mu_lv - mu_w] / [ 1 + (V/Vw)^8 ] ^
    //                                        (1/8) ] mu_lv = mu_0 - (b-a) ln (V/V0)

    totalShearStressYZ[pointIndex] =
        std::sqrt(std::pow(initialStressInFaultCS[ltsFace][pointIndex][3] +
                               faultStresses.stressXYGP[timeIndex][pointIndex],
                           2) +
                  std::pow(initialStressInFaultCS[ltsFace][pointIndex][5] +
                               faultStresses.stressXZGP[timeIndex][pointIndex],
                           2));

    // We use the regularized rate-and-state friction, after Rice & Ben-Zion (1996)
    // ( Numerical note: ASINH(X)=LOG(X+SQRT(X^2+1)) )
    // Careful, the state variable must always be corrected using stateVar0 and not
    // localStateVariable!
    stateVarZero[pointIndex] = localStateVariable[pointIndex];

    // The following process is adapted from that described by Kaneko et al. (2008)
    slipRateMagnitude[ltsFace][pointIndex] =
        std::sqrt(std::pow(slipRateStrike[ltsFace][pointIndex], 2) +
                  std::pow(slipRateDip[ltsFace][pointIndex], 2));
    slipRateMagnitude[ltsFace][pointIndex] =
        std::max(almostZero, slipRateMagnitude[ltsFace][pointIndex]);
    temporarySlipRate[pointIndex] = slipRateMagnitude[ltsFace][pointIndex];
  } // End of pointIndex-loop
}

void RateAndStateFastVelocityWeakeningLaw::updateStateVariableIterative(
    bool& hasConverged,
    std::array<real, numPaddedPoints>& stateVarZero,
    std::array<real, numPaddedPoints>& srTmp,
    std::array<real, numPaddedPoints>& localStateVariable,
    std::array<real, numPaddedPoints>& pF,
    std::array<real, numPaddedPoints>& normalStress,
    std::array<real, numPaddedPoints>& totalShearStressYZ,
    std::array<real, numPaddedPoints>& srTest,
    FaultStresses& faultStresses,
    unsigned int timeIndex,
    unsigned int ltsFace) {
  for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
    // fault strength using LocMu and pF from previous timestep/iteration
    // 1.update SV using Vold from the previous time step
    updateStateVariable(pointIndex,
                        ltsFace,
                        stateVarZero[pointIndex],
                        deltaT[timeIndex],
                        srTmp[pointIndex],
                        localStateVariable[pointIndex]);
    normalStress[pointIndex] = faultStresses.normalStressGP[timeIndex][pointIndex] +
                               initialStressInFaultCS[ltsFace][pointIndex][0] - pF[pointIndex];
  } // End of pointIndex-loop

  // 2. solve for Vnew , applying the Newton-Raphson algorithm
  // effective normal stress including initial stresses and pore fluid pressure
  hasConverged = IterativelyInvertSR(
      ltsFace, numberSlipRateUpdates, localStateVariable, normalStress, totalShearStressYZ, srTest);

  for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
    // 3. update theta, now using V=(Vnew+Vold)/2
    // For the next SV update, use the mean slip rate between the initial guess and the one found
    // (Kaneko 2008, step 6)
    srTmp[pointIndex] = 0.5 * (slipRateMagnitude[ltsFace][pointIndex] + fabs(srTest[pointIndex]));

    // 4. solve again for Vnew
    slipRateMagnitude[ltsFace][pointIndex] = fabs(srTest[pointIndex]);

    // update LocMu
    updateMu(ltsFace, pointIndex, localStateVariable[pointIndex]);
  } // End of pointIndex-loop
}

void RateAndStateFastVelocityWeakeningLaw::executeIfNotConverged(
    std::array<real, numPaddedPoints>& localStateVariable, unsigned ltsFace) {
  real tmp = 0.5 / drParameters.rateAndState.sr0 * exp(localStateVariable[0] / a[ltsFace][0]) *
             slipRateMagnitude[ltsFace][0];
  if (std::isnan(tmp)) {
    logError() << "nonConvergence RS Newton , time = " << fullUpdateTime;

  } else {
    logWarning() << "nonConvergence RS Newton, time = " << fullUpdateTime;
  }

}

void RateAndStateFastVelocityWeakeningLaw::calcSlipRateAndTraction(
    std::array<real, numPaddedPoints>& stateVarZero,
    std::array<real, numPaddedPoints>& srTmp,
    std::array<real, numPaddedPoints>& localStateVariable,
    std::array<real, numPaddedPoints>& normalStress,
    std::array<real, numPaddedPoints>& totalShearStressYZ,
    std::array<real, numPaddedPoints>& tmpSlip,
    real deltaStateVar[numPaddedPoints],
    FaultStresses& faultStresses,
    unsigned int timeIndex,
    unsigned int ltsFace) {
  std::array<real, numPaddedPoints> localSlipRateMagnitude{0};

  for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
    //! SV from mean slip rate in tmp
    updateStateVariable(pointIndex,
                        ltsFace,
                        stateVarZero[pointIndex],
                        deltaT[timeIndex],
                        srTmp[pointIndex],
                        localStateVariable[pointIndex]);

    //! update LocMu for next strength determination, only needed for last update
    updateMu(ltsFace, pointIndex, localStateVariable[pointIndex]);

    //! update stress change
    tractionXY[ltsFace][pointIndex] = -((initialStressInFaultCS[ltsFace][pointIndex][3] +
                                         faultStresses.stressXYGP[timeIndex][pointIndex]) /
                                        totalShearStressYZ[pointIndex]) *
                                      mu[ltsFace][pointIndex] * normalStress[pointIndex];
    tractionXZ[ltsFace][pointIndex] = -((initialStressInFaultCS[ltsFace][pointIndex][5] +
                                         faultStresses.stressXZGP[timeIndex][pointIndex]) /
                                        totalShearStressYZ[pointIndex]) *
                                      mu[ltsFace][pointIndex] * normalStress[pointIndex];
    tractionXY[ltsFace][pointIndex] -= initialStressInFaultCS[ltsFace][pointIndex][3];
    tractionXZ[ltsFace][pointIndex] -= initialStressInFaultCS[ltsFace][pointIndex][5];

    // Compute slip
    //! ABS of locSlipRate removed as it would be the accumulated slip that is usually not needed in
    //! the solver, see linear slip weakening
    slip[ltsFace][pointIndex] += slipRateMagnitude[ltsFace][pointIndex] * deltaT[timeIndex];

    //! Update slip rate (notice that locSlipRate(T=0)=-2c_s/mu*s_xy^{Godunov} is the slip rate
    //! caused by a free surface!)
    slipRateStrike[ltsFace][pointIndex] =
        -impAndEta[ltsFace].invEtaS *
        (tractionXY[ltsFace][pointIndex] - faultStresses.stressXYGP[timeIndex][pointIndex]);
    slipRateDip[ltsFace][pointIndex] =
        -impAndEta[ltsFace].invEtaS *
        (tractionXZ[ltsFace][pointIndex] - faultStresses.stressXZGP[timeIndex][pointIndex]);

    //! TU 07.07.16: correct slipRateStrike and slipRateDip to avoid numerical errors
    localSlipRateMagnitude[pointIndex] = sqrt(std::pow(slipRateStrike[ltsFace][pointIndex], 2) +
                                              std::pow(slipRateDip[ltsFace][pointIndex], 2));
    if (localSlipRateMagnitude[pointIndex] != 0) {
      slipRateStrike[ltsFace][pointIndex] *=
          slipRateMagnitude[ltsFace][pointIndex] / localSlipRateMagnitude[pointIndex];
      slipRateDip[ltsFace][pointIndex] *=
          slipRateMagnitude[ltsFace][pointIndex] / localSlipRateMagnitude[pointIndex];
    }
    tmpSlip[pointIndex] += localSlipRateMagnitude[pointIndex] * deltaT[timeIndex];

    slipStrike[ltsFace][pointIndex] += slipRateStrike[ltsFace][pointIndex] * deltaT[timeIndex];
    slipDip[ltsFace][pointIndex] += slipRateDip[ltsFace][pointIndex] * deltaT[timeIndex];

    //! Save traction for flux computation
    faultStresses.tractionXYResultGP[timeIndex][pointIndex] = tractionXY[ltsFace][pointIndex];
    faultStresses.tractionXZResultGP[timeIndex][pointIndex] = tractionXZ[ltsFace][pointIndex];

    // Could be outside TimeLoop, since only last time result is used later
    deltaStateVar[pointIndex] = localStateVariable[pointIndex] - stateVariable[ltsFace][pointIndex];
  } // End of BndGP-loop
}

void RateAndStateFastVelocityWeakeningLaw::resampleStateVar(real deltaStateVar[numPaddedPoints],
                                                            unsigned int ltsFace) {
  dynamicRupture::kernel::resampleParameter resampleKrnl;
  resampleKrnl.resampleM = init::resample::Values;
  real resampledDeltaStateVariable[numPaddedPoints];
  resampleKrnl.resamplePar = deltaStateVar;
  resampleKrnl.resampledPar = resampledDeltaStateVariable; // output from execute
  resampleKrnl.execute();

  for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
    // write back State Variable to lts tree
    stateVariable[ltsFace][pointIndex] =
        stateVariable[ltsFace][pointIndex] + resampledDeltaStateVariable[pointIndex];
  }
}

void RateAndStateFastVelocityWeakeningLaw::saveDynamicStressOutput(unsigned int face) {
  for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {

    if (ruptureTime[face][pointIndex] > 0.0 && ruptureTime[face][pointIndex] <= fullUpdateTime &&
        ds[pointIndex] &&
        mu[face][pointIndex] <=
            (drParameters.rateAndState.muW +
             0.05 * (drParameters.rateAndState.f0 - drParameters.rateAndState.muW))) {
      dynStressTime[face][pointIndex] = fullUpdateTime;
      ds[face][pointIndex] = false;
    }
  }
}

void RateAndStateFastVelocityWeakeningLaw::hookSetInitialFluidPressure(
    std::array<real, numPaddedPoints>& fluidPressure, unsigned int ltsFace) {
  for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
    fluidPressure[pointIndex] = 0.0;
  }
}

void RateAndStateFastVelocityWeakeningLaw::hookCalcFluidPressure(
    std::array<real, numPaddedPoints>& fluidPressure,
    FaultStresses& faultStresses,
    bool saveTmpInTP,
    unsigned int timeIndex,
    unsigned int ltsFace) {}

void RateAndStateFastVelocityWeakeningLaw::updateStateVariable(int pointIndex,
                                                               unsigned int face,
                                                               real sv0,
                                                               real timeIncrement,
                                                               real& srTmp,
                                                               real& localStateVariable) {
  double fw = drParameters.rateAndState.muW;
  double localSrW = srW[face][pointIndex];
  double localA = a[face][pointIndex];
  double localSl0 = sl0[face][pointIndex];

  // low-velocity steady state friction coefficient
  double lowVelocityFriction =
      drParameters.rateAndState.f0 -
      (drParameters.rateAndState.b - localA) * log(srTmp / drParameters.rateAndState.sr0);
  // steady state friction coefficient
  double steadyStateFriction =
      fw + (lowVelocityFriction - fw) / pow(1.0 + std::pow(srTmp / localSrW, 8.0), 1.0 / 8.0);
  // steady-state state variable
  // For compiling reasons we write SINH(X)=(EXP(X)-EXP(-X))/2
  double steadyStateStateVariable =
      localA * log(2.0 * drParameters.rateAndState.sr0 / srTmp *
                   (exp(steadyStateFriction / localA) - exp(-steadyStateFriction / localA)) / 2.0);

  // exact integration of dSV/dt DGL, assuming constant V over integration step
  double exp1 = exp(-srTmp * (timeIncrement / localSl0));
  localStateVariable = steadyStateStateVariable * (1.0 - exp1) + exp1 * sv0;

  assert(!(std::isnan(localStateVariable) && pointIndex < numberOfPoints) && "NaN detected");
}

bool RateAndStateFastVelocityWeakeningLaw::IterativelyInvertSR(
    unsigned int ltsFace,
    int nSRupdates,
    std::array<real, numPaddedPoints>& localStateVariable,
    std::array<real, numPaddedPoints>& normalStress,
    std::array<real, numPaddedPoints>& shearStress,
    std::array<real, numPaddedPoints>& slipRateTest) {

  real tmp[numPaddedPoints], tmp2[numPaddedPoints], tmp3[numPaddedPoints], muF[numPaddedPoints],
      dMuF[numPaddedPoints], newtonRaphson[numPaddedPoints],
      newtonRaphsonDerivative[numPaddedPoints];
  // double AlmostZero = 1e-45;
  bool hasConverged = false;

  //! solve for Vnew = SR , applying the Newton-Raphson algorithm
  //! SR fulfills g(SR)=f(SR)
  //!-> find root of newtonRaphson=f-g using a Newton-Raphson algorithm with newtonRaphsonDerivative
  //!= d(newtonRaphson)/d(SR)
  //! SR_{i+1}=SR_i-( NR_i / dNR_i )
  //!
  //!        equalize:
  //!         g = SR*MU/2/cs + T^G             (eq. 18 of de la Puente et al. (2009))
  //!         f = (mu*P_0-|S_0|)*S_0/|S_0|     (Coulomb's model of friction)
  //!  where mu = friction coefficient, dependening on the RSF law used

  for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
    //! first guess = SR value of the previous step
    slipRateTest[pointIndex] = slipRateMagnitude[ltsFace][pointIndex];
    tmp[pointIndex] = 0.5 / drParameters.rateAndState.sr0 *
                      exp(localStateVariable[pointIndex] / a[ltsFace][pointIndex]);
  }

  for (int i = 0; i < nSRupdates; i++) {
    for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {

      //! f = ( tmp2 * ABS(LocP+P_0)- ABS(S_0))*(S_0)/ABS(S_0)
      //! g = slipRateTest * 1.0/(1.0/w_speed(2)/rho+1.0/w_speed_neig(2)/rho_neig) + ABS(ShTest)
      //! for compiling reasons ASINH(X)=LOG(X+SQRT(X^2+1))

      //! calculate friction coefficient
      tmp2[pointIndex] = tmp[pointIndex] * slipRateTest[pointIndex];
      muF[pointIndex] = a[ltsFace][pointIndex] *
                        log(tmp2[pointIndex] + sqrt(std::pow(tmp2[pointIndex], 2) + 1.0));
      dMuF[pointIndex] =
          a[ltsFace][pointIndex] / sqrt(1.0 + std::pow(tmp2[pointIndex], 2)) * tmp[pointIndex];
      newtonRaphson[pointIndex] =
          -impAndEta[ltsFace].invEtaS *
              (fabs(normalStress[pointIndex]) * muF[pointIndex] - shearStress[pointIndex]) -
          slipRateTest[pointIndex];
    }

    hasConverged = true;

    // max element of newtonRaphson must be smaller then aTolF
    for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
      if (fabs(newtonRaphson[pointIndex]) >= aTolF) {
        hasConverged = false;
        break;
      }
    }
    if (hasConverged) {
      return hasConverged;
    }
    for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {

      //! derivative of newtonRaphson
      newtonRaphsonDerivative[pointIndex] =
          -impAndEta[ltsFace].invEtaS * (fabs(normalStress[pointIndex]) * dMuF[pointIndex]) - 1.0;
      //! ratio
      tmp3[pointIndex] = newtonRaphson[pointIndex] / newtonRaphsonDerivative[pointIndex];

      //! update slipRateTest
      slipRateTest[pointIndex] = std::max(almostZero, slipRateTest[pointIndex] - tmp3[pointIndex]);
    }
  }
  return hasConverged;
}

bool RateAndStateFastVelocityWeakeningLaw::IterativelyInvertSR_Brent(
    unsigned int ltsFace,
    int nSRupdates,
    std::array<real, numPaddedPoints>& localStateVariable,
    std::array<real, numPaddedPoints>& normalStress,
    std::array<real, numPaddedPoints>& shearStress,
    std::array<real, numPaddedPoints>& srTest) {
  std::function<double(double, int)> function;
  double tol = 1e-30;

  real* localA = a[ltsFace];
  double sr0 = drParameters.rateAndState.sr0;
  double invEta = impAndEta[ltsFace].invEtaS;

  function = [invEta, &shearStress, normalStress, localA, localStateVariable, sr0](double slipRate,
                                                                                   int pointIndex) {
    double tmp =
        0.5 / sr0 * std::exp(localStateVariable[pointIndex] / localA[pointIndex]) * slipRate;
    double muF = localA[pointIndex] * std::log(tmp + std::sqrt(std::pow(tmp, 2) + 1.0));
    return -invEta * (fabs(normalStress[pointIndex]) * muF - shearStress[pointIndex]) - slipRate;
  };

  for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
    // TODO: use better boundaries?
    double a = slipRateMagnitude[ltsFace][pointIndex] -
               impAndEta[ltsFace].invEtaS * shearStress[pointIndex];
    double b = slipRateMagnitude[ltsFace][pointIndex] +
               impAndEta[ltsFace].invEtaS * shearStress[pointIndex];

    double eps = std::numeric_limits<double>::epsilon();
    double fEvalAtA = function(a, pointIndex);
    // if(std::isinf(fEvalAtA)){
    //  fEvalAtA = std::numeric_limits<double>::max();
    //}
    double fEvalAtB = function(b, pointIndex);
    assert(std::copysign(fEvalAtA, fEvalAtB) !=
           fEvalAtA); // fEvalAtA and fEvalAtB have different signs
    double c = a;
    double fEvalAtC = fEvalAtA;
    double d = b - a;
    double e = d;
    while (fEvalAtB != 0.0) {
      if (std::copysign(fEvalAtB, fEvalAtC) == fEvalAtB) {
        c = a;
        fEvalAtC = fEvalAtA;
        d = b - a;
        e = d;
      }
      if (std::fabs(fEvalAtC) < std::fabs(fEvalAtB)) {
        a = b;
        b = c;
        c = a;
        fEvalAtA = fEvalAtB;
        fEvalAtB = fEvalAtC;
        fEvalAtC = fEvalAtA;
      }
      // Convergence test
      double xm = 0.5 * (c - b);
      double tol1 = 2.0 * eps * std::fabs(b) + 0.5 * tol;
      if (std::fabs(xm) <= tol1 || fEvalAtB == 0.0) {
        break;
      }
      if (std::fabs(e) < tol1 || std::fabs(fEvalAtA) <= std::fabs(fEvalAtB)) {
        // bisection
        d = xm;
        e = d;
      } else {
        double s = fEvalAtB / fEvalAtA;
        double p, q;
        if (a != c) {
          // linear interpolation
          q = fEvalAtA / fEvalAtC;
          double r = fEvalAtB / fEvalAtC;
          p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
          q = (q - 1.0) * (r - 1.0) * (s - 1.0);
        } else {
          // inverse quadratic interpolation
          p = 2.0 * xm * s;
          q = 1.0 - s;
        }
        if (p > 0) {
          q = -q;
        } else {
          p = -p;
        }
        if (2.0 * p < 3.0 * xm * q - std::fabs(tol1 * q) && p < std::fabs(0.5 * e * q)) {
          e = d;
          d = p / q;
        } else {
          // bisection
          d = xm;
          e = d;
        }
      }
      a = b;
      fEvalAtA = fEvalAtB;
      if (std::fabs(d) > tol1) {
        b += d;
      } else {
        b += std::copysign(tol1, xm);
      }
      fEvalAtB = function(b, pointIndex);
    }
    srTest[pointIndex] = b;
  }
  return true;
}
void RateAndStateFastVelocityWeakeningLaw::updateMu(unsigned int ltsFace,
                                                    unsigned int pointIndex,
                                                    real localStateVariable) {
  //! X in Asinh(x) for mu calculation
  real tmp = 0.5 / drParameters.rateAndState.sr0 *
             exp(localStateVariable / a[ltsFace][pointIndex]) *
             slipRateMagnitude[ltsFace][pointIndex];
  //! mu from locSlipRate with SINH(X)=LOG(X+SQRT(X^2+1))
  mu[ltsFace][pointIndex] =
      a[ltsFace][pointIndex] * std::log(tmp + std::sqrt(std::pow(tmp, 2) + 1.0));
}

void RateAndStateThermalPressurizationLaw::initializeTP(
    seissol::Interoperability& interoperability) {
  interoperability.getDynRupTP(tpGrid, tpDFinv);
}

void RateAndStateThermalPressurizationLaw::copyLtsTreeToLocalRS(
    seissol::initializers::Layer& layerData,
    seissol::initializers::DynamicRupture* dynRup,
    real fullUpdateTime) {
  // first copy all Variables from the Base Lts dynRup tree
  RateAndStateFastVelocityWeakeningLaw::copyLtsTreeToLocalRS(layerData, dynRup, fullUpdateTime);

  // maybe change later to const_cast?
  auto* concreteLts =
      dynamic_cast<seissol::initializers::LTS_RateAndStateThermalPressurisation*>(dynRup);
  temperature = layerData.var(concreteLts->temperature);
  pressure = layerData.var(concreteLts->pressure);
  tpTheta = layerData.var(concreteLts->TP_theta);
  tpSigma = layerData.var(concreteLts->TP_sigma);
  tpHalfWidthShearZone = layerData.var(concreteLts->TP_halfWidthShearZone);
  tpAlphaHy = layerData.var(concreteLts->TP_alphaHy);
}

void RateAndStateThermalPressurizationLaw::hookSetInitialFluidPressure(
    std::array<real, numPaddedPoints>& pF, unsigned int ltsFace) {
  for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
    pF[pointIndex] = pressure[ltsFace][pointIndex];
  }
}

void RateAndStateThermalPressurizationLaw::hookCalcFluidPressure(
    std::array<real, numPaddedPoints>& pF,
    FaultStresses& faultStresses,
    bool saveTmpInTP,
    unsigned int timeIndex,
    unsigned int ltsFace) {
  for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {

    // compute fault strength (Sh)
    faultStrength[pointIndex] = -mu[ltsFace][pointIndex] *
                                (faultStresses.normalStressGP[timeIndex][pointIndex] +
                                 initialStressInFaultCS[ltsFace][pointIndex][0] - pF[pointIndex]);

    for (unsigned int tpGridIndex = 0; tpGridIndex < numberOfTPGridPoints; tpGridIndex++) {
      //! recover original values as it gets overwritten in the ThermalPressure routine
      theta[tpGridIndex] = tpTheta[ltsFace][pointIndex][tpGridIndex];
      sigma[tpGridIndex] = tpSigma[ltsFace][pointIndex][tpGridIndex];
    }
    //! use Theta/Sigma from last call in this update, dt/2 and new SR from NS
    updateTemperatureAndPressure(pointIndex, timeIndex, ltsFace);

    pF[pointIndex] = pressure[ltsFace][pointIndex];
    if (saveTmpInTP) {
      for (unsigned int tpGridIndex = 0; tpGridIndex < numberOfTPGridPoints; tpGridIndex++) {
        tpTheta[ltsFace][pointIndex][tpGridIndex] = theta[tpGridIndex];
        tpSigma[ltsFace][pointIndex][tpGridIndex] = sigma[tpGridIndex];
      }
    }
  }
}

void RateAndStateThermalPressurizationLaw::updateTemperatureAndPressure(unsigned int pointIndex,
                                                                        unsigned int timeIndex,
                                                                        unsigned int ltsFace) {
  real localTemperature = 0.0;
  real localPressure = 0.0;

  real tauV = faultStrength[pointIndex] *
              slipRateMagnitude[ltsFace][pointIndex]; //! fault strenght*slip rate
  real lambdaPrime = drParameters.rateAndState.tpLambda * drParameters.rateAndState.tpAlpha /
                     (tpAlphaHy[ltsFace][pointIndex] - drParameters.rateAndState.tpAlpha);

  real tmp[numberOfTPGridPoints];
  real omegaTheta[numberOfTPGridPoints]{};
  real omegaSigma[numberOfTPGridPoints]{};
  real thetaCurrent[numberOfTPGridPoints]{};
  real sigmaCurrent[numberOfTPGridPoints]{};
  for (unsigned int tpGridIndex = 0; tpGridIndex < numberOfTPGridPoints; tpGridIndex++) {
    //! Gaussian shear zone in spectral domain, normalized by w
    tmp[tpGridIndex] = std::pow(tpGrid[tpGridIndex] / tpHalfWidthShearZone[ltsFace][pointIndex], 2);
    //! 1. Calculate diffusion of the field at previous timestep

    //! temperature
    thetaCurrent[tpGridIndex] = theta[tpGridIndex] * exp(-drParameters.rateAndState.tpAlpha *
                                                         deltaT[timeIndex] * tmp[tpGridIndex]);
    //! pore pressure + lambda'*temp
    sigmaCurrent[tpGridIndex] = sigma[tpGridIndex] * exp(-tpAlphaHy[ltsFace][pointIndex] *
                                                         deltaT[timeIndex] * tmp[tpGridIndex]);

    //! 2. Add current contribution and get new temperature
    omegaTheta[tpGridIndex] =
        heatSource(tmp[tpGridIndex], drParameters.rateAndState.tpAlpha, tpGridIndex, timeIndex);
    theta[tpGridIndex] = thetaCurrent[tpGridIndex] +
                         (tauV / drParameters.rateAndState.rhoC) * omegaTheta[tpGridIndex];
    omegaSigma[tpGridIndex] =
        heatSource(tmp[tpGridIndex], tpAlphaHy[ltsFace][pointIndex], tpGridIndex, timeIndex);
    sigma[tpGridIndex] =
        sigmaCurrent[tpGridIndex] + ((drParameters.rateAndState.tpLambda + lambdaPrime) * tauV) /
                                        (drParameters.rateAndState.rhoC) * omegaSigma[tpGridIndex];

    //! 3. Recover temperature and pressure using inverse Fourier
    //! transformation with the calculated fourier coefficients

    //! new contribution
    localTemperature +=
        (tpDFinv[tpGridIndex] / tpHalfWidthShearZone[ltsFace][pointIndex]) * theta[tpGridIndex];
    localPressure +=
        (tpDFinv[tpGridIndex] / tpHalfWidthShearZone[ltsFace][pointIndex]) * sigma[tpGridIndex];
  }
  // Update pore pressure change (sigma = pore pressure + lambda'*temp)
  // In the BIEM code (Lapusta) they use T without initial value
  localPressure = localPressure - lambdaPrime * localTemperature;

  // Temp and pore pressure change at single GP on the fault + initial values
  temperature[ltsFace][pointIndex] =
      localTemperature + drParameters.rateAndState.initialTemperature;
  pressure[ltsFace][pointIndex] = -localPressure + drParameters.rateAndState.initialPressure;
}

real RateAndStateThermalPressurizationLaw::heatSource(real tmp,
                                                      real alpha,
                                                      unsigned int tpGridIndex,
                                                      unsigned int timeIndex) {
  //! original function in spatial domain
  //! omega = 1/(w*sqrt(2*pi))*exp(-0.5*(z/TP_halfWidthShearZone).^2);
  //! function in the wavenumber domain *including additional factors in front of the heat source
  //! function* omega =
  //! 1/(*alpha*Dwn**2**(sqrt(2.0*pi))*exp(-0.5*(Dwn*TP_halfWidthShearZone)**2)*(1-exp(-alpha**dt**tmp))
  //! inserting Dwn/TP_halfWidthShearZone (scaled) for Dwn cancels out TP_halfWidthShearZone
  return 1.0 / (alpha * tmp * (sqrt(2.0 * M_PI))) * exp(-0.5 * std::pow(tpGrid[tpGridIndex], 2)) *
         (1.0 - exp(-alpha * deltaT[timeIndex] * tmp));
}
} // namespace seissol::dr::friction_law