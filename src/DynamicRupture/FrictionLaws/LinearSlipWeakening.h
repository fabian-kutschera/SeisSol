#ifndef SEISSOL_LINEARSLIPWEAKENING_H
#define SEISSOL_LINEARSLIPWEAKENING_H

#include "BaseFrictionLaw.h"

namespace seissol::dr::friction_law {
template <class Derived>
class LinearSlipWeakeningLaw;              // generalization of Linear slip weakening laws
class LinearSlipWeakeningLawFL2;           // linear slip weakening
class LinearSlipWeakeningLawFL16;          // Linear slip weakening forced time rapture
class LinearSlipWeakeningLawBimaterialFL6; // solver for bimaterial faults, currently has a bug,
                                           // solution to the bug inside the function
} // namespace seissol::dr::friction_law

/*
 * Abstract Class implementing the general structure of linear slip weakening friction laws.
 * specific implementation is done by overriding and implementing the hook functions (via CRTP).
 */
template <class Derived>
class seissol::dr::friction_law::LinearSlipWeakeningLaw
    : public seissol::dr::friction_law::BaseFrictionLaw {
  protected:
  static constexpr real u_0 = 10e-14; // critical velocity at which slip rate is considered as being
                                      // zero for instaneous healing
  real (*d_c)[numPaddedPoints];
  real (*mu_S)[numPaddedPoints];
  real (*mu_D)[numPaddedPoints];
  bool (*DS)[numPaddedPoints];
  real (*dynStress_time)[numPaddedPoints];

  public:
  virtual void
      evaluate(seissol::initializers::Layer& layerData,
               seissol::initializers::DynamicRupture* dynRup,
               real (*QInterpolatedPlus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
               real (*QInterpolatedMinus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
               real fullUpdateTime,
               double timeWeights[CONVERGENCE_ORDER]) = 0;

  protected:
  /*
   * copies all parameters from the DynamicRupture LTS to the local attributes
   */
  void copyLtsTreeToLocal(seissol::initializers::Layer& layerData,
                          seissol::initializers::DynamicRupture* dynRup,
                          real fullUpdateTime) override {
    // first copy all Variables from the Base Lts dynRup tree
    BaseFrictionLaw::copyLtsTreeToLocal(layerData, dynRup, fullUpdateTime);

    seissol::initializers::LTS_LinearSlipWeakeningFL2* ConcreteLts =
        dynamic_cast<seissol::initializers::LTS_LinearSlipWeakeningFL2*>(dynRup);
    d_c = layerData.var(ConcreteLts->d_c);
    mu_S = layerData.var(ConcreteLts->mu_S);
    mu_D = layerData.var(ConcreteLts->mu_D);
    DS = layerData.var(ConcreteLts->DS);
    averaged_Slip = layerData.var(ConcreteLts->averaged_Slip);
    dynStress_time = layerData.var(ConcreteLts->dynStress_time);
  }

  /*
   * Hook for FL16 to set tn equal to m_fullUpdateTime outside the calculation loop
   */
  virtual void setTimeHook(unsigned int ltsFace) {}

  /*
   *  compute the slip rate and the traction from the fault strength and fault stresses
   *  also updates the directional slipStrike and slipDip
   */
  virtual void calcSlipRateAndTraction(std::array<real, numPaddedPoints>& Strength,
                                       FaultStresses& faultStresses,
                                       unsigned int timeIndex,
                                       unsigned int ltsFace) {
    std::array<real, numPaddedPoints> TotalShearStressYZ;
    for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
      //-------------------------------------
      // calculate TotalShearStress in Y and Z direction
      TotalShearStressYZ[pointIndex] =
          std::sqrt(std::pow(initialStressInFaultCS[ltsFace][pointIndex][3] +
                                 faultStresses.XYStressGP[timeIndex][pointIndex],
                             2) +
                    std::pow(initialStressInFaultCS[ltsFace][pointIndex][5] +
                                 faultStresses.XZStressGP[timeIndex][pointIndex],
                             2));

      //-------------------------------------
      // calculate SlipRates
      slipRateMagnitude[ltsFace][pointIndex] = std::max(
          static_cast<real>(0.0),
          (TotalShearStressYZ[pointIndex] - Strength[pointIndex]) * impAndEta[ltsFace].inv_eta_s);

      slipRateStrike[ltsFace][pointIndex] = slipRateMagnitude[ltsFace][pointIndex] *
                                            (initialStressInFaultCS[ltsFace][pointIndex][3] +
                                             faultStresses.XYStressGP[timeIndex][pointIndex]) /
                                            TotalShearStressYZ[pointIndex];
      slipRateDip[ltsFace][pointIndex] = slipRateMagnitude[ltsFace][pointIndex] *
                                         (initialStressInFaultCS[ltsFace][pointIndex][5] +
                                          faultStresses.XZStressGP[timeIndex][pointIndex]) /
                                         TotalShearStressYZ[pointIndex];

      //-------------------------------------
      // calculateTraction
      faultStresses.XYTractionResultGP[timeIndex][pointIndex] =
          faultStresses.XYStressGP[timeIndex][pointIndex] -
          impAndEta[ltsFace].eta_s * slipRateStrike[ltsFace][pointIndex];
      faultStresses.XZTractionResultGP[timeIndex][pointIndex] =
          faultStresses.XZStressGP[timeIndex][pointIndex] -
          impAndEta[ltsFace].eta_s * slipRateDip[ltsFace][pointIndex];
      tractionXY[ltsFace][pointIndex] = faultStresses.XYTractionResultGP[timeIndex][pointIndex];
      tractionXZ[ltsFace][pointIndex] = faultStresses.XYTractionResultGP[timeIndex][pointIndex];

      //-------------------------------------
      // update Directional Slip
      slipStrike[ltsFace][pointIndex] += slipRateStrike[ltsFace][pointIndex] * deltaT[timeIndex];
      slipDip[ltsFace][pointIndex] += slipRateDip[ltsFace][pointIndex] * deltaT[timeIndex];
    }
  }

  /*
   * evaluate friction law: updated mu -> friction law
   * for example see Carsten Uphoff's thesis: Eq. 2.45
   */
  virtual void frictionFunctionHook(std::array<real, numPaddedPoints>& stateVariablePsi,
                                    unsigned int ltsFace) {
    for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
      mu[ltsFace][pointIndex] =
          mu_S[ltsFace][pointIndex] -
          (mu_S[ltsFace][pointIndex] - mu_D[ltsFace][pointIndex]) * stateVariablePsi[pointIndex];
    }
  }

  /*
   * instantaneous healing option Reset Mu and Slip
   */
  virtual void instantaneousHealing(unsigned int ltsFace) {
    for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {
      if (slipRateMagnitude[ltsFace][pointIndex] < u_0) {
        mu[ltsFace][pointIndex] = mu_S[ltsFace][pointIndex];
        slip[ltsFace][pointIndex] = 0.0;
      }
    }
  }
  /*
   * output time when shear stress is equal to the dynamic stress after rupture arrived
   * currently only for linear slip weakening
   */
  virtual void saveDynamicStressOutput(unsigned int ltsFace) {
    for (int pointIndex = 0; pointIndex < numPaddedPoints; pointIndex++) {

      if (ruptureTime[ltsFace][pointIndex] > 0.0 &&
          ruptureTime[ltsFace][pointIndex] <= m_fullUpdateTime && DS[pointIndex] &&
          std::fabs(slip[ltsFace][pointIndex]) >= d_c[ltsFace][pointIndex]) {
        dynStress_time[ltsFace][pointIndex] = m_fullUpdateTime;
        DS[ltsFace][pointIndex] = false;
      }
    }
  }

}; // End of Class LinearSlipWeakeningLaw

class seissol::dr::friction_law::LinearSlipWeakeningLawFL2
    : public seissol::dr::friction_law::LinearSlipWeakeningLaw<
          seissol::dr::friction_law::LinearSlipWeakeningLawFL2> {
  public:
      virtual void evaluate(seissol::initializers::Layer& layerData,
               seissol::initializers::DynamicRupture* dynRup,
               real (*QInterpolatedPlus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
               real (*QInterpolatedMinus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
               real fullUpdateTime,
               double timeWeights[CONVERGENCE_ORDER]) override;

  /*
   * compute fault strength
   */
  virtual void calcStrengthHook(std::array<real, numPaddedPoints>& Strength,
                                FaultStresses& faultStresses,
                                unsigned int timeIndex,
                                unsigned int ltsFace);

  /*
   * compute state variable
   */
  virtual void calcStateVariableHook(std::array<real, numPaddedPoints>& stateVariablePsi,
                                     std::array<real, numPaddedPoints>& outputSlip,
                                     dynamicRupture::kernel::resampleParameter& resampleKrnl,
                                     unsigned int timeIndex,
                                     unsigned int ltsFace);
}; // End of Class LinearSlipWeakeningLawFL2

class seissol::dr::friction_law::LinearSlipWeakeningLawFL16
    : public seissol::dr::friction_law::LinearSlipWeakeningLawFL2 {
      public:
      void evaluate(seissol::initializers::Layer& layerData,
               seissol::initializers::DynamicRupture* dynRup,
               real (*QInterpolatedPlus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
               real (*QInterpolatedMinus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
               real fullUpdateTime,
               double timeWeights[CONVERGENCE_ORDER]) {};
  protected:
  real (*forced_rupture_time)[numPaddedPoints];
  real* tn;

  /*
   * copies all parameters from the DynamicRupture LTS to the local attributes
   */
  void copyLtsTreeToLocal(seissol::initializers::Layer& layerData,
                          seissol::initializers::DynamicRupture* dynRup,
                          real fullUpdateTime) override;

  virtual void setTimeHook(unsigned int ltsFace) override;

  virtual void calcStateVariableHook(std::array<real, numPaddedPoints>& stateVariablePsi,
                                     std::array<real, numPaddedPoints>& outputSlip,
                                     dynamicRupture::kernel::resampleParameter& resampleKrnl,
                                     unsigned int timeIndex,
                                     unsigned int ltsFace) override;
}; // End of Class LinearSlipWeakeningLawFL16

/*
 * Law for Bimaterial faults, implements strength regularization (according to prakash clifton)
 * currently regularized strength is not used (bug)
 * State variable (slip) is not resampled in this friction law!
 */
class seissol::dr::friction_law::LinearSlipWeakeningLawBimaterialFL6
    : public seissol::dr::friction_law::LinearSlipWeakeningLaw<
          seissol::dr::friction_law::LinearSlipWeakeningLawBimaterialFL6> {
      public:
      void evaluate(seissol::initializers::Layer& layerData,
               seissol::initializers::DynamicRupture* dynRup,
               real (*QInterpolatedPlus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
               real (*QInterpolatedMinus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
               real fullUpdateTime,
               double timeWeights[CONVERGENCE_ORDER]) {};
  virtual void calcStrengthHook(std::array<real, numPaddedPoints>& Strength,
                                FaultStresses& faultStresses,
                                unsigned int timeIndex,
                                unsigned int ltsFace);

  virtual void calcStateVariableHook(std::array<real, numPaddedPoints>& stateVariablePsi,
                                     std::array<real, numPaddedPoints>& outputSlip,
                                     dynamicRupture::kernel::resampleParameter& resampleKrnl,
                                     unsigned int timeIndex,
                                     unsigned int ltsFace);

  protected:
  // Attributes
  real (*strengthData)[numPaddedPoints];

  /*
   * copies all parameters from the DynamicRupture LTS to the local attributes
   */
  void copyLtsTreeToLocal(seissol::initializers::Layer& layerData,
                          seissol::initializers::DynamicRupture* dynRup,
                          real fullUpdateTime) override;

  /*
   * calculates strength
   */
  void prak_clif_mod(real& strength, real& sigma, real& LocSlipRate, real& mu, real& dt);
};

#endif // SEISSOL_LINEARSLIPWEAKENING_H
