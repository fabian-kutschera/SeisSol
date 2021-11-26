#ifndef SEISSOL_RATEANDSTATE_H
#define SEISSOL_RATEANDSTATE_H

#include "BaseFrictionLaw.h"
#include "Solver/Interoperability.h"

namespace seissol::dr::friction_law {
/**
 * General implementation of a rate and state solver
 * Methods are inherited via CRTP and must be implemented in the child class.
 */
template <class Derived>
class RateAndStateBase : public BaseFrictionLaw<RateAndStateBase<Derived>> {
  public:
  // Attributes
  // CS = coordinate system
  real (*nucleationStressInFaultCS)[numPaddedPoints][6];
  real dt = 0;
  real gNuc = 0;

  real (*a)[numPaddedPoints];
  real (*srW)[numPaddedPoints];
  real (*sl0)[numPaddedPoints];

  bool (*ds)[numPaddedPoints];
  real (*stateVariable)[numPaddedPoints];
  real (*dynStressTime)[numPaddedPoints];

  //! TU 7.07.16: if the SR is too close to zero, we will have problems (NaN)
  //! as a consequence, the SR is affected the AlmostZero value when too small
  const real almostZero = 1e-45;

  /**
   * Parameters of the optimisation loops
   * absolute tolerance on the function to be optimized
   * This value is quite arbitrary (a bit bigger as the expected numerical error) and may not be
   * the most adapted Number of iteration in the loops
   */
  const unsigned int numberSlipRateUpdates = 60;
  const unsigned int nSVupdates = 2;

  const double aTolF = 1e-8;

  using BaseFrictionLaw<RateAndStateBase<Derived>>::BaseFrictionLaw;

  protected:
  public:
  void updateFrictionAndSlip(FaultStresses& faultStresses,
                             std::array<real, numPaddedPoints>& stateVariableBuffer,
                             std::array<real, numPaddedPoints>& strengthBuffer,
                             unsigned& ltsFace,
                             unsigned& timeIndex) {
    bool has_converged = false;

    // for thermalPressure
    std::array<real, numPaddedPoints> P_f{0};
    std::array<real, numPaddedPoints> SR_tmp{0};
    std::array<real, numPaddedPoints> stateVarZero{0};
    std::array<real, numPaddedPoints> normalStress{0};
    std::array<real, numPaddedPoints> TotalShearStressYZ{0};
    std::array<real, numPaddedPoints> SRtest{0};
    // compute initial slip rates
    static_cast<Derived*>(this)->calcInitialSlipRate(TotalShearStressYZ,
                                                     faultStresses,
                                                     stateVarZero,
                                                     stateVariableBuffer,
                                                     SR_tmp,
                                                     timeIndex,
                                                     ltsFace);
    // compute initial thermal pressure (ony for FL103 TP)
    static_cast<Derived*>(this)->hookSetInitialP_f(P_f, ltsFace);

    for (unsigned int j = 0; j < nSVupdates; j++) {
      // compute pressure from thermal pressurization (only FL103 TP)
      static_cast<Derived*>(this)->hookCalcP_f(P_f, faultStresses, false, timeIndex, ltsFace);
      // compute slip rates by solving non-linear system of equations (with newton)
      static_cast<Derived*>(this)->updateStateVariableIterative(has_converged,
                                                                stateVarZero,
                                                                SR_tmp,
                                                                stateVariableBuffer,
                                                                P_f,
                                                                normalStress,
                                                                TotalShearStressYZ,
                                                                SRtest,
                                                                faultStresses,
                                                                timeIndex,
                                                                ltsFace);
    }

    // check for convergence
    if (!has_converged) {
      static_cast<Derived*>(this)->executeIfNotConverged(stateVariableBuffer, ltsFace);
    }
    // compute final thermal pressure for FL103TP
    static_cast<Derived*>(this)->hookCalcP_f(P_f, faultStresses, true, timeIndex, ltsFace);
    // compute final slip rates and traction from median value of the iterative solution and
    // initial guess
    static_cast<Derived*>(this)->calcSlipRateAndTraction(stateVarZero,
                                                         SR_tmp,
                                                         stateVariableBuffer,
                                                         normalStress,
                                                         TotalShearStressYZ,
                                                         faultStresses,
                                                         timeIndex,
                                                         ltsFace);
  }

  void preHook(std::array<real, numPaddedPoints>& stateVariableBuffer, unsigned ltsFace) {
    // compute time increments (Gnuc)
    static_cast<Derived*>(this)->preCalcTime();

    // Compute Initial stress (only for FL103), and set initial StateVariable
    static_cast<Derived*>(this)->setInitialValues(stateVariableBuffer, ltsFace);
  }

  void postHook(std::array<real, numPaddedPoints>& stateVariableBuffer, unsigned ltsFace) {
    std::array<real, numPaddedPoints> deltaStateVar = {0};
    for (int pointIndex = 0; pointIndex < numPaddedPoints; ++pointIndex) {
      deltaStateVar[pointIndex] =
          stateVariableBuffer[pointIndex] - this->stateVariable[ltsFace][pointIndex];
    }
    // resample state variables
    static_cast<Derived*>(this)->resampleStateVar(deltaStateVar, ltsFace);
  }

  void copyLtsTreeToLocal(seissol::initializers::Layer& layerData,
                          seissol::initializers::DynamicRupture* dynRup,
                          real fullUpdateTime) {
    static_cast<Derived*>(this)->copyLtsTreeToLocal(layerData, dynRup, fullUpdateTime);
  }
};

class RateAndStateFastVelocityWeakeningLaw
    : public RateAndStateBase<RateAndStateFastVelocityWeakeningLaw> {
  public:
  using RateAndStateBase<RateAndStateFastVelocityWeakeningLaw>::RateAndStateBase;

  protected:
  public:
  /*
   * copies all parameters from the DynamicRupture LTS to the local attributes
   */
  void copyLtsTreeToLocal(seissol::initializers::Layer& layerData,
                          seissol::initializers::DynamicRupture* dynRup,
                          real fullUpdateTime);

  /*
   * compute time increments (Gnuc)
   */
  void preCalcTime();

  /*
   * compute initial stress from nucleation Stress (only in FL103)
   * and set intial value of state variables
   */
  void setInitialValues(std::array<real, numPaddedPoints>& localStateVariable,
                        unsigned int ltsFace);

  /*
   * Compute shear stress magnitude, set reference state variable (StateVarZero)
   * Computre slip rate magnitude and set it as inital guess for iterations
   */
  void calcInitialSlipRate(std::array<real, numPaddedPoints>& totalShearStressYZ,
                           FaultStresses& faultStresses,
                           std::array<real, numPaddedPoints>& stateVarZero,
                           std::array<real, numPaddedPoints>& localStateVariable,
                           std::array<real, numPaddedPoints>& temporarySlipRate,
                           unsigned int timeIndex,
                           unsigned int ltsFace);

  /*
   * state variable and normal stresses (required for TP) are updated from old slip rate values
   * Slip rate is computed by solving a non-linear equation iteratively
   * friction mu is updated with new slip rate values
   */
  void updateStateVariableIterative(bool& has_converged,
                                    std::array<real, numPaddedPoints>& stateVarZero,
                                    std::array<real, numPaddedPoints>& SR_tmp,
                                    std::array<real, numPaddedPoints>& LocSV,
                                    std::array<real, numPaddedPoints>& P_f,
                                    std::array<real, numPaddedPoints>& normalStress,
                                    std::array<real, numPaddedPoints>& TotalShearStressYZ,
                                    std::array<real, numPaddedPoints>& SRtest,
                                    FaultStresses& faultStresses,
                                    unsigned int timeIndex,
                                    unsigned int ltsFace);

  /*
   * Since Newton may have stability issues and not convergence to a solution, output an error in
   * this case.
   */
  void executeIfNotConverged(std::array<real, numPaddedPoints>& LocSV, unsigned ltsFace);

  /*
   * Final slip rates and stresses are computed from the final slip rate values
   */
  void calcSlipRateAndTraction(std::array<real, numPaddedPoints>& stateVarZero,
                               std::array<real, numPaddedPoints>& SR_tmp,
                               std::array<real, numPaddedPoints>& localStateVariable,
                               std::array<real, numPaddedPoints>& normalStress,
                               std::array<real, numPaddedPoints>& TotalShearStressYZ,
                               FaultStresses& faultStresses,
                               unsigned int timeIndex,
                               unsigned int ltsFace);

  /*
   * resample output state variable
   */
  void resampleStateVar(std::array<real, numPaddedPoints>& deltaStateVar, unsigned int ltsFace);

  // output time when shear stress is equal to the dynamic stress after rupture arrived
  // currently only for linear slip weakening
  void saveDynamicStressOutput(unsigned int face);

  // set to zero since only required for Thermal pressure
  void hookSetInitialP_f(std::array<real, numPaddedPoints>& P_f, unsigned int ltsFace);

  // empty since only required for Thermal pressure
  void hookCalcP_f(std::array<real, numPaddedPoints>& P_f,
                   FaultStresses& faultStresses,
                   bool saveTmpInTP,
                   unsigned int timeIndex,
                   unsigned int ltsFace);

  protected:
  /*
   * Update local state variable (LocSV) from reference state variable and computed slip rate
   * (SR_tmp)
   */
  void updateStateVariable(
      int pointIndex, unsigned int face, real SV0, real time_inc, real& SR_tmp, real& LocSV);

  /*
   * If the function did not converge it returns false
   * Newton method to solve non-linear equation of slip rate
   * solution returned in slipRateTest
   */
  bool IterativelyInvertSR(unsigned int ltsFace,
                           int nSRupdates,
                           std::array<real, numPaddedPoints>& localStateVariable,
                           std::array<real, numPaddedPoints>& normalStress,
                           std::array<real, numPaddedPoints>& shearStress,
                           std::array<real, numPaddedPoints>& slipRateTest);

  /*
   * Alternative to IterativelyInvertSR: Instead of newton, brents method is used:
   * slower convergence but higher stability
   * if the function did not converge it returns false
   * From Upoffc git Zero.ccp (https://github.com/TEAR-ERC/tandem/blob/master/src/util/Zero.cpp)
   */
  bool IterativelyInvertSR_Brent(unsigned int ltsFace,
                                 int nSRupdates,
                                 std::array<real, numPaddedPoints>& LocSV,
                                 std::array<real, numPaddedPoints>& n_stress,
                                 std::array<real, numPaddedPoints>& sh_stress,
                                 std::array<real, numPaddedPoints>& SRtest);

  /*
   * update friction mu according to:
   * mu = a * arcsinh[ V/(2*V0) * exp(SV/a) ]
   */
  void updateMu(unsigned int ltsFace, unsigned int pointIndex, real localStateVariable);
};

class RateAndStateThermalPressurizationLaw : public RateAndStateFastVelocityWeakeningLaw {
  public:
  using RateAndStateFastVelocityWeakeningLaw::RateAndStateFastVelocityWeakeningLaw;

  protected:
  real (*temperature)[numPaddedPoints];
  real (*pressure)[numPaddedPoints];
  real (*TP_Theta)[numPaddedPoints][TP_grid_nz];
  real (*TP_sigma)[numPaddedPoints][TP_grid_nz];
  real (*TP_halfWidthShearZone)[numPaddedPoints];
  real (*alphaHy)[numPaddedPoints];

  real TP_grid[TP_grid_nz];
  real TP_DFinv[TP_grid_nz];

  real faultStrength[numPaddedPoints];
  real Theta_tmp[TP_grid_nz];
  real Sigma_tmp[TP_grid_nz];

  public:
  /*
   * initialize local attributes (used in initializer class respectively)
   */
  void initializeTP(seissol::Interoperability& e_interoperability);

  /*
   * copies all parameters from the DynamicRupture LTS to the local attributes
   */
  void copyLtsTreeToLocal(seissol::initializers::Layer& layerData,
                          seissol::initializers::DynamicRupture* dynRup,
                          real fullUpdateTime);

  protected:
  /*
   * set initial value of thermal pressure
   */
  void hookSetInitialP_f(std::array<real, numPaddedPoints>& P_f, unsigned int ltsFace);

  /*
   * compute thermal pressure according to Noda and Lapusta 2010
   * bool saveTmpInTP is used to save final thermal pressure values for theta and sigma
   */
  void hookCalcP_f(std::array<real, numPaddedPoints>& P_f,
                   FaultStresses& faultStresses,
                   bool saveTmpInTP,
                   unsigned int timeIndex,
                   unsigned int ltsFace);

  /*
   * compute thermal pressure according to Noda and Lapusta 2010
   */
  void updateTemperatureAndPressure(unsigned int pointIndex,
                                    unsigned int timeIndex,
                                    unsigned int ltsFace);

  real heatSource(real tmp, real alpha, unsigned int iTP_grid_nz, unsigned int timeIndex);
};

} // namespace seissol::dr::friction_law
#endif // SEISSOL_RATEANDSTATE_H