#ifndef SEISSOL_RATEANDSTATE_H
#define SEISSOL_RATEANDSTATE_H

#include "BaseFrictionLaw.h"
#include "Solver/Interoperability.h"

namespace seissol::dr::friction_law {
template <class Derived>
class RateAndStateSolver;       // general concept of rate and state solver
class RateAndStateNucFL103;     // rate and state slip law with time and space dependent nucleation
class RateAndStateThermalFL103; // specialization of FL103, extended with
} // namespace seissol::dr::friction_law

/*
 * General implementation of a rate and state solver
 * Methods are inherited via CRTP and must be implemented in the child class.
 */
template <class Derived>
class seissol::dr::friction_law::RateAndStateSolver
    : public seissol::dr::friction_law::BaseFrictionLaw {

  protected:
  //! PARAMETERS of THE optimisation loops
  //! absolute tolerance on the function to be optimzed
  //! This value is quite arbitrary (a bit bigger as the expected numerical error) and may not be
  //! the most adapted Number of iteration in the loops
  const unsigned int nSRupdates = 60;
  const unsigned int nSVupdates = 2;

  public:
  virtual void
      evaluate(seissol::initializers::Layer& layerData,
               seissol::initializers::DynamicRupture* dynRup,
               real (*QInterpolatedPlus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
               real (*QInterpolatedMinus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
               real fullUpdateTime,
               double timeWeights[CONVERGENCE_ORDER]) override {
    // first copy all Variables from the Base Lts dynRup tree
    static_cast<Derived*>(this)->copyLtsTreeToLocalRS(layerData, dynRup, fullUpdateTime);

    // compute time increments (Gnuc)
    static_cast<Derived*>(this)->preCalcTime();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (unsigned ltsFace = 0; ltsFace < layerData.getNumberOfCells(); ++ltsFace) {

      // initialize local variables inside parallel face loop
      bool has_converged = false;
      FaultStresses faultStresses = {};
      real deltaStateVar[numOfPointsPadded] = {0};
      std::array<real, numOfPointsPadded> tmpSlip{0}; // required for averageSlip calculation
      std::array<real, numOfPointsPadded> normalStress{0};
      std::array<real, numOfPointsPadded> TotalShearStressYZ{0};
      std::array<real, numOfPointsPadded> stateVarZero{0};
      std::array<real, numOfPointsPadded> SR_tmp{0};
      std::array<real, numOfPointsPadded> LocSV{0};
      std::array<real, numOfPointsPadded> SRtest{0};

      // for thermalPressure
      std::array<real, numOfPointsPadded> P_f{0};

      // compute Godunov state
      precomputeStressFromQInterpolated(
          faultStresses, QInterpolatedPlus[ltsFace], QInterpolatedMinus[ltsFace], ltsFace);

      // Compute Initial stress (only for FL103), and set initial StateVariable
      static_cast<Derived*>(this)->setInitialValues(LocSV, ltsFace);

      for (int iTimeGP = 0; iTimeGP < CONVERGENCE_ORDER; iTimeGP++) {
        // compute initial slip rates
        static_cast<Derived*>(this)->calcInitialSlipRate(
            TotalShearStressYZ, faultStresses, stateVarZero, LocSV, SR_tmp, iTimeGP, ltsFace);
        // compute initial thermal pressure (ony for FL103 TP)
        static_cast<Derived*>(this)->hookSetInitialP_f(P_f, ltsFace);

        for (unsigned int j = 0; j < nSVupdates; j++) {
          // compute pressure from thermal pressurization (only FL103 TP)
          static_cast<Derived*>(this)->hookCalcP_f(P_f, faultStresses, false, iTimeGP, ltsFace);
          // compute slip rates by solving non-linear system of equations (with newton)
          static_cast<Derived*>(this)->updateStateVariableIterative(has_converged,
                                                                    stateVarZero,
                                                                    SR_tmp,
                                                                    LocSV,
                                                                    P_f,
                                                                    normalStress,
                                                                    TotalShearStressYZ,
                                                                    SRtest,
                                                                    faultStresses,
                                                                    iTimeGP,
                                                                    ltsFace);
        } // End nSVupdates-loop   j=1,nSVupdates   !This loop corrects SV values

        // check for convergence
        if (!has_converged) {
          static_cast<Derived*>(this)->executeIfNotConverged(LocSV, ltsFace);
        }
        // compute final thermal pressure for FL103TP
        static_cast<Derived*>(this)->hookCalcP_f(P_f, faultStresses, true, iTimeGP, ltsFace);
        // compute final slip rates and traction from median value of the iterative solution and the
        // initial guess
        static_cast<Derived*>(this)->calcSlipRateAndTraction(stateVarZero,
                                                             SR_tmp,
                                                             LocSV,
                                                             normalStress,
                                                             TotalShearStressYZ,
                                                             tmpSlip,
                                                             deltaStateVar,
                                                             faultStresses,
                                                             iTimeGP,
                                                             ltsFace);

      } // End of iTimeGP-loop

      // resample state variables
      static_cast<Derived*>(this)->resampleStateVar(deltaStateVar, ltsFace);

      //---------------------------------------------

      // output rupture front
      // outside of iTimeGP loop in order to safe an 'if' in a loop
      // this way, no subtimestep resolution possible
      saveRuptureFrontOutput(ltsFace);

      // save maximal slip rates
      savePeakSlipRateOutput(ltsFace);

      // output time when shear stress is equal to the dynamic stress after rupture arrived
      // currently only for linear slip weakening
      static_cast<Derived*>(this)->saveDynamicStressOutput(ltsFace);

      //---compute and store slip to determine the magnitude of an earthquake ---
      //    to this end, here the slip is computed and averaged per element
      //    in calc_seissol.f90 this value will be multiplied by the element surface
      //    and an output happened once at the end of the simulation
      saveAverageSlipOutput(tmpSlip, ltsFace);

      // compute resulting stresses (+/- side) by time integration from godunov state
      postcomputeImposedStateFromNewStress(QInterpolatedPlus[ltsFace],
                                           QInterpolatedMinus[ltsFace],
                                           faultStresses,
                                           timeWeights,
                                           ltsFace);
    } // end face loop
  }   // end evaluate function
};

/*
 * Rate and state solver FL103, time and space dependent nucleation parameters: RS_a_array,
 * RS_srW_array, RS_sl0_array
 */
class seissol::dr::friction_law::RateAndStateNucFL103
    : public seissol::dr::friction_law::RateAndStateSolver<
          seissol::dr::friction_law::RateAndStateNucFL103> {
  protected:
  // Attributes
  real (*nucleationStressInFaultCS)[numOfPointsPadded][6];
  real dt = 0;
  real Gnuc = 0;

  real (*RS_a_array)[numOfPointsPadded];
  real (*RS_srW_array)[numOfPointsPadded];
  real (*RS_sl0_array)[numOfPointsPadded];

  bool (*DS)[numOfPointsPadded];
  real (*stateVar)[numOfPointsPadded];
  real (*dynStress_time)[numOfPointsPadded];

  //! TU 7.07.16: if the SR is too close to zero, we will have problems (NaN)
  //! as a consequence, the SR is affected the AlmostZero value when too small
  const real AlmostZero = 1e-45;

  //! PARAMETERS of THE optimisation loops
  //! absolute tolerance on the function to be optimzed
  //! This value is quite arbitrary (a bit bigger as the expected numerical error) and may not be
  //! the most adapted Number of iteration in the loops
  const unsigned int nSRupdates = 60;
  const unsigned int nSVupdates = 2;

  const double aTolF = 1e-8;

  public:
  /*
   * copies all parameters from the DynamicRupture LTS to the local attributes
   */
  virtual void copyLtsTreeToLocalRS(seissol::initializers::Layer& layerData,
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
  void setInitialValues(std::array<real, numOfPointsPadded>& LocSV, unsigned int ltsFace);

  /*
   * Compute shear stress magnitude, set reference state variable (StateVarZero)
   * Computre slip rate magnitude and set it as inital guess for iterations
   */
  void calcInitialSlipRate(std::array<real, numOfPointsPadded>& TotalShearStressYZ,
                           FaultStresses& faultStresses,
                           std::array<real, numOfPointsPadded>& stateVarZero,
                           std::array<real, numOfPointsPadded>& LocSV,
                           std::array<real, numOfPointsPadded>& SR_tmp,
                           unsigned int iTimeGP,
                           unsigned int ltsFace);

  /*
   * state variable and normal stresses (required for TP) are updated from old slip rate values
   * Slip rate is computed by solving a non-linear equation iteratively
   * friction mu is updated with new slip rate values
   */
  void updateStateVariableIterative(bool& has_converged,
                                    std::array<real, numOfPointsPadded>& stateVarZero,
                                    std::array<real, numOfPointsPadded>& SR_tmp,
                                    std::array<real, numOfPointsPadded>& LocSV,
                                    std::array<real, numOfPointsPadded>& P_f,
                                    std::array<real, numOfPointsPadded>& normalStress,
                                    std::array<real, numOfPointsPadded>& TotalShearStressYZ,
                                    std::array<real, numOfPointsPadded>& SRtest,
                                    FaultStresses& faultStresses,
                                    unsigned int iTimeGP,
                                    unsigned int ltsFace);

  /*
   * Since Newton may have stability issues and not convergence to a solution, output an error in
   * this case.
   */
  void executeIfNotConverged(std::array<real, numOfPointsPadded>& LocSV, unsigned ltsFace);

  /*
   * Final slip rates and stresses are computed from the final slip rate values
   */
  void calcSlipRateAndTraction(std::array<real, numOfPointsPadded>& stateVarZero,
                               std::array<real, numOfPointsPadded>& SR_tmp,
                               std::array<real, numOfPointsPadded>& LocSV,
                               std::array<real, numOfPointsPadded>& normalStress,
                               std::array<real, numOfPointsPadded>& TotalShearStressYZ,
                               std::array<real, numOfPointsPadded>& tmpSlip,
                               real deltaStateVar[numOfPointsPadded],
                               FaultStresses& faultStresses,
                               unsigned int iTimeGP,
                               unsigned int ltsFace);

  /*
   * resample output state variable
   */
  void resampleStateVar(real deltaStateVar[numOfPointsPadded], unsigned int ltsFace);

  // output time when shear stress is equal to the dynamic stress after rupture arrived
  // currently only for linear slip weakening
  void saveDynamicStressOutput(unsigned int face);

  // set to zero since only required for Thermal pressure
  virtual void hookSetInitialP_f(std::array<real, numOfPointsPadded>& P_f, unsigned int ltsFace);

  // empty since only required for Thermal pressure
  virtual void hookCalcP_f(std::array<real, numOfPointsPadded>& P_f,
                           FaultStresses& faultStresses,
                           bool saveTmpInTP,
                           unsigned int iTimeGP,
                           unsigned int ltsFace);

  protected:
  /*
   * Update local state variable (LocSV) from reference state variable and computed slip rate
   * (SR_tmp)
   */
  void updateStateVariable(
      int iBndGP, unsigned int face, real SV0, real time_inc, real& SR_tmp, real& LocSV);

  /*
   * If the function did not converge it returns false
   * Newton method to solve non-linear equation of slip rate
   * solution returned in SRtest
   */
  bool IterativelyInvertSR(unsigned int ltsFace,
                           int nSRupdates,
                           std::array<real, numOfPointsPadded>& LocSV,
                           std::array<real, numOfPointsPadded>& n_stress,
                           std::array<real, numOfPointsPadded>& sh_stress,
                           std::array<real, numOfPointsPadded>& SRtest);

  /*
   * Alternative to IterativelyInvertSR: Instead of newton, brents method is used:
   * slower convergence but higher stability
   * if the function did not converge it returns false
   * From Upoffc git Zero.ccp (https://github.com/TEAR-ERC/tandem/blob/master/src/util/Zero.cpp)
   */
  bool IterativelyInvertSR_Brent(unsigned int ltsFace,
                                 int nSRupdates,
                                 std::array<real, numOfPointsPadded>& LocSV,
                                 std::array<real, numOfPointsPadded>& n_stress,
                                 std::array<real, numOfPointsPadded>& sh_stress,
                                 std::array<real, numOfPointsPadded>& SRtest);

  /*
   * update friction mu according to:
   * mu = a * arcsinh[ V/(2*V0) * exp(SV/a) ]
   */
  void updateMu(unsigned int ltsFace, unsigned int iBndGP, real LocSV);
}; // end class FL_103

class seissol::dr::friction_law::RateAndStateThermalFL103
    : public seissol::dr::friction_law::RateAndStateNucFL103 {
  protected:
  real (*temperature)[numOfPointsPadded];
  real (*pressure)[numOfPointsPadded];
  real (*TP_Theta)[numOfPointsPadded][TP_grid_nz];
  real (*TP_sigma)[numOfPointsPadded][TP_grid_nz];
  real (*TP_half_width_shear_zone)[numOfPointsPadded];
  real (*alpha_hy)[numOfPointsPadded];

  real TP_grid[TP_grid_nz];
  real TP_DFinv[TP_grid_nz];

  real faultStrength[numOfPointsPadded];
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
  virtual void copyLtsTreeToLocalRS(seissol::initializers::Layer& layerData,
                                    seissol::initializers::DynamicRupture* dynRup,
                                    real fullUpdateTime) override;

  protected:
  /*
   * set initial value of thermal pressure
   */
  void hookSetInitialP_f(std::array<real, numOfPointsPadded>& P_f, unsigned int ltsFace) override;

  /*
   * compute thermal pressure according to Noda and Lapusta 2010
   * bool saveTmpInTP is used to save final thermal pressure values for theta and sigma
   */
  void hookCalcP_f(std::array<real, numOfPointsPadded>& P_f,
                   FaultStresses& faultStresses,
                   bool saveTmpInTP,
                   unsigned int iTimeGP,
                   unsigned int ltsFace) override;

  /*
   * compute thermal pressure according to Noda and Lapusta 2010
   */
  void Calc_ThermalPressure(unsigned int iBndGP, unsigned int iTimeGP, unsigned int ltsFace);

  real heat_source(real tmp, real alpha, unsigned int iTP_grid_nz, unsigned int iTimeGP);
};

#endif // SEISSOL_RATEANDSTATE_H