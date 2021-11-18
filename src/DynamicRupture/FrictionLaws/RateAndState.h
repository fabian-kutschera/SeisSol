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
class RateAndStateSolver : public BaseFrictionLaw {
  public:
  using BaseFrictionLaw::BaseFrictionLaw;

  protected:
  /**
   * PARAMETERS of THE optimisation loops
   * absolute tolerance on the function to be optimzed
   * This value is quite arbitrary (a bit bigger as the expected numerical error) and may not be
   * the most adapted Number of iteration in the loops
   */
  const unsigned int nSRupdates = 60;
  const unsigned int nSVupdates = 2;

  public:
  virtual void
      evaluate(seissol::initializers::Layer& layerData,
               seissol::initializers::DynamicRupture* dynRup,
               real (*qInterpolatedPlus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
               real (*qInterpolatedMinus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
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
      bool hasConverged = false;
      FaultStresses faultStresses = {};
      real deltaStateVar[numPaddedPoints] = {0};
      std::array<real, numPaddedPoints> tmpSlip{0}; // required for averageSlip calculation
      std::array<real, numPaddedPoints> normalStress{0};
      std::array<real, numPaddedPoints> totalShearStressYZ{0};
      std::array<real, numPaddedPoints> stateVarZero{0};
      std::array<real, numPaddedPoints> tmpSlipRate{0};
      std::array<real, numPaddedPoints> localStateVariable{0};
      std::array<real, numPaddedPoints> srTest{0};

      // for thermalPressure
      std::array<real, numPaddedPoints> pF{0};

      // compute Godunov state
      precomputeStressFromQInterpolated(
          faultStresses, qInterpolatedPlus[ltsFace], qInterpolatedMinus[ltsFace], ltsFace);

      // Compute Initial stress (only for FL103), and set initial StateVariable
      static_cast<Derived*>(this)->setInitialValues(localStateVariable, ltsFace);

      for (int timeIndex = 0; timeIndex < CONVERGENCE_ORDER; timeIndex++) {
        // compute initial slip rates
        static_cast<Derived*>(this)->calcInitialSlipRate(totalShearStressYZ,
                                                         faultStresses,
                                                         stateVarZero,
                                                         localStateVariable,
                                                         tmpSlipRate,
                                                         timeIndex,
                                                         ltsFace);
        // compute initial thermal pressure (ony for FL103 TP)
        static_cast<Derived*>(this)->hookSetInitialFluidPressure(pF, ltsFace);

        for (unsigned int j = 0; j < nSVupdates; j++) {
          // compute pressure from thermal pressurization (only FL103 TP)
          static_cast<Derived*>(this)->hookCalcFluidPressure(
              pF, faultStresses, false, timeIndex, ltsFace);
          // compute slip rates by solving non-linear system of equations (with newton)
          static_cast<Derived*>(this)->updateStateVariableIterative(hasConverged,
                                                                    stateVarZero,
                                                                    tmpSlipRate,
                                                                    localStateVariable,
                                                                    pF,
                                                                    normalStress,
                                                                    totalShearStressYZ,
                                                                    srTest,
                                                                    faultStresses,
                                                                    timeIndex,
                                                                    ltsFace);
        } // End nSVupdates-loop   j=1,nSVupdates   !This loop corrects SV values

        // check for convergence
        if (!hasConverged) {
          static_cast<Derived*>(this)->executeIfNotConverged(localStateVariable, ltsFace);
        }
        // compute final thermal pressure for FL103TP
        static_cast<Derived*>(this)->hookCalcFluidPressure(
            pF, faultStresses, true, timeIndex, ltsFace);
        // compute final slip rates and traction from median value of the iterative solution and the
        // initial guess
        static_cast<Derived*>(this)->calcSlipRateAndTraction(stateVarZero,
                                                             tmpSlipRate,
                                                             localStateVariable,
                                                             normalStress,
                                                             totalShearStressYZ,
                                                             tmpSlip,
                                                             deltaStateVar,
                                                             faultStresses,
                                                             timeIndex,
                                                             ltsFace);

      } // End of timeIndex-loop

      // resample state variables
      static_cast<Derived*>(this)->resampleStateVar(deltaStateVar, ltsFace);

      //---------------------------------------------

      // output rupture front
      // outside of timeIndex loop in order to safe an 'if' in a loop
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
      postcomputeImposedStateFromNewStress(qInterpolatedPlus[ltsFace],
                                           qInterpolatedMinus[ltsFace],
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
class RateAndStateFastVelocityWeakeningLaw
    : public RateAndStateSolver<RateAndStateFastVelocityWeakeningLaw> {
  public:
  using RateAndStateSolver::RateAndStateSolver;

  protected:
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

  //! PARAMETERS of THE optimisation loops
  //! absolute tolerance on the function to be optimzed
  //! This value is quite arbitrary (a bit bigger as the expected numerical error) and may not be
  //! the most adapted Number of iteration in the loops
  const unsigned int numberSlipRateUpdates = 60;
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
  void updateStateVariableIterative(bool& hasConverged,
                                    std::array<real, numPaddedPoints>& stateVarZero,
                                    std::array<real, numPaddedPoints>& srTmp,
                                    std::array<real, numPaddedPoints>& localStateVariable,
                                    std::array<real, numPaddedPoints>& pF,
                                    std::array<real, numPaddedPoints>& normalStress,
                                    std::array<real, numPaddedPoints>& totalShearStressYZ,
                                    std::array<real, numPaddedPoints>& srTest,
                                    FaultStresses& faultStresses,
                                    unsigned int timeIndex,
                                    unsigned int ltsFace);

  /*
   * Since Newton may have stability issues and not convergence to a solution, output an error in
   * this case.
   */
  void executeIfNotConverged(std::array<real, numPaddedPoints>& localStateVariable,
                             unsigned ltsFace);

  /*
   * Final slip rates and stresses are computed from the final slip rate values
   */
  void calcSlipRateAndTraction(std::array<real, numPaddedPoints>& stateVarZero,
                               std::array<real, numPaddedPoints>& srTmp,
                               std::array<real, numPaddedPoints>& localStateVariable,
                               std::array<real, numPaddedPoints>& normalStress,
                               std::array<real, numPaddedPoints>& totalShearStressYZ,
                               std::array<real, numPaddedPoints>& tmpSlip,
                               real deltaStateVar[numPaddedPoints],
                               FaultStresses& faultStresses,
                               unsigned int timeIndex,
                               unsigned int ltsFace);

  /*
   * resample output state variable
   */
  void resampleStateVar(real deltaStateVar[numPaddedPoints], unsigned int ltsFace);

  // output time when shear stress is equal to the dynamic stress after rupture arrived
  // currently only for linear slip weakening
  void saveDynamicStressOutput(unsigned int face);

  // set to zero since only required for Thermal pressure
  virtual void hookSetInitialFluidPressure(std::array<real, numPaddedPoints>& fluidPressure,
                                           unsigned int ltsFace);

  // empty since only required for Thermal pressure
  virtual void hookCalcFluidPressure(std::array<real, numPaddedPoints>& fluidPressure,
                                     FaultStresses& faultStresses,
                                     bool saveTmpInTP,
                                     unsigned int timeIndex,
                                     unsigned int ltsFace);

  protected:
  /*
   * Update local state variable (localStateVariable) from reference state variable and computed
   * slip rate (srTmp)
   */
  void updateStateVariable(int pointIndex,
                           unsigned int face,
                           real sv0,
                           real timeIncrement,
                           real& srTmp,
                           real& localStateVariable);

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
  bool IterativelyInvertSR_Brent(unsigned int slipRate,
                                 int nSRupdates,
                                 std::array<real, numPaddedPoints>& localStateVariable,
                                 std::array<real, numPaddedPoints>& normalStress,
                                 std::array<real, numPaddedPoints>& shearStress,
                                 std::array<real, numPaddedPoints>& srTest);

  /*
   * update friction mu according to:
   * mu = a * arcsinh[ V/(2*V0) * exp(SV/a) ]
   */
  void updateMu(unsigned int ltsFace, unsigned int pointIndex, real localStateVariable);
}; // end class FL_103

/**
 * As a reference, see https://www.scec.org/meetings/2020/am/poster/158
 */
class RateAndStateThermalPressurizationLaw : public RateAndStateFastVelocityWeakeningLaw {
  public:
  using RateAndStateFastVelocityWeakeningLaw::RateAndStateFastVelocityWeakeningLaw;

  protected:
  real (*temperature)[numPaddedPoints];
  real (*pressure)[numPaddedPoints];
  real (*tpTheta)[numPaddedPoints][numberOfTPGridPoints];
  real (*tpSigma)[numPaddedPoints][numberOfTPGridPoints];
  real (*tpHalfWidthShearZone)[numPaddedPoints];
  real (*tpAlphaHy)[numPaddedPoints];

  real tpGrid[numberOfTPGridPoints];
  real tpDFinv[numberOfTPGridPoints];

  real faultStrength[numPaddedPoints];
  real theta[numberOfTPGridPoints];
  real sigma[numberOfTPGridPoints];

  public:
  /*
   * initialize local attributes (used in initializer class respectively)
   */
  void initializeTP(seissol::Interoperability& interoperability);

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
  void hookSetInitialFluidPressure(std::array<real, numPaddedPoints>& pF,
                                   unsigned int ltsFace) override;

  /*
   * compute thermal pressure according to Noda and Lapusta 2010
   * bool saveTmpInTP is used to save final thermal pressure values for theta and sigma
   */
  void hookCalcFluidPressure(std::array<real, numPaddedPoints>& pF,
                             FaultStresses& faultStresses,
                             bool saveTmpInTP,
                             unsigned int timeIndex,
                             unsigned int ltsFace) override;

  /*
   * compute thermal pressure according to Noda and Lapusta 2010
   */
  void updateTemperatureAndPressure(unsigned int pointIndex,
                                    unsigned int timeIndex,
                                    unsigned int ltsFace);

  real heatSource(real tmp, real alpha, unsigned int tpGridIndex, unsigned int timeIndex);
};

} // namespace seissol::dr::friction_law
#endif // SEISSOL_RATEANDSTATE_H