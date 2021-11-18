#ifndef SEISSOL_BASEFRICTIONLAW_H
#define SEISSOL_BASEFRICTIONLAW_H

#include <yaml-cpp/yaml.h>

#include "DynamicRupture/Parameters.h"
#include "Initializer/DynamicRupture.h"
#include "Kernels/DynamicRupture.h"

namespace seissol::dr::friction_law {
// Base class, has implementations of methods that are used by each friction law
class BaseFrictionLaw {
  public:
  BaseFrictionLaw(dr::DRParameters& drParameters) : drParameters(drParameters){};

  virtual ~BaseFrictionLaw(){};

  protected:
  static constexpr int numberOfPoints = tensor::QInterpolated::Shape[0]; // DISC%Galerkin%nBndGP
  static constexpr int numPaddedPoints =
      init::QInterpolated::Stop[0]; // number of points padded to next dividable number by four
  // YAML::Node m_InputParam;
  dr::DRParameters& drParameters;
  ImpedancesAndEta* impAndEta;
  real fullUpdateTime;
  real deltaT[CONVERGENCE_ORDER] = {};
  // CS = coordinate system
  real (*initialStressInFaultCS)[numPaddedPoints][6];
  real (*cohesion)[numPaddedPoints];
  real (*mu)[numPaddedPoints];
  real (*slip)[numPaddedPoints];
  real (*slipStrike)[numPaddedPoints];
  real (*slipDip)[numPaddedPoints];
  real (*slipRateMagnitude)[numPaddedPoints];
  real (*slipRateStrike)[numPaddedPoints];
  real (*slipRateDip)[numPaddedPoints];
  real (*ruptureTime)[numPaddedPoints];
  bool (*ruptureFront)[numPaddedPoints];
  real (*peakSlipRate)[numPaddedPoints];
  real (*tractionXY)[numPaddedPoints];
  real (*tractionXZ)[numPaddedPoints];
  real (*imposedStatePlus)[tensor::QInterpolated::size()];
  real (*imposedStateMinus)[tensor::QInterpolated::size()];

  // be careful only for some FLs initialized:
  real* averagedSlip;

  /*
   * Struct that contains all input stresses and output stresses
   * IN: normalStressGP, stressXYGP, stressXZGP (Godunov stresses computed by
   * precomputeStressFromQInterpolated) OUT: tractionXYResultGP, tractionXZResultGP and
   * normalStressGP (used to compute resulting +/- sided stress results by
   * postcomputeImposedStateFromNewStress)
   */
  struct FaultStresses {
    real tractionXYResultGP[CONVERGENCE_ORDER][numPaddedPoints] = {
        {}}; // OUT: updated Traction 2D array with size [1:i_numberOfPoints, CONVERGENCE_ORDER]
    real tractionXZResultGP[CONVERGENCE_ORDER][numPaddedPoints] = {
        {}}; // OUT: updated Traction 2D array with size [1:i_numberOfPoints, CONVERGENCE_ORDER]
    real normalStressGP[CONVERGENCE_ORDER][numPaddedPoints] = {{}};
    real stressXYGP[CONVERGENCE_ORDER][numPaddedPoints] = {{}};
    real stressXZGP[CONVERGENCE_ORDER][numPaddedPoints] = {{}};
  };

  /*
   * copies all parameters from the DynamicRupture LTS to the local attributes
   */
  virtual void copyLtsTreeToLocal(seissol::initializers::Layer& layerData,
                                  seissol::initializers::DynamicRupture* dynRup,
                                  real fullUpdateTime);

  /*
   * output:
   * NorStressGP, stressXYGP, stressXZGP
   *
   * input:
   * QInterpolatedPlus, QInterpolatedMinus, eta_p, Zp, Zp_neig, eta_s, Zs, Zs_neig
   *
   * Calculate godunov state from jump of plus and minus side
   * using equations (A2) from Pelites et al. 2014
   * Definiton of eta and impedance Z are found in dissertation of Carsten Uphoff
   */
  virtual void precomputeStressFromQInterpolated(
      FaultStresses& faultStresses,
      real qInterpolatedPlus[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
      real qInterpolatedMinus[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
      unsigned int ltsFace);

  /*
   * Output: imposedStatePlus, imposedStateMinus
   *
   * Integrate over all Time points with the time weights and calculate the traction vor each side
   * according to Carsten Uphoff Thesis: EQ.: 4.60 IN: normalStressGP, tractionXYResultGP,
   * tractionXZResultGP OUT: imposedStatePlus, imposedStateMinus
   */
  void postcomputeImposedStateFromNewStress(
      real qInterpolatedPlus[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
      real qInterpolatedMinus[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
      const FaultStresses& faultStresses,
      double timeWeights[CONVERGENCE_ORDER],
      unsigned int ltsFace);

  /*
   * https://strike.scec.org/cvws/download/SCEC_validation_slip_law.pdf
   */
  real calcSmoothStepIncrement(real currentTime, real dt);

  /*
   * https://strike.scec.org/cvws/download/SCEC_validation_slip_law.pdf
   */
  real calcSmoothStep(real currentTime);

  /*
   * output rupture front, saves update time of the rupture front
   * rupture front is the first registered change in slip rates that exceeds 0.001
   */
  void saveRuptureFrontOutput(unsigned int ltsFace);

  /*
   * save the maximal computed slip rate magnitude in RpeakSlipRate
   */
  void savePeakSlipRateOutput(unsigned int ltsFace);

  //---compute and store slip to determine the magnitude of an earthquake ---
  //    to this end, here the slip is computed and averaged per element
  //    in calc_seissol.f90 this value will be multiplied by the element surface
  //    and an output happened once at the end of the simulation
  void saveAverageSlipOutput(std::array<real, numPaddedPoints>& tmpSlip, unsigned int ltsFace);

  public:
  /*
   * evaluates the current friction model
   * Friction laws (child classes) implement this function
   */
  virtual void
      evaluate(seissol::initializers::Layer& layerData,
               seissol::initializers::DynamicRupture* dynRup,
               real (*qInterpolatedPlus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
               real (*qInterpolatedMinus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
               real fullUpdateTime,
               double timeWeights[CONVERGENCE_ORDER]) = 0;

  /*
   * compute the DeltaT from the current timePoints
   * call this function before evaluate to set the correct DeltaT
   */
  void computeDeltaT(double timePoints[CONVERGENCE_ORDER]);
};
} // namespace seissol::dr::friction_law

#endif // SEISSOL_BASEFRICTIONLAW_H
