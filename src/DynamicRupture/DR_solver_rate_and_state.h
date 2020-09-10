//
// Created by adrian on 03.09.20.
//

#ifndef SEISSOL_DR_SOLVER_RATE_AND_STATE_H
#define SEISSOL_DR_SOLVER_RATE_AND_STATE_H

#include "DR_friction_law.h"

#include <c++/8.3.0/iostream>
#include "DR_math.h"
#include <yaml-cpp/yaml.h>



namespace seissol {
  namespace dr {
    namespace fr_law {
      class Solver_FL_3; //rate and state aging law
      class Solver_FL_4; //rate and state slip law
      class Solver_FL_103;  //rate and state nuc103
    }
  }
}

class seissol::dr::fr_law::Solver_FL_3 : public seissol::dr::fr_law::BaseFrictionSolver {
protected:
  virtual real calcStateVariableHook(real SV0, real tmp, real time_inc, real RS_sl0) {
    return SV0*exp(-tmp*time_inc/RS_sl0)+RS_sl0/tmp*(1.0-exp(-tmp*time_inc/RS_sl0));
  }

public:
  virtual void evaluate(seissol::initializers::Layer&  layerData,
                        seissol::initializers::DynamicRupture *dynRup,
                        real (*QInterpolatedPlus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
                        real (*QInterpolatedMinus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
                        real fullUpdateTime,
                        real timeWeights[CONVERGENCE_ORDER],
                        real DeltaT[CONVERGENCE_ORDER]) override {

    seissol::initializers::DR_FL_3 *ConcreteLts = dynamic_cast<seissol::initializers::DR_FL_3 *>(dynRup);

    seissol::model::IsotropicWaveSpeeds *waveSpeedsPlus                           = layerData.var(ConcreteLts->waveSpeedsPlus);
    seissol::model::IsotropicWaveSpeeds *waveSpeedsMinus                          = layerData.var(ConcreteLts->waveSpeedsMinus);
    real                    (*initialStressInFaultCS)[numOfPointsPadded][6]       = layerData.var(ConcreteLts->initialStressInFaultCS);
    real                    (*cohesion)[numOfPointsPadded]                        = layerData.var(ConcreteLts->cohesion);
    real*                   RS_f0                                                 = layerData.var(ConcreteLts->RS_f0);
    real*                   RS_a                                                  = layerData.var(ConcreteLts->RS_a);
    real*                   RS_b                                                  = layerData.var(ConcreteLts->RS_b);
    real*                   RS_sl0                                                = layerData.var(ConcreteLts->RS_sl0);
    real*                   RS_sr0                                                = layerData.var(ConcreteLts->RS_sr0);

    real                    (*mu)[numOfPointsPadded]                              = layerData.var(ConcreteLts->mu);
    real                    (*slip)[numOfPointsPadded]                            = layerData.var(ConcreteLts->slip);
    real                    (*slip1)[numOfPointsPadded]                           = layerData.var(ConcreteLts->slip1);
    real                    (*slip2)[numOfPointsPadded]                           = layerData.var(ConcreteLts->slip2);
    real                    (*slipRate1)[numOfPointsPadded]                       = layerData.var(ConcreteLts->slipRate1);
    real                    (*slipRate2)[numOfPointsPadded]                       = layerData.var(ConcreteLts->slipRate2);
    real                    (*rupture_time)[numOfPointsPadded]                    = layerData.var(ConcreteLts->rupture_time);
    bool                    (*RF)[numOfPointsPadded]                              = layerData.var(ConcreteLts->RF);
    real                    (*peakSR)[numOfPointsPadded]                          = layerData.var(ConcreteLts->peakSR);
    real                    (*StateVar)[numOfPointsPadded]                        = layerData.var(ConcreteLts->StateVar);

    real                    (*tracXY)[numOfPointsPadded]                          = layerData.var(ConcreteLts->tracXY);
    real                    (*tracXZ)[numOfPointsPadded]                          = layerData.var(ConcreteLts->tracXZ);
    real                    (*imposedStatePlus)[tensor::QInterpolated::size()]    = layerData.var(ConcreteLts->imposedStatePlus);
    real                    (*imposedStateMinus)[tensor::QInterpolated::size()]   = layerData.var(ConcreteLts->imposedStateMinus);

    //loop parameter are fixed, not variable??
    unsigned int nSRupdates, nSVupdates;
    nSRupdates = 5;
    nSVupdates = 2;



#ifdef _OPENMP
#pragma omp parallel for schedule(static) //private(QInterpolatedPlus,QInterpolatedMinus)
#endif
    for (unsigned face = 0; face < layerData.getNumberOfCells(); ++face) {

      FaultStresses faultStresses{};

      precomputeStressFromQInterpolated(faultStresses, QInterpolatedPlus[face], QInterpolatedMinus[face], face);

      real LocSlip, LocSlip1, LocSlip2, LocSR1, LocSR2, LocSV, LocCohesion, P_0, LocP, time_inc, P, TotalShearStressYZ, SV0, tmp,tmp2, SlipRateGuess, NR, dNR, LocMu;
      real LocTracXY, LocTracXZ;
      real LocSlipRate[seissol::tensor::resamplePar::size()];

      for(int iBndGP = 0; iBndGP < numOfPointsPadded; iBndGP++) {

        LocSlip = slip[face][iBndGP]; //DISC%DynRup%Slip(iBndGP,iFace)              //!< Slip path at given fault node
        LocSlip1 = slip1[face][iBndGP]; //DISC%DynRup%Slip1(iBndGP,iFace)            //!< Slip at given fault node along loc dir 1
        LocSlip2 = slip2[face][iBndGP]; //DISC%DynRup%Slip2(iBndGP,iFace)            // !< Slip at given fault node along loc dir 2
        LocSR1 = slipRate1[face][iBndGP]; //DISC%DynRup%SlipRate1(iBndGP,iFace)         // !< Slip Rate at given fault node
        LocSR2 = slipRate2[face][iBndGP]; //DISC%DynRup%SlipRate2(iBndGP,iFace)         // !< Slip Rate at given fault node
        LocSV = StateVar[face][iBndGP];     //DISC%DynRup%StateVar(iBndGP,iFace)
        LocCohesion = cohesion[face][iBndGP]; //DISC%DynRup%cohesion(iBndGP,iFace)          // !< cohesion at given fault node  (should be negative since negative normal stress is compression)
        P_0 = initialStressInFaultCS[face][iBndGP][0]; //EQN%InitialStressInFaultCS[iBndGP][1][iFace];

        for (int iTimeGP = 0; iTimeGP < CONVERGENCE_ORDER; iTimeGP++) {
          LocP = faultStresses.NorStressGP[iTimeGP][iBndGP];
          time_inc = DeltaT[iTimeGP];

          //SignSR1   = SIGN(1.0,LocSR1)                    ! Gets the sign of the slip rate
          //SignSR2   = SIGN(1.0,LocSR2)                    ! Gets the sign of the slip rate

          // load traction and normal stress
          P = LocP + P_0;

          TotalShearStressYZ = std::sqrt(
              seissol::dr::aux::power(initialStressInFaultCS[face][iBndGP][3] + faultStresses.XYStressGP[iTimeGP][iBndGP], 2) +
              seissol::dr::aux::power(initialStressInFaultCS[face][iBndGP][5] + faultStresses.XZStressGP[iTimeGP][iBndGP], 2));

          // We use the regularized rate-and-state friction, after Rice & Ben-Zion (1996) //TODO: look up
          // ( Numerical note: ASINH(X)=LOG(X+SQRT(X^2+1)) )

          SV0 = LocSV;    // Careful, the SV must always be corrected using SV0 and not LocSV!

          // The following process is adapted from that described by Kaneko et al. (2008) TODO: look up

          LocSlipRate[iBndGP]      = std::sqrt(seissol::dr::aux::power(LocSR1,2) + seissol::dr::aux::power(LocSR2,2));
          tmp        = fabs( LocSlipRate[iBndGP]);

          for(int j = 0; j < nSVupdates; j++){ //!This loop corrects SV values
            LocSlipRate[iBndGP]=fabs( LocSlipRate[iBndGP]);

            // FL= 3 aging law and FL=4 slip law
            LocSV = calcStateVariableHook( SV0,  tmp,  time_inc,  RS_sl0[face]);

            // Newton-Raphson algorithm to determine the value of the slip rate.
            // We wish to find SR that fulfills g(SR)=f(SR), by building up the function NR=f-g , which has
            //  a derivative dNR = d(NR)/d(SR). We can then find SR by iterating SR_{i+1}=SR_i-( NR_i / dNR_i ).
            // In our case we equalize the values of the traction for two equations:
            //             g = SR*mu/2/cs + T^G             (eq. 18 of de la Puente et al. (2009))
            //             f = (mu*P_0-|S_0|)*S_0/|S_0|     (Coulomb's model of friction)
            //               where mu=a*asinh(SR/2/SR0*exp((F0+b*log(SR0*SV/L))/a (eq. 2a of Lapusta and Rice (2003))

            SlipRateGuess = LocSlipRate[iBndGP];   // SRtest: We use as first guess the SR value of the previous time step

            for(int i = 0; i < nSRupdates; i++){   //!This loop corrects SR values
              tmp          = 0.5/RS_sr0[face]* exp( (RS_f0[face]+RS_b[face]*log(RS_sr0[face]*LocSV/RS_sl0[face]) ) /RS_a[face]);
              tmp2         = tmp * SlipRateGuess;
              NR           = -(1.0/waveSpeedsPlus->sWaveVelocity/waveSpeedsPlus->density+1.0/waveSpeedsMinus->sWaveVelocity/waveSpeedsMinus->density) *
                             (fabs(P)*RS_a[face]*log(tmp2+sqrt(seissol::dr::aux::power(tmp2,2)+1.0))-TotalShearStressYZ)-SlipRateGuess;    //!TODO: author before me: not sure if ShTest=TotalShearStressYZ should be + or -...
              dNR          = -(1.0/waveSpeedsPlus->sWaveVelocity/waveSpeedsPlus->density+1.0/waveSpeedsMinus->sWaveVelocity/waveSpeedsMinus->density) *
                             (fabs(P)*RS_a[face]/sqrt(1+pow(tmp2,2))*tmp)-1.0;
              SlipRateGuess = fabs(SlipRateGuess-NR/dNR);             // no ABS needed around NR/dNR at least for aging law
            }   // End
            tmp=0.5*( LocSlipRate[iBndGP]+fabs(SlipRateGuess));  //! For the next SV update, use the mean slip rate between the initial guess and the one found (Kaneko 2008, step 6)
            LocSlipRate[iBndGP]=fabs(SlipRateGuess);
          }   // End SV-Loop

          // FL= 3 aging law and FL=4 slip law
          LocSV= calcStateVariableHook( SV0,  tmp,  time_inc,  RS_sl0[face]);

          //TODO: reused calc from above -> simplify
          tmp  = 0.5 * ( LocSlipRate[iBndGP])/RS_sr0[face] * exp((RS_f0[face] + RS_b[face]*log(RS_sr0[face]*LocSV/RS_sl0[face])) / RS_a[face]);

          LocMu    = RS_a[face] * log(tmp + sqrt(seissol::dr::aux::power(tmp,2) + 1.0));

          // 2D:
          // LocTrac  = -(ABS(S_0)-LocMu*(LocP+P_0))*(S_0/ABS(S_0))
          // LocTrac  = ABS(LocTrac)*(-SignSR)  !!! line commented as it leads NOT to correct results
          // update stress change
          LocTracXY = -((initialStressInFaultCS[face][iBndGP][3] + faultStresses.XYStressGP[iBndGP][iTimeGP])/TotalShearStressYZ)*(LocMu*P+fabs(LocCohesion));
          LocTracXZ = -((initialStressInFaultCS[face][iBndGP][5] + faultStresses.XZStressGP[iBndGP][iTimeGP])/TotalShearStressYZ)*(LocMu*P+fabs(LocCohesion));
          LocTracXY = LocTracXY - initialStressInFaultCS[face][iBndGP][3];
          LocTracXZ = LocTracXZ - initialStressInFaultCS[face][iBndGP][5];

          // Compute slip
          LocSlip   = LocSlip  + ( LocSlipRate[iBndGP])*time_inc; // ABS of LocSR removed as it would be the accumulated slip that is usually not needed in the solver, see linear slip weakening

          //Update slip rate (notice that LocSR(T=0)=-2c_s/mu*s_xy^{Godunov} is the slip rate caused by a free surface!)
          LocSR1     = -(1.0/(waveSpeedsPlus->sWaveVelocity*waveSpeedsPlus->density)+1.0/(waveSpeedsMinus->sWaveVelocity*waveSpeedsMinus->density))*(LocTracXY-faultStresses.XYStressGP[iTimeGP][iBndGP]);
          LocSR2     = -(1.0/(waveSpeedsPlus->sWaveVelocity*waveSpeedsPlus->density)+1.0/(waveSpeedsMinus->sWaveVelocity*waveSpeedsMinus->density))*(LocTracXZ-faultStresses.XZStressGP[iTimeGP][iBndGP]);

          LocSlip1   = LocSlip1  + (LocSR1)*time_inc;
          LocSlip2   = LocSlip2  + (LocSR2)*time_inc;

          //LocSR1     = SignSR1*ABS(LocSR1)
          //LocSR2     = SignSR2*ABS(LocSR2)

          //Save traction for flux computation
          faultStresses.TractionGP_XY[iTimeGP][iBndGP] = LocTracXY;
          faultStresses.TractionGP_XZ[iTimeGP][iBndGP] = LocTracXZ;
        }//End of iTimeGP- loop

        mu[face][iBndGP]       = LocMu;
        slipRate1[face][iBndGP]  = LocSR1;
        slipRate2[face][iBndGP]  = LocSR2;
        slip[face][iBndGP]       = LocSlip;
        slip1[face][iBndGP]     = LocSlip1;
        slip2[face][iBndGP]      = LocSlip2;
        StateVar[face][iBndGP]   = LocSV;
        tracXY[face][iBndGP] = LocTracXY;
        tracXZ[face][iBndGP] = LocTracXZ;

      }//End of iBndGP-loop

      // output rupture front
      // outside of iTimeGP loop in order to safe an 'if' in a loop
      // this way, no subtimestep resolution possible
      outputRuptureFront(LocSlipRate, fullUpdateTime, face);

      calcPeakSlipRate(LocSlipRate, face);


      postcomputeImposedStateFromNewStress(QInterpolatedPlus[face], QInterpolatedMinus[face],
                                           faultStresses, timeWeights, face);
    } //end face-loop
  } //end evaluate function
};

class seissol::dr::fr_law::Solver_FL_4 : public seissol::dr::fr_law::Solver_FL_3 {
public:

  virtual real calcStateVariableHook(real SV0, real tmp, real time_inc, real RS_sl0) override {
    return RS_sl0/tmp*seissol::dr::aux::power(tmp*SV0/RS_sl0, exp(-tmp*time_inc/RS_sl0));
  }

};


class seissol::dr::fr_law::Solver_FL_103 : public seissol::dr::fr_law::BaseFrictionSolver {
protected:
  //Attributes
  yateto::DenseTensorView<2,double,unsigned> resampleMatrixView = init::resample::view::create(const_cast<double *>(init::resample::Values));

  real  (*nucleationStressInFaultCS)[numOfPointsPadded][6];
//  bool  *magnitude_out;
//  real  t_0;                        //face independent
  real  RS_f0;                      //face independent
  real  RS_b;                       //face independent
  real  RS_sr0;                     //face independent
  real  Mu_w;                       //face independent
  real  (*RS_a_array)[numOfPointsPadded];
  real  (*RS_srW_array)[numOfPointsPadded];
  real  (*RS_sl0_array)[numOfPointsPadded];



  bool  (*DS)[numOfPointsPadded];
  real  *averaged_Slip;
  real  (*stateVar)[numOfPointsPadded];
  real  (*dynStress_time)[numOfPointsPadded];


  /*
 * Function in NucleationFunctions_mod
 */
  double Calc_SmoothStepIncrement(double fullUpdateTime, double Tnuc, double dt){
    double Gnuc;
    double prevtime;
    if(fullUpdateTime > 0.0 && fullUpdateTime <= Tnuc){
      Gnuc = Calc_SmoothStep(fullUpdateTime, Tnuc);
      prevtime = fullUpdateTime - dt;
      if(prevtime > 0.0){
        Gnuc = Gnuc - Calc_SmoothStep(prevtime, Tnuc);
      }
    }else{
      Gnuc = 0.0;
    }
    return Gnuc;
  }

  /*
 * Function in NucleationFunctions_mod
 */
  double Calc_SmoothStep(double fullUpdateTime, double Tnuc){
    double Gnuc;
    if (fullUpdateTime <= 0){
      Gnuc=0.0;
    }else{
      if (fullUpdateTime < Tnuc){
        Gnuc = std::exp(seissol::dr::aux::power(fullUpdateTime - Tnuc, 2) / (fullUpdateTime * (fullUpdateTime - 2.0 * Tnuc)));
      }else{
        Gnuc=1.0;
      }
    }
    return Gnuc;
  }

  void updateStateVariable(int iBndGP, unsigned int face, real SV0, real time_inc, real &SR_tmp, real &LocSV){
    double flv, fss, SVss;
    double RS_fw = Mu_w;
    double RS_srW = RS_srW_array[face][iBndGP];
    double RS_a = RS_a_array[face][iBndGP];
    double RS_sl0 = RS_sl0_array[face][iBndGP];
    double exp1;

    // low-velocity steady state friction coefficient
    flv = RS_f0 - (RS_b-RS_a)* log(SR_tmp/RS_sr0);
    // steady state friction coefficient
    fss = RS_fw + (flv - RS_fw)/pow(1.0+seissol::dr::aux::power(SR_tmp/RS_srW,8.0) ,1.0/8.0);
    // steady-state state variable
    // For compiling reasons we write SINH(X)=(EXP(X)-EXP(-X))/2
    SVss = RS_a * log(2.0*RS_sr0/SR_tmp * (exp(fss/RS_a)-exp(-fss/RS_a))/2.0);

    // exact integration of dSV/dt DGL, assuming constant V over integration step

    exp1 = exp(-SR_tmp*(time_inc/RS_sl0) );
    LocSV = SVss*(1.0-exp1)+exp1*SV0;

    //LocSV = SVss*(1.0-exp(-SR_tmp*time_inc/RS_sl0))+exp(-SR_tmp*time_inc/RS_sl0)*SV0;


    /*  //TODO log error NaN detected
    if (ANY(IsNaN(LocSV)) == true){
        logError(*) 'NaN detected'
    }
     */
    assert( !std::isnan(LocSV) && "NaN detected");
  }

  /*
   * If the function did not converge it returns false
   */
  bool IterativelyInvertSR (unsigned int face, int nSRupdates,  real LocSR[seissol::tensor::resamplePar::size()],
                            std::array<real, numOfPointsPadded> &LocSV, std::array<real, numOfPointsPadded> &n_stress,
                            std::array<real, numOfPointsPadded> &sh_stress, double invZ,  std::array<real, numOfPointsPadded> &SRtest ){

    double tmp[numberOfPoints], tmp2[numberOfPoints], tmp3[numberOfPoints], mu_f[numberOfPoints], dmu_f[numberOfPoints], NR[numberOfPoints], dNR[numberOfPoints];
    double aTolF = 1e-8;
    double AlmostZero = 1e-45;
    bool has_converged = false;

    //!solve for Vnew = SR , applying the Newton-Raphson algorithm
    //!SR fulfills g(SR)=f(SR)
    //!-> find root of NR=f-g using a Newton-Raphson algorithm with dNR = d(NR)/d(SR)
    //!SR_{i+1}=SR_i-( NR_i / dNR_i )
    //!
    //!        equalize:
    //!         g = SR*MU/2/cs + T^G             (eq. 18 of de la Puente et al. (2009))
    //!         f = (mu*P_0-|S_0|)*S_0/|S_0|     (Coulomb's model of friction)
    //!  where mu = friction coefficient, dependening on the RSF law used

    //TODO: padded?
    for(int iBndGP = 0; iBndGP < numberOfPoints; iBndGP++){
      //! first guess = SR value of the previous step
      SRtest[iBndGP] = LocSR[iBndGP];
      tmp[iBndGP]   =  0.5 / RS_sr0 *exp(LocSV[iBndGP]/RS_a_array[face][iBndGP]);
    }

    for(int i = 0; i < nSRupdates; i++){
      //TODO: padded?
      for(int iBndGP = 0; iBndGP < numberOfPoints; iBndGP++){


        //!f = ( tmp2 * ABS(LocP+P_0)- ABS(S_0))*(S_0)/ABS(S_0)
        //!g = SRtest * 1.0/(1.0/w_speed(2)/rho+1.0/w_speed_neig(2)/rho_neig) + ABS(ShTest)
        //!for compiling reasons ASINH(X)=LOG(X+SQRT(X^2+1))

        //!calculate friction coefficient
        tmp2[iBndGP]  = tmp[iBndGP]*SRtest[iBndGP];
        mu_f[iBndGP]  = RS_a_array[face][iBndGP] * log(tmp2[iBndGP]+sqrt(seissol::dr::aux::power(tmp2[iBndGP],2)+1.0));
        dmu_f[iBndGP] = RS_a_array[face][iBndGP] / sqrt(1.0+seissol::dr::aux::power(tmp2[iBndGP],2))*tmp[iBndGP];
        NR[iBndGP]    = -invZ * (fabs(n_stress[iBndGP])*mu_f[iBndGP]-sh_stress[iBndGP])-SRtest[iBndGP];
      }

      has_converged = true;

      //TODO: write max element function for absolute values
      //TODO: padded?
      for(int iBndGP = 0; iBndGP < numberOfPoints; iBndGP++){
        if (fabs(NR[iBndGP]) >= aTolF ){
          has_converged = false;
          break;
        }
      }
      if(has_converged){
        return has_converged;
      }
      for(int iBndGP = 0; iBndGP < numberOfPoints; iBndGP++){

        //!derivative of NR
        dNR[iBndGP]   = -invZ * (fabs(n_stress[iBndGP])*dmu_f[iBndGP]) - 1.0;
        //!ratio
        tmp3[iBndGP] = NR[iBndGP]/dNR[iBndGP];

        //!update SRtest
        SRtest[iBndGP] = std::max(AlmostZero,SRtest[iBndGP]-tmp3[iBndGP]);
      }
    }
  }

  /*
 * If the function did not converge it returns false
   * From Upoffc git Zero.ccp
 */
  bool IterativelyInvertSR_Brent(unsigned int face, int nSRupdates, real LocSR[seissol::tensor::resamplePar::size()],
                            std::array<real, numOfPointsPadded> &LocSV, std::array<real, numOfPointsPadded> &n_stress,
                            std::array<real, numOfPointsPadded> &sh_stress, double invZ,  std::array<real, numOfPointsPadded> &SRtest ){


    std::function<double(double, int)> F;
    double tol = 1e-24;

    double RS_fw = Mu_w;
    double *RS_srW = RS_srW_array[face];
    double *RS_a = RS_a_array[face];
    double *RS_sl0 = RS_sl0_array[face];
    double RS_sr0_ = RS_sr0;

    F = [invZ, &sh_stress, n_stress, RS_a, LocSV, RS_sr0_](double SR, int iBndGP){
      double tmp   =  0.5 / RS_sr0_ *exp(LocSV[iBndGP]/RS_a[iBndGP]);
      double tmp2  = tmp*SR;
      double mu_f  = RS_a[iBndGP] * log(tmp2+sqrt(seissol::dr::aux::power(tmp2,2)+1.0));
      return -invZ * (fabs(n_stress[iBndGP])*mu_f-sh_stress[iBndGP])-SR;
    };

    //TODO: padded?
    for(int iBndGP = 0; iBndGP < numberOfPoints; iBndGP++){
      double a = LocSR[iBndGP] -impAndEta[face].eta_s;
      double b  = LocSR[iBndGP] + impAndEta[face].eta_s;

      double eps = std::numeric_limits<double>::epsilon();
      double Fa = F(a, iBndGP);
      //if(std::isinf(Fa)){
      //  Fa = std::numeric_limits<double>::max();
      //}
      double Fb = F(b, iBndGP);
      assert(std::copysign(Fa, Fb) != Fa); // Fa and Fb have different signs
      double c = a;
      double Fc = Fa;
      double d = b - a;
      double e = d;
      while (Fb != 0.0) {
        if (std::copysign(Fb, Fc) == Fb) {
          c = a;
          Fc = Fa;
          d = b - a;
          e = d;
        }
        if (std::fabs(Fc) < std::fabs(Fb)) {
          a = b;
          b = c;
          c = a;
          Fa = Fb;
          Fb = Fc;
          Fc = Fa;
        }
        // Convergence test
        double xm = 0.5 * (c - b);
        double tol1 = 2.0 * eps * std::fabs(b) + 0.5 * tol;
        if (std::fabs(xm) <= tol1 || Fb == 0.0) {
          break;
        }
        if (std::fabs(e) < tol1 || std::fabs(Fa) <= std::fabs(Fb)) {
          // bisection
          d = xm;
          e = d;
        } else {
          double s = Fb / Fa;
          double p, q;
          if (a != c) {
            // linear interpolation
            q = Fa / Fc;
            double r = Fb / Fc;
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
        Fa = Fb;
        if (std::fabs(d) > tol1) {
          b += d;
        } else {
          b += std::copysign(tol1, xm);
        }
        Fb = F(b, iBndGP);
      }
      SRtest[iBndGP] = b;
    }

    return true;
  }

  //output time when shear stress is equal to the dynamic stress after rupture arrived
  //currently only for linear slip weakening
  void outputDynamicStress(
      real fullUpdateTime,
      unsigned int face
  ){
    for (int iBndGP = 0; iBndGP < numOfPointsPadded; iBndGP++) {

      if (rupture_time[face][iBndGP] > 0.0 &&
          rupture_time[face][iBndGP] <= fullUpdateTime &&
          DS[iBndGP] &&
          mu[face][iBndGP] <= ( Mu_w+0.05*(RS_f0-Mu_w) ) ) {
        dynStress_time[face][iBndGP] = fullUpdateTime;
        DS[face][iBndGP] = false;
      }
    }
  }

  //TODO: move this to Base class with the attributes
  //---compute and store slip to determine the magnitude of an earthquake ---
  //    to this end, here the slip is computed and averaged per element
  //    in calc_seissol.f90 this value will be multiplied by the element surface
  //    and an output happened once at the end of the simulation
  void calcAverageSlip(
      std::array<real, numOfPointsPadded> &tmpSlip,
      unsigned int face
  ){
    real sum_tmpSlip = 0;
    if (m_Params.IsMagnitudeOutputOn) {
      for (int iBndGP = 0; iBndGP < numOfPointsPadded; iBndGP++)
        sum_tmpSlip += tmpSlip[iBndGP];
      averaged_Slip[face] = averaged_Slip[face] + sum_tmpSlip / numberOfPoints;
    }
  }


public:
  virtual void evaluate(seissol::initializers::Layer&  layerData,
                        seissol::initializers::DynamicRupture *dynRup,
                        real (*QInterpolatedPlus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
                        real (*QInterpolatedMinus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
                        real fullUpdateTime,
                        real timeWeights[CONVERGENCE_ORDER],
                        real DeltaT[CONVERGENCE_ORDER]) override {


    //***********************************
    // GET THESE FROM DATA STRUCT
    /*
    //required input for thermal pressure:
    int TP_grid_nz; //DISC%dynRup%TP_grid_nz;  !< number of grid points to solve advection for TP in z-direction
    double TP[nBndGP][nFace][unkown];     //DISC%DynRup%TP(:,iFace,2)    !< Temperature and Pressure for TP along each fault point
    double TP_Theta[nBndGP][nFace][TP_grid_nz]; //DISC%DynRup%TP_Theta(iBndGP, iFace,:) !< Fourier transformed pressure
    double TP_sigma[nBndGP][nFace][TP_grid_nz]; //DISC%DynRup%TP_sigma(iBndGP, iFace,:) !< Fourier transformed temperature
    double TP_half_width_shear_zone[nBndGP][nFace];  //DISC%DynRup%TP_half_width_shear_zone(iBndGP,iFace)    !< spatial dependent half width of the shearing layer for TP
    double alpha_th;        //DISC%DynRup%alpha_th   !< thermal diffusion parameter for TP
    double alpha_hy[nBndGP][nFace];    //DISC%DynRup%alpha_hy(iBndGP,iFace)    !< spatial dependent hydraulic diffusion parameter for TP
    double rho_c;        //DISC%DynRup%rho_c !< heat capacity for TP
    double TP_Lambda;          // DISC%DynRup%TP_Lambda    !< pore pressure increase per unit increase
    double TP_grid[TP_grid_nz];    //DISC%DynRup%TP_grid   !< grid for TP
    double TP_DFinv[TP_grid_nz]; //DISC%DynRup%TP_DFinv  !< inverse Fourier coefficients
    double temp_0;          //EQN%Temp_0     !< Initial temperature for TP
    double pressure_0;            //EQN%Pressure_0           !< Initial pressure for TP
*/
    //***********************************
    //double S[nBndGP];
    //double Theta_tmp[TP_grid_nz], Sigma_tmp[TP_grid_nz];


    //--------------------------------------------------------
    int ThermalPress = 0; //DISC%DynRup%ThermalPress   !< thermal pressurization switch

    //first copy all Variables from the Base Lts dynRup tree
    BaseFrictionSolver::copyLtsTreeToLocal(layerData, dynRup);
    seissol::initializers::DR_FL_103 *ConcreteLts = dynamic_cast<seissol::initializers::DR_FL_103 *>(dynRup);
    nucleationStressInFaultCS =  layerData.var(ConcreteLts->nucleationStressInFaultCS); ;
//  magnitude_out                                 = layerData.var(ConcreteLts->magnitude_out);

//TODO: replace all local values with m_Params access
    RS_f0  = m_Params.rs_f0;
    RS_b  = m_Params.rs_b;
    RS_sr0  = m_Params.rs_sr0;
    Mu_w = m_Params.mu_w;
    RS_sl0_array  = layerData.var(ConcreteLts->RS_sl0_array);
    RS_a_array  = layerData.var(ConcreteLts->RS_a_array);
    RS_srW_array  = layerData.var(ConcreteLts->RS_srW_array);
    DS                       = layerData.var(ConcreteLts->DS);
    averaged_Slip                                 = layerData.var(ConcreteLts->averaged_Slip);
    stateVar           = layerData.var(ConcreteLts->stateVar);
    dynStress_time           = layerData.var(ConcreteLts->dynStress_time);


    //initialize local variables
    double dt = 0;
    double Gnuc = 0;
    double invZ = 0;
    bool has_converged = false;
    double matmul = 0;
    std::array<real, numOfPointsPadded> ShTest{0};
    std::array<real, numOfPointsPadded> P{0};
    std::array<real, numOfPointsPadded> P_f{0};

    std::array<real, numOfPointsPadded> SV0{0};
    //TODO: rename LocSlipRate:
    real LocSR[seissol::tensor::resamplePar::size()];
    //std::array<real, numOfPointsPadded> LocSR{0};
    std::array<real, numOfPointsPadded> LocSR1{0};
    std::array<real, numOfPointsPadded> LocSR2{0};
    std::array<real, numOfPointsPadded> SR_tmp{0};

    std::array<real, numOfPointsPadded> LocMu{0};
    std::array<real, numOfPointsPadded> LocSlip{0};
    std::array<real, numOfPointsPadded> LocSlip1{0};
    std::array<real, numOfPointsPadded> LocSlip2{0};
    std::array<real, numOfPointsPadded> LocSV{0};

    std::array<real, numOfPointsPadded> n_stress{0};
    std::array<real, numOfPointsPadded> SRtest{0};


    std::array<real, numOfPointsPadded> tmp{0};
    std::array<real, numOfPointsPadded> tmp2{0};

    std::array<real, numOfPointsPadded> LocTracXY{0};
    std::array<real, numOfPointsPadded> LocTracXZ{0};


    // switch for Gauss node wise stress assignment
    bool nodewise; //= true;    //TODO: configureable? not used in this FL

    //Apply time dependent nucleation at global time step not sub time steps for simplicity
    //initialize time and space dependent nucleation
    double Tnuc = m_Params.t_0;

    //!TU 7.07.16: if the SR is too close to zero, we will have problems (NaN)
    //!as a consequence, the SR is affected the AlmostZero value when too small
    double AlmostZero = 1e-45; //d-45;

    //!PARAMETERS of THE optimisation loops
    //!absolute tolerance on the function to be optimzed
    //! This value is quite arbitrary (a bit bigger as the expected numerical error) and may not be the most adapted
    //! Number of iteration in the loops
    unsigned int nSRupdates = 60;
    unsigned int nSVupdates = 2;


#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (unsigned face = 0; face < layerData.getNumberOfCells(); ++face) {
      FaultStresses faultStresses;

      //declare local variables
      std::array<real, numOfPointsPadded> tmpSlip{0};
      //               std::array<real, numOfPointsPadded> LocSlipRate;

      precomputeStressFromQInterpolated(faultStresses, QInterpolatedPlus[face], QInterpolatedMinus[face],  face);

      //TODO: outside face loop
      dt = 0;
      for (int iTimeGP = 0; iTimeGP < CONVERGENCE_ORDER; iTimeGP++) {
        dt += DeltaT[iTimeGP];
      }

      if (fullUpdateTime <= Tnuc) {
        Gnuc = Calc_SmoothStepIncrement(fullUpdateTime, Tnuc, dt) ;
        //TODO: test padded
        for (int iBndGP = 0; iBndGP < numberOfPoints; iBndGP++) {
          for (int i = 0; i < 6; i++) {
            initialStressInFaultCS[face][iBndGP][i] = initialStressInFaultCS[face][iBndGP][i] + nucleationStressInFaultCS[face][iBndGP][i] * Gnuc;
          }
        }

      } //end If-Tnuc

      //debugging
      real  (*TestInitialStress)[numOfPointsPadded][6] = &initialStressInFaultCS[face];
      real  (*TestNucleationStress)[numOfPointsPadded][6] = &nucleationStressInFaultCS[face];

      for (int iBndGP = 0; iBndGP < numberOfPoints; iBndGP++) {
        //TODO: remove local values as much as possible
        LocMu[iBndGP] = mu[face][iBndGP];     // Current friction coefficient at given fault node
        LocSlip[iBndGP] = slip[face][iBndGP]; //DISC%DynRup%Slip(iBndGP,iFace)              //!< Slip path at given fault node
        LocSlip1[iBndGP] = slip1[face][iBndGP]; //DISC%DynRup%Slip1(iBndGP,iFace)            //!< Slip at given fault node along loc dir 1
        LocSlip2[iBndGP] = slip2[face][iBndGP]; //DISC%DynRup%Slip2(iBndGP,iFace)            // !< Slip at given fault node along loc dir 2
        LocSR1[iBndGP] = slipRate1[face][iBndGP]; //DISC%DynRup%SlipRate1(iBndGP,iFace)         // !< Slip Rate at given fault node
        LocSR2[iBndGP] = slipRate2[face][iBndGP]; //DISC%DynRup%SlipRate2(iBndGP,iFace)         // !< Slip Rate at given fault node
        LocSV[iBndGP] = stateVar[face][iBndGP];     //DISC%DynRup%StateVar(iBndGP,iFace)
      }

      for (int iTimeGP = 0; iTimeGP < CONVERGENCE_ORDER; iTimeGP++) {

        //TODO: outside time loop
        invZ = 1.0 / impAndEta[face].Zs + 1.0 / impAndEta[face].Zs_neig;

        //TODO: test padded:
        for (int iBndGP = 0; iBndGP < numberOfPoints; iBndGP++) {

          // friction develops as                    mu = a * arcsinh[ V/(2*V0) * exp(SV/a) ]
          // state variable SV develops as        dSV / dt = -(V - L) * (SV - SV_ss)
          //                                      SV_ss = a * ln[ 2*V0/V * sinh(mu_ss/a) ]
          //                                      mu_ss = mu_w + [mu_lv - mu_w] / [ 1 + (V/Vw)^8 ] ^ (1/8) ]
          //                                      mu_lv = mu_0 - (b-a) ln (V/V0)

          // load traction and normal stress
          P[iBndGP] = faultStresses.NorStressGP[iTimeGP][iBndGP] + initialStressInFaultCS[face][iBndGP][0];
          //TODO:rename ShTest
          ShTest[iBndGP] = std::sqrt(
              seissol::dr::aux::power(initialStressInFaultCS[face][iBndGP][3] + faultStresses.XYStressGP[iTimeGP][iBndGP], 2) +
              seissol::dr::aux::power(initialStressInFaultCS[face][iBndGP][5] + faultStresses.XZStressGP[iTimeGP][iBndGP], 2));

          // We use the regularized rate-and-state friction, after Rice & Ben-Zion (1996) //TODO: look up
          // ( Numerical note: ASINH(X)=LOG(X+SQRT(X^2+1)) )

          SV0[iBndGP] = LocSV[iBndGP];    // Careful, the SV must always be corrected using SV0 and not LocSV!

          // The following process is adapted from that described by Kaneko et al. (2008)
          LocSR[iBndGP] = std::sqrt(seissol::dr::aux::power(LocSR1[iBndGP], 2) + seissol::dr::aux::power(LocSR2[iBndGP], 2) );
          LocSR[iBndGP] = std::max(AlmostZero, LocSR[iBndGP]);

          SR_tmp[iBndGP] = LocSR[iBndGP];


          if (ThermalPress == 1) {
            //P_f[iBndGP] = TP[iBndGP][iFace][1];
          } else {
            P_f[iBndGP] = 0.0;
          }
        }// End of iBndGP-loop

        for (int j = 0; j < nSVupdates; j++) {
          //TODO: test for padded:
          for (int iBndGP = 0; iBndGP < numberOfPoints; iBndGP++) {
            //fault strength using LocMu and P_f from previous timestep/iteration
            //1.update SV using Vold from the previous time step
            updateStateVariable(iBndGP, face, SV0[iBndGP], DeltaT[iTimeGP], SR_tmp[iBndGP], LocSV[iBndGP]);
            /*
            if(face == 9 && iTimeGP == 0 && iBndGP == 0){
              std::cout.precision(17);
              std::cout << "C++ change in StateVar[face=9,iTimeGP=0,iBndGP=0] 1: " << std::fixed << LocSV[0] << std::endl;
            }
            //*/
            if (ThermalPress == 1) {
              /*
              S[iBndGP] = -LocMu[iBndGP] * (P[iBndGP] - P_f[iBndGP]);

              for (int iTP_grid_nz = 0; iTP_grid_nz < TP_grid_nz; iTP_grid_nz++) {
                //!recover original values as it gets overwritten in the ThermalPressure routine
                Theta_tmp[iTP_grid_nz] = TP_Theta[iBndGP][iFace][iTP_grid_nz];
                Sigma_tmp[iTP_grid_nz] = TP_sigma[iBndGP][iFace][iTP_grid_nz];
              }
              Calc_ThermalPressure(temp_0, pressure_0, time_inc, TP_grid_nz, TP_half_width_shear_zone[iBndGP][iFace],
                                   alpha_th, alpha_hy[iBndGP][iFace], rho_c, TP_Lambda, Theta_tmp, Sigma_tmp, S[iBndGP],
                                   LocSR[iBndGP], TP_grid,
                                   TP_DFinv, TP[iBndGP][iFace][0], TP[iBndGP][iFace][1]);
              P_f[iBndGP] = TP[iBndGP][iFace][1];
               */
            }
            //2. solve for Vnew , applying the Newton-Raphson algorithm
            //effective normal stress including initial stresses and pore fluid pressure
            n_stress[iBndGP] = P[iBndGP] - P_f[iBndGP];

          }// End of iBndGP-loop

          has_converged = IterativelyInvertSR(face, nSRupdates, LocSR, LocSV, n_stress, ShTest, invZ, SRtest);

          //TODO: test padded
          for (int iBndGP = 0; iBndGP < numberOfPoints; iBndGP++) {

            // 3. update theta, now using V=(Vnew+Vold)/2
            // For the next SV update, use the mean slip rate between the initial guess and the one found (Kaneko 2008, step 6)
            SR_tmp[iBndGP] = 0.5 * (LocSR[iBndGP] + fabs(SRtest[iBndGP]));

            // 4. solve again for Vnew
            LocSR[iBndGP] = fabs(SRtest[iBndGP]);
            //!update LocMu
            tmp[iBndGP] = 0.5 / RS_sr0 * exp(LocSV[iBndGP] / RS_a_array[face][iBndGP]);
            tmp2[iBndGP] = LocSR[iBndGP] * tmp[iBndGP];
            // mu from LocSR
            LocMu[iBndGP] = RS_a_array[face][iBndGP] * log(tmp2[iBndGP] + sqrt(seissol::dr::aux::power(tmp2[iBndGP], 2) + 1.0));
          }// End of iBndGP-loop

        } //End nSVupdates-loop   j=1,nSVupdates   !This loop corrects SV values


        if (!has_converged) {
          //!logError(*) 'nonConvergence RS Newton', time
          //TODO: error logging : logError(*) 'NaN detected', time
          //std::cout << "nonConvergence RS Newton" << std::endl;
          assert( !std::isnan(tmp[0]) && "nonConvergence RS Newton");
        }

        for (int iBndGP = 0; iBndGP < numberOfPoints; iBndGP++) {
          updateStateVariable(iBndGP, face, SV0[iBndGP], DeltaT[iTimeGP], SR_tmp[iBndGP], LocSV[iBndGP]);

          //! 5. get final theta, mu, traction and slip
          //! SV from mean slip rate in tmp

          if (ThermalPress == 1) {
            /*
            S[iBndGP] = -LocMu[iBndGP] * (P[iBndGP] - P_f[iBndGP]);

            for (int iTP_grid_nz = 0; iTP_grid_nz < TP_grid_nz; iTP_grid_nz++) {
              Theta_tmp[iTP_grid_nz] = TP_Theta[iBndGP][iFace][iTP_grid_nz];
              Sigma_tmp[iTP_grid_nz] = TP_sigma[iBndGP][iFace][iTP_grid_nz];
              //!use Theta/Sigma from last call in this update, dt/2 and new SR from NS

              Calc_ThermalPressure(temp_0, pressure_0, time_inc, TP_grid_nz, TP_half_width_shear_zone[iBndGP][iFace],
                                   alpha_th, alpha_hy[iBndGP][iFace], rho_c, TP_Lambda, Theta_tmp, Sigma_tmp, S[iBndGP],
                                   LocSR[iBndGP], TP_grid,
                                   TP_DFinv, TP[iBndGP][iFace][0], TP[iBndGP][iFace][1]);

              P_f[iBndGP] = TP[iBndGP][iFace][1];
              TP_Theta[iBndGP][iFace][iTP_grid_nz] = Theta_tmp[iTP_grid_nz];
              TP_sigma[iBndGP][iFace][iTP_grid_nz] = Sigma_tmp[iTP_grid_nz];
            }
            */
          }

          //!update LocMu for next strength determination, only needed for last update
          //! X in Asinh(x) for mu calculation
          tmp[iBndGP] = 0.5 / RS_sr0 * exp(LocSV[iBndGP] / RS_a_array[face][iBndGP]);
          tmp2[iBndGP] = LocSR[iBndGP] * tmp[iBndGP];
          //! mu from LocSR
          LocMu[iBndGP] = RS_a_array[face][iBndGP] * log(tmp2[iBndGP] + sqrt(seissol::dr::aux::power(tmp2[iBndGP], 2) + 1.0));

          //debugging:
          /*
          if(face == 9 && iTimeGP == 0 && iBndGP == 0){
            std::cout.precision(17);
            std::cout << "C++ change in tmp[face=9,iTimeGP=0,iBndGP=0] 1: " << std::scientific << tmp[0] << std::endl;
            std::cout << "C++ change in tmp2[face=9,iTimeGP=0,iBndGP=0] 1: " << std::scientific << tmp2[0] << std::endl;
            std::cout << "C++ change in LocMu[face=9,iTimeGP=0,iBndGP=0] 1: " << std::scientific << LocMu[0] << std::endl;
          }
          //*/
          real LocTracXY_inbetween;


          //! update stress change
          LocTracXY_inbetween = -((initialStressInFaultCS[face][iBndGP][3] + faultStresses.XYStressGP[iTimeGP][iBndGP]) / ShTest[iBndGP]) * LocMu[iBndGP] * (P[iBndGP] - P_f[iBndGP]);
          LocTracXZ[iBndGP] = -((initialStressInFaultCS[face][iBndGP][5] + faultStresses.XZStressGP[iTimeGP][iBndGP]) / ShTest[iBndGP]) * LocMu[iBndGP] * (P[iBndGP] - P_f[iBndGP]);
          LocTracXY[iBndGP] = LocTracXY_inbetween - initialStressInFaultCS[face][iBndGP][3];
          LocTracXZ[iBndGP] = LocTracXZ[iBndGP] - initialStressInFaultCS[face][iBndGP][5];
          //testing alternative calculation:
          //LocTracXY[iBndGP] = (initialStressInFaultCS[face][iBndGP][3] * ( LocMu[iBndGP] * (P[iBndGP] - P_f[iBndGP]) - ShTest[iBndGP]) +  XYStressGP[iTimeGP][iBndGP] *  LocMu[iBndGP] * (P[iBndGP] - P_f[iBndGP]) ) / ShTest[iBndGP];

          //Compute slip
          //! ABS of LocSR removed as it would be the accumulated slip that is usually not needed in the solver, see linear slip weakening
          LocSlip[iBndGP] = LocSlip[iBndGP] + (LocSR[iBndGP]) * DeltaT[iTimeGP];

          //!Update slip rate (notice that LocSR(T=0)=-2c_s/mu*s_xy^{Godunov} is the slip rate caused by a free surface!)
          LocSR1[iBndGP] = -invZ * (LocTracXY[iBndGP] - faultStresses.XYStressGP[iTimeGP][iBndGP]);
          LocSR2[iBndGP] = -invZ * (LocTracXZ[iBndGP] - faultStresses.XZStressGP[iTimeGP][iBndGP]);

          //!TU 07.07.16: correct LocSR1_2 to avoid numerical errors
          tmp[iBndGP] = sqrt(pow(LocSR1[iBndGP], 2) + pow(LocSR2[iBndGP], 2));
          if (tmp[iBndGP] != 0) {
            LocSR1[iBndGP] = LocSR[iBndGP] * LocSR1[iBndGP] / tmp[iBndGP];
            LocSR2[iBndGP] = LocSR[iBndGP] * LocSR2[iBndGP] / tmp[iBndGP];
          }

          tmpSlip[iBndGP] = tmpSlip[iBndGP] + tmp[iBndGP] * DeltaT[iTimeGP];

          LocSlip1[iBndGP] = LocSlip1[iBndGP] + (LocSR1[iBndGP]) * DeltaT[iTimeGP];
          LocSlip2[iBndGP] = LocSlip2[iBndGP] + (LocSR2[iBndGP]) * DeltaT[iTimeGP];
          //LocSR1     = SignSR1*ABS(LocSR1)
          //LocSR2     = SignSR2*ABS(LocSR2)

          //!Save traction for flux computation
          faultStresses.TractionGP_XY[iTimeGP][iBndGP] = LocTracXY[iBndGP];
          faultStresses.TractionGP_XZ[iTimeGP][iBndGP] = LocTracXZ[iBndGP];
/*
          if(face == 9 && iBndGP == 0){
            std::cout.precision(17);
            std::cout << "C++ change in LocTracXZ[face=9,iTimeGP=" << iTimeGP << ",iBndGP=0] 1: " << std::scientific << LocTracXZ[0] << std::endl;
          }
          //*/
        }

      } // End of iTimeGP-loop

      //TODO: test padded
      for (int iBndGP = 0; iBndGP < numberOfPoints; iBndGP++) {

        //TODO: dont use these local variables if possible:
        mu[face][iBndGP] = LocMu[iBndGP];
        slipRate1[face][iBndGP] = LocSR1[iBndGP];
        slipRate2[face][iBndGP] = LocSR2[iBndGP];
        slip[face][iBndGP] = LocSlip[iBndGP];
        slip1[face][iBndGP] = LocSlip1[iBndGP];
        slip2[face][iBndGP] = LocSlip2[iBndGP];
        tracXY[face][iBndGP] = LocTracXY[iBndGP];
        tracXZ[face][iBndGP] = LocTracXZ[iBndGP];

        matmul = 0.0;
        for (int j = 0; j < numberOfPoints; j++) {
          matmul += resampleMatrixView(iBndGP, j) * (LocSV[j] - stateVar[face][j]);
        }
        stateVar[face][iBndGP] = stateVar[face][iBndGP] + matmul;
      }

      // output rupture front
      // outside of iTimeGP loop in order to safe an 'if' in a loop
      // this way, no subtimestep resolution possible
      outputRuptureFront(LocSR,fullUpdateTime, face);

      calcPeakSlipRate(LocSR, face);

      //output time when shear stress is equal to the dynamic stress after rupture arrived
      //currently only for linear slip weakening
      outputDynamicStress(fullUpdateTime, face);

      //---compute and store slip to determine the magnitude of an earthquake ---
      //    to this end, here the slip is computed and averaged per element
      //    in calc_seissol.f90 this value will be multiplied by the element surface
      //    and an output happened once at the end of the simulation
      calcAverageSlip(tmpSlip, face);

      postcomputeImposedStateFromNewStress(
          QInterpolatedPlus[face], QInterpolatedMinus[face],
          faultStresses, timeWeights, face);

      /*
      //debugging
      for(int i = 0; i < tensor::QInterpolated::size(); i++){
        assert( !std::isnan(imposedStatePlus[face][i]) );
      }
      //*/

    }//end face loop
  }//end evaluate function
}; //end class Init_FL_103

#endif //SEISSOL_DR_SOLVER_RATE_AND_STATE_H