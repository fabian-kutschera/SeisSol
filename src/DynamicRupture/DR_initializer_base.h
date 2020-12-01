//
// Created by adrian on 09.07.20.
//

#ifndef SEISSOL_DR_INITIALIZER_BASE_H
#define SEISSOL_DR_INITIALIZER_BASE_H


#include <Solver/Interoperability.h>
#include <yaml-cpp/yaml.h>
#include "Initializer/InputAux.hpp"
#include "DR_solver_base.h"
#include "DR_solver_rate_and_state.h"
#include <DynamicRupture/DR_Parameters.h>




namespace seissol {
    namespace initializers {
      class BaseDrInitializer;            //general parameters initialized that are required by all friction laws
      class Init_NoFaultFL0;              //No SolverNoFaultFL0
      class Init_LinearSlipWeakeningFL2;  //general initialization for linear slip laws
      class Init_RateAndStateFL3;         //aging law (need revisit)
      class Init_LinearBimaterialFL6;     //for bimaterial faults
      class Init_LinearSlipWeakeningFL16; //FL2 extended by forced rupture time
      class Init_ImposedSlipRatesFL33;    //imposed slip rates on bounday
      class Init_RateAndStateFL103;       //rate and state with time and space dependent nucleation parameter
      class Init_RateAndStateFL103TP;     //Fl103 extended with thermal pressurization
    }
}
/*
 * initial values are obtain
 * from m_Params (direct in read from parameter.par file)
 * from std::unordered_map<std::string, double*> faultParameters -> allocated matrices in Fortran with C++ pointer from Easi inread.
 * from interoperabilty functions that access Fortran values and copy them in C++ memory (copy by value) (e.g. e_interoperability.getDynRupParameters())
 */
class seissol::initializers::BaseDrInitializer {
protected:
  static constexpr int numberOfPoints = tensor::QInterpolated::Shape[0];
  static constexpr int numOfPointsPadded = init::QInterpolated::Stop[0];
  //YAML::Node m_InputParam;
  dr::DrParameterT *m_Params;

public:
  virtual ~BaseDrInitializer() {}

  //set the parameters from .par file with yaml to this class attributes.
  void setInputParam(dr::DrParameterT *DynRupParameter) {
    m_Params = DynRupParameter;
  }

  virtual void initializeFrictionMatrices(seissol::initializers::DynamicRupture *dynRup,
        initializers::LTSTree* dynRupTree,
        seissol::dr::fr_law::BaseFrictionSolver* FrictionSolver,
        std::unordered_map<std::string, double*> faultParameters,
        unsigned* ltsFaceToMeshFace,
        seissol::Interoperability &e_interoperability) {
    unsigned* layerLtsFaceToMeshFace = ltsFaceToMeshFace;

    for (initializers::LTSTree::leaf_iterator it = dynRupTree->beginLeaf(initializers::LayerMask(Ghost)); it != dynRupTree->endLeaf(); ++it) {
      real  (*iniBulkXX)[numOfPointsPadded]                = it->var(dynRup->iniBulkXX);                //get from faultParameters
      real  (*iniBulkYY)[numOfPointsPadded]                = it->var(dynRup->iniBulkYY);                //get from faultParameters
      real  (*iniBulkZZ)[numOfPointsPadded]                = it->var(dynRup->iniBulkZZ);                //get from faultParameters
      real  (*iniShearXY)[numOfPointsPadded]                = it->var(dynRup->iniShearXY);                //get from faultParameters
      real  (*iniShearXZ)[numOfPointsPadded]                = it->var(dynRup->iniShearXZ);                //get from faultParameters
      real  (*iniShearYZ)[numOfPointsPadded]                = it->var(dynRup->iniShearYZ);                //get from faultParameters
      real  (*initialStressInFaultCS)[numOfPointsPadded][6] = it->var(dynRup->initialStressInFaultCS);  //get from fortran  EQN%InitialStressInFaultCS
      real  (*cohesion)[numOfPointsPadded]                  = it->var(dynRup->cohesion);                //get from faultParameters
      real  (*mu)[ numOfPointsPadded ]            = it->var(dynRup->mu);                                //get from fortran  EQN%IniMu(:,:)
      real  (*slip)[ numOfPointsPadded ]          = it->var(dynRup->slip);                              // = 0
      real  (*slipStrike)[numOfPointsPadded ]          = it->var(dynRup->slipStrike);                            // = 0
      real  (*slipDip)[ numOfPointsPadded ]         = it->var(dynRup->slipDip);                             // = 0
      real  (*slipRateMagnitude)[ numOfPointsPadded ]     = it->var(dynRup->slipRateMagnitude);                     // = 0
      real  (*slipRateStrike)[ numOfPointsPadded ]     = it->var(dynRup->slipRateStrike);                         //get from fortran  EQN%IniSlipRate1
      real  (*slipRateDip)[numOfPointsPadded ]      = it->var(dynRup->slipRateDip);                         //get from fortran  EQN%IniSlipRate2
      real  (*rupture_time)[ numOfPointsPadded ]  = it->var(dynRup->rupture_time);                      // = 0
      bool  (*RF)[ numOfPointsPadded ]            = it->var(dynRup->RF);                                //get from fortran
      real  (*peakSR)[ numOfPointsPadded ]        = it->var(dynRup->peakSR);                            // = 0
      real  (*tractionXY)[ numOfPointsPadded ]        = it->var(dynRup->tractionXY);                            // = 0
      real  (*tractionXZ)[ numOfPointsPadded ]        = it->var(dynRup->tractionXZ);                            // = 0

      dynRup->IsFaultParameterizedByTraction = false;

      for (unsigned ltsFace = 0; ltsFace < it->getNumberOfCells(); ++ltsFace) {
        unsigned meshFace = layerLtsFaceToMeshFace[ltsFace];
        for (unsigned iBndGP = 0; iBndGP < init::QInterpolated::Stop[0]; ++iBndGP) {    //loop includes padded elements
          slip[ltsFace][iBndGP] = 0.0;
          slipStrike[ltsFace][iBndGP] = 0.0;
          slipDip[ltsFace][iBndGP] = 0.0;
          slipRateMagnitude[ltsFace][iBndGP] = 0.0;
          rupture_time[ltsFace][iBndGP] = 0.0;
          peakSR[ltsFace][iBndGP] = 0.0;
          tractionXY[ltsFace][iBndGP] = 0.0;
          tractionXZ[ltsFace][iBndGP] = 0.0;
        }
        //get initial values from fortran
        for (unsigned iBndGP = 0; iBndGP < numberOfPoints; ++iBndGP) {
          if(faultParameters["T_n"] != NULL ){
            iniBulkXX[ltsFace][iBndGP] = static_cast<real>( faultParameters["T_n"][(meshFace) * numberOfPoints + iBndGP] );
            dynRup->IsFaultParameterizedByTraction = true;
          }
          else if(faultParameters["s_xx"] != NULL )
            iniBulkXX[ltsFace][iBndGP] = static_cast<real>( faultParameters["s_xx"][(meshFace) * numberOfPoints + iBndGP] );
          else
            iniBulkXX[ltsFace][iBndGP] = 0.0;

          if(faultParameters["s_yy"] != NULL )
            iniBulkYY[ltsFace][iBndGP] = static_cast<real>( faultParameters["s_yy"][(meshFace) * numberOfPoints + iBndGP] );
          else
            iniBulkYY[ltsFace][iBndGP] = 0.0;

          if(faultParameters["s_zz"] != NULL )
            iniBulkZZ[ltsFace][iBndGP] = static_cast<real>( faultParameters["s_zz"][(meshFace) * numberOfPoints + iBndGP] );
          else
            iniBulkZZ[ltsFace][iBndGP] = 0.0;

          if(faultParameters["T_s"] != NULL )
            iniShearXY[ltsFace][iBndGP] = static_cast<real>( faultParameters["T_s"][(meshFace) * numberOfPoints + iBndGP] );
          else if(faultParameters["s_xy"] != NULL )
            iniShearXY[ltsFace][iBndGP] = static_cast<real>( faultParameters["s_xy"][(meshFace) * numberOfPoints + iBndGP] );
          else
            iniShearXY[ltsFace][iBndGP] = 0.0;

          if(faultParameters["T_d"] != NULL )
            iniShearXZ[ltsFace][iBndGP] = static_cast<real>( faultParameters["T_d"][(meshFace) * numberOfPoints + iBndGP] );
          else if(faultParameters["s_xz"] != NULL )
            iniShearXZ[ltsFace][iBndGP] = static_cast<real>( faultParameters["s_xz"][(meshFace) * numberOfPoints + iBndGP] );
          else
            iniShearXZ[ltsFace][iBndGP] = 0.0;

          if(faultParameters["s_yz"] != NULL )
            iniShearYZ[ltsFace][iBndGP] = static_cast<real>( faultParameters["s_yz"][(meshFace) * numberOfPoints + iBndGP] );
          else
            iniShearYZ[ltsFace][iBndGP] = 0.0;

          if(faultParameters["cohesion"] != NULL ){
            cohesion[ltsFace][iBndGP] = static_cast<real>( faultParameters["cohesion"][(meshFace) * numberOfPoints + iBndGP] );
          }else{
            cohesion[ltsFace][iBndGP] = 0.0;
          }
        }
        //initialize padded elements for vectorization
        for (unsigned iBndGP = numberOfPoints; iBndGP < numOfPointsPadded; ++iBndGP) {
          cohesion[ltsFace][iBndGP]               = 0.0;
        }
        e_interoperability.getDynRupParameters(ltsFace, meshFace, initialStressInFaultCS, mu, slipRateStrike, slipRateDip, RF);

      }//lts-face loop
      layerLtsFaceToMeshFace += it->getNumberOfCells();
    }//leaf_iterator loop
  }

};

/*
 * only base initialization
 */
class seissol::initializers::Init_NoFaultFL0: public seissol::initializers::BaseDrInitializer {
public:
  virtual void initializeFrictionMatrices(seissol::initializers::DynamicRupture *dynRup,
                                          initializers::LTSTree* dynRupTree,
                                          seissol::dr::fr_law::BaseFrictionSolver* FrictionSolver,
                                          std::unordered_map<std::string,
                                              double*> faultParameters,
                                          unsigned* ltsFaceToMeshFace,
                                          seissol::Interoperability &e_interoperability) override {
    BaseDrInitializer::initializeFrictionMatrices(dynRup, dynRupTree, FrictionSolver, faultParameters, ltsFaceToMeshFace, e_interoperability);
  }
};

/*
 * additional parameters for linear slip weakening
 */
class seissol::initializers::Init_LinearSlipWeakeningFL2 : public seissol::initializers::BaseDrInitializer {
public:
  virtual void initializeFrictionMatrices(seissol::initializers::DynamicRupture *dynRup,
        initializers::LTSTree* dynRupTree,
        seissol::dr::fr_law::BaseFrictionSolver* FrictionSolver,
        std::unordered_map<std::string,
        double*> faultParameters,
        unsigned* ltsFaceToMeshFace,
        seissol::Interoperability &e_interoperability) override {
    BaseDrInitializer::initializeFrictionMatrices(dynRup, dynRupTree, FrictionSolver, faultParameters, ltsFaceToMeshFace, e_interoperability);
    seissol::initializers::LTS_LinearSlipWeakeningFL2 *ConcreteLts = dynamic_cast<seissol::initializers::LTS_LinearSlipWeakeningFL2 *>(dynRup);

    unsigned* layerLtsFaceToMeshFace = ltsFaceToMeshFace;

    for (initializers::LTSTree::leaf_iterator it = dynRupTree->beginLeaf(initializers::LayerMask(Ghost)); it != dynRupTree->endLeaf(); ++it) {
      real (*d_c)[numOfPointsPadded]                       = it->var(ConcreteLts->d_c);                 //from faultParameters
      real (*mu_S)[numOfPointsPadded]                      = it->var(ConcreteLts->mu_S);                //from faultParameters
      real (*mu_D)[numOfPointsPadded]                      = it->var(ConcreteLts->mu_D);                //from faultParameters
      bool (*DS)[numOfPointsPadded]                        = it->var(ConcreteLts->DS);                  //from parameter file
      real *averaged_Slip                                  = it->var(ConcreteLts->averaged_Slip);       // = 0
      real (*dynStress_time)[numOfPointsPadded]            = it->var(ConcreteLts->dynStress_time);      // = 0


      for (unsigned ltsFace = 0; ltsFace < it->getNumberOfCells(); ++ltsFace) {
        unsigned meshFace = layerLtsFaceToMeshFace[ltsFace];
        //initialize padded elements for vectorization
        for (unsigned iBndGP = 0; iBndGP < numOfPointsPadded; ++iBndGP) {
          d_c[ltsFace][iBndGP]                    = 0.0;
          mu_S[ltsFace][iBndGP]                   = 0.0;
          mu_D[ltsFace][iBndGP]                   = 0.0;
        }
        for (unsigned iBndGP = 0; iBndGP < numberOfPoints; ++iBndGP) {
          d_c[ltsFace][iBndGP]                    = static_cast<real>( faultParameters["d_c"][(meshFace) * numberOfPoints+ iBndGP] );
          mu_S[ltsFace][iBndGP]                   = static_cast<real>( faultParameters["mu_s"][(meshFace) * numberOfPoints+ iBndGP] );
          mu_D[ltsFace][iBndGP]                   = static_cast<real>( faultParameters["mu_d"][(meshFace) * numberOfPoints+ iBndGP] );
        }
        averaged_Slip[ltsFace]= 0.0;
        for (unsigned iBndGP = 0; iBndGP < numOfPointsPadded; ++iBndGP) {    //loop includes padded elements
          dynStress_time[ltsFace][iBndGP] = 0.0;
          DS[ltsFace][iBndGP] = m_Params->IsDsOutputOn;
        }
      }//lts-face loop
      layerLtsFaceToMeshFace += it->getNumberOfCells();
    }//leaf_iterator loop
  }
};

/*
 * forced rupture times initialized
 */
class seissol::initializers::Init_LinearSlipWeakeningFL16 : public seissol::initializers::Init_LinearSlipWeakeningFL2 {
public:
  virtual void initializeFrictionMatrices(seissol::initializers::DynamicRupture *dynRup,
                                          initializers::LTSTree* dynRupTree,
                                          seissol::dr::fr_law::BaseFrictionSolver* FrictionSolver,
                                          std::unordered_map<std::string, double*> faultParameters,
                                          unsigned* ltsFaceToMeshFace,
                                          seissol::Interoperability &e_interoperability) override {
    Init_LinearSlipWeakeningFL2::initializeFrictionMatrices(dynRup, dynRupTree, FrictionSolver, faultParameters, ltsFaceToMeshFace, e_interoperability);
    seissol::initializers::LTS_LinearSlipWeakeningFL16 *ConcreteLts = dynamic_cast<seissol::initializers::LTS_LinearSlipWeakeningFL16 *>(dynRup);

    unsigned* layerLtsFaceToMeshFace = ltsFaceToMeshFace;

    for (initializers::LTSTree::leaf_iterator it = dynRupTree->beginLeaf(initializers::LayerMask(Ghost)); it != dynRupTree->endLeaf(); ++it) {

      real (*forced_rupture_time)[numOfPointsPadded]       = it->var(ConcreteLts->forced_rupture_time); //from faultParameters
      real *tn                                             = it->var(ConcreteLts->tn);                  // = 0

      for (unsigned ltsFace = 0; ltsFace < it->getNumberOfCells(); ++ltsFace) {
        unsigned meshFace = layerLtsFaceToMeshFace[ltsFace];

        //initialize padded elements for vectorization
        for (unsigned iBndGP = 0; iBndGP < numOfPointsPadded; ++iBndGP) {
          forced_rupture_time[ltsFace][iBndGP]    = 0.0;
        }
        for (unsigned iBndGP = 0; iBndGP < numberOfPoints; ++iBndGP) {
          if(faultParameters["forced_rupture_time"] != NULL ){
            forced_rupture_time[ltsFace][iBndGP]    = static_cast<real>( faultParameters["forced_rupture_time"][(meshFace) * numberOfPoints + iBndGP] );
          }
        }
        tn[ltsFace]= 0.0;
      }//lts-face loop
      layerLtsFaceToMeshFace += it->getNumberOfCells();
    }//leaf_iterator loop
  }
};



/*
 * nucleationStressInFaultCS initialized which is used to impose slip rates on the fault surface
 */
class seissol::initializers::Init_ImposedSlipRatesFL33 : public seissol::initializers::BaseDrInitializer {
public:
  virtual void initializeFrictionMatrices(seissol::initializers::DynamicRupture *dynRup,
          initializers::LTSTree* dynRupTree,
          seissol::dr::fr_law::BaseFrictionSolver* FrictionSolver,
          std::unordered_map<std::string, double*> faultParameters,
          unsigned* ltsFaceToMeshFace,
          seissol::Interoperability &e_interoperability) override {
    BaseDrInitializer::initializeFrictionMatrices(dynRup, dynRupTree, FrictionSolver, faultParameters, ltsFaceToMeshFace, e_interoperability);
    seissol::initializers::LTS_ImposedSlipRatesFL33 *ConcreteLts = dynamic_cast<seissol::initializers::LTS_ImposedSlipRatesFL33 *>(dynRup);

    unsigned* layerLtsFaceToMeshFace = ltsFaceToMeshFace;

    for (initializers::LTSTree::leaf_iterator it = dynRupTree->beginLeaf(initializers::LayerMask(Ghost)); it != dynRupTree->endLeaf(); ++it) {

      real  (*nucleationStressInFaultCS)[numOfPointsPadded][6]  = it->var(ConcreteLts->nucleationStressInFaultCS); //get from fortran
      real *averaged_Slip                                       = it->var(ConcreteLts->averaged_Slip);      // = 0

      for (unsigned ltsFace = 0; ltsFace < it->getNumberOfCells(); ++ltsFace) {
        unsigned meshFace = layerLtsFaceToMeshFace[ltsFace];

        e_interoperability.getDynRupNucStress(ltsFace, meshFace, nucleationStressInFaultCS);
        averaged_Slip[ltsFace]= 0.0;

      }//lts-face loop
      layerLtsFaceToMeshFace += it->getNumberOfCells();
    }//leaf_iterator loop
  }
};

/*
 * time and space dependent nucleation parameter for FL103
 * plus state Variable and dynamic stress required for rate and state friction laws
 */
class seissol::initializers::Init_RateAndStateFL103 : public seissol::initializers::BaseDrInitializer {
public:
  virtual void initializeFrictionMatrices(seissol::initializers::DynamicRupture *dynRup,
                                          initializers::LTSTree* dynRupTree,
                                          seissol::dr::fr_law::BaseFrictionSolver* FrictionSolver,
                                          std::unordered_map<std::string, double*> faultParameters,
                                          unsigned* ltsFaceToMeshFace,
                                          seissol::Interoperability &e_interoperability) override {
    BaseDrInitializer::initializeFrictionMatrices(dynRup, dynRupTree, FrictionSolver, faultParameters, ltsFaceToMeshFace, e_interoperability);
    seissol::initializers::LTS_RateAndStateFL103 *ConcreteLts = dynamic_cast<seissol::initializers::LTS_RateAndStateFL103 *>(dynRup);

    unsigned* layerLtsFaceToMeshFace = ltsFaceToMeshFace;

    for (initializers::LTSTree::leaf_iterator it = dynRupTree->beginLeaf(initializers::LayerMask(Ghost)); it != dynRupTree->endLeaf(); ++it) {


      real  (*nucleationStressInFaultCS)[numOfPointsPadded][6]  = it->var(ConcreteLts->nucleationStressInFaultCS); //get from fortran
      real (*RS_sl0_array)[numOfPointsPadded]                   = it->var(ConcreteLts->RS_sl0_array);       //get from faultParameters
      real (*RS_a_array)[numOfPointsPadded]                     = it->var(ConcreteLts->RS_a_array);         //get from faultParameters
      real (*RS_srW_array)[numOfPointsPadded]                   = it->var(ConcreteLts->RS_srW_array);       //get from faultParameters
      bool (*DS)[numOfPointsPadded]                             = it->var(ConcreteLts->DS);                 //par file
      real *averaged_Slip                                       = it->var(ConcreteLts->averaged_Slip);      // = 0
      real (*stateVar)[numOfPointsPadded]                       = it->var(ConcreteLts->stateVar);           //get from Fortran = EQN%IniStateVar
      real (*dynStress_time)[numOfPointsPadded]                 = it->var(ConcreteLts->dynStress_time);     // = 0

      for (unsigned ltsFace = 0; ltsFace < it->getNumberOfCells(); ++ltsFace) {
        unsigned meshFace = layerLtsFaceToMeshFace[ltsFace];

        e_interoperability.getDynRupStateVar(ltsFace, meshFace, stateVar);
        e_interoperability.getDynRupNucStress(ltsFace, meshFace, nucleationStressInFaultCS);

        for (unsigned iBndGP = 0; iBndGP < numOfPointsPadded; ++iBndGP) {    //loop includes padded elements
          dynStress_time[ltsFace][iBndGP] = 0.0;
          DS[ltsFace][iBndGP] = m_Params->IsDsOutputOn;
        }
        averaged_Slip[ltsFace]= 0.0;

        for (unsigned iBndGP = 0; iBndGP < numberOfPoints; ++iBndGP) {
          RS_a_array[ltsFace][iBndGP] = static_cast<real>( faultParameters["rs_a"][(meshFace) * numberOfPoints+ iBndGP] );
          RS_srW_array[ltsFace][iBndGP] = static_cast<real>( faultParameters["rs_srW"][(meshFace) * numberOfPoints+ iBndGP] );
          RS_sl0_array[ltsFace][iBndGP] = static_cast<real>( faultParameters["RS_sl0"][(meshFace) * numberOfPoints+ iBndGP] );
        }
        //initialize padded elements for vectorization
        for (unsigned iBndGP = numberOfPoints; iBndGP < numOfPointsPadded; ++iBndGP) {
          RS_a_array[ltsFace][iBndGP] = 0.0;
          RS_srW_array[ltsFace][iBndGP] = 0.0;
          RS_sl0_array[ltsFace][iBndGP] = 0.0;
        }

      }//lts-face loop
      layerLtsFaceToMeshFace += it->getNumberOfCells();
    }//leaf_iterator loop
  }
};

/*
 * initialize all thermal pressure parameters
 */
class seissol::initializers::Init_RateAndStateFL103TP : public seissol::initializers::Init_RateAndStateFL103 {
public:

  virtual void initializeFrictionMatrices(seissol::initializers::DynamicRupture *dynRup,
                                          initializers::LTSTree* dynRupTree,
                                          seissol::dr::fr_law::BaseFrictionSolver* FrictionSolver,
                                          std::unordered_map<std::string, double*> faultParameters,
                                          unsigned* ltsFaceToMeshFace,
                                          seissol::Interoperability &e_interoperability) override {


    //BaseDrInitializer::initializeFrictionMatrices(dynRup, dynRupTree, faultParameters, ltsFaceToMeshFace, e_interoperability);
    Init_RateAndStateFL103::initializeFrictionMatrices(dynRup, dynRupTree, FrictionSolver, faultParameters, ltsFaceToMeshFace, e_interoperability);

    seissol::initializers::LTS_RateAndStateFL103TP *ConcreteLts = dynamic_cast<seissol::initializers::LTS_RateAndStateFL103TP *>(dynRup);
    seissol::dr::fr_law::RateAndStateThermalFL103 *SolverFL103 = dynamic_cast<seissol::dr::fr_law::RateAndStateThermalFL103 *>(FrictionSolver);
    SolverFL103->initializeTP(e_interoperability);

    unsigned* layerLtsFaceToMeshFace = ltsFaceToMeshFace;

    for (initializers::LTSTree::leaf_iterator it = dynRupTree->beginLeaf(initializers::LayerMask(Ghost)); it != dynRupTree->endLeaf(); ++it) {


      real (*temperature)[numOfPointsPadded]                    = it->var(ConcreteLts->temperature);
      real (*pressure)[numOfPointsPadded]                       = it->var(ConcreteLts->pressure);

      real (*TP_Theta)[numOfPointsPadded][TP_grid_nz]           = it->var(ConcreteLts->TP_theta);
      real (*TP_sigma)[numOfPointsPadded][TP_grid_nz]           = it->var(ConcreteLts->TP_sigma);

      real (*TP_half_width_shear_zone)[numOfPointsPadded]       = it->var(ConcreteLts->TP_half_width_shear_zone);
      real (*alpha_hy)[numOfPointsPadded]                       = it->var(ConcreteLts->alpha_hy);


      for (unsigned ltsFace = 0; ltsFace < it->getNumberOfCells(); ++ltsFace) {
        unsigned meshFace = layerLtsFaceToMeshFace[ltsFace];

        for (unsigned iBndGP = 0; iBndGP < numOfPointsPadded; ++iBndGP) {
          temperature[ltsFace][iBndGP] = m_Params->IniTemp;
          pressure[ltsFace][iBndGP] = m_Params->IniPressure;
          TP_half_width_shear_zone[ltsFace][iBndGP] = static_cast<real>( faultParameters["TP_half_width_shear_zone"][(meshFace) * numberOfPoints + iBndGP] );
          alpha_hy[ltsFace][iBndGP] = static_cast<real>( faultParameters["alpha_hy"][(meshFace) * numberOfPoints + iBndGP] );
          for (unsigned iTP_grid_nz = 0; iTP_grid_nz < TP_grid_nz; ++iTP_grid_nz) {
            TP_Theta[ltsFace][iBndGP][iTP_grid_nz] = 0.0;
            TP_sigma[ltsFace][iBndGP][iTP_grid_nz] = 0.0;
          }
        }
      }//lts-face loop
      layerLtsFaceToMeshFace += it->getNumberOfCells();
    }//leaf_iterator loop

  }
};

/*
 * strength data is additionally initialized by computing it from friction and inital normal stress
 */
class seissol::initializers::Init_LinearBimaterialFL6 : public seissol::initializers::Init_LinearSlipWeakeningFL2 {
public:
  virtual void initializeFrictionMatrices(seissol::initializers::DynamicRupture *dynRup,
                                          initializers::LTSTree* dynRupTree,
                                          seissol::dr::fr_law::BaseFrictionSolver* FrictionSolver,
                                          std::unordered_map<std::string,
                                          double*> faultParameters,
                                          unsigned* ltsFaceToMeshFace,
                                          seissol::Interoperability &e_interoperability) override {
    Init_LinearSlipWeakeningFL2::initializeFrictionMatrices(dynRup, dynRupTree, FrictionSolver, faultParameters, ltsFaceToMeshFace, e_interoperability);
    seissol::initializers::LTS_LinearBimaterialFL6 *ConcreteLts = dynamic_cast<seissol::initializers::LTS_LinearBimaterialFL6 *>(dynRup);

    unsigned* layerLtsFaceToMeshFace = ltsFaceToMeshFace;

    for (initializers::LTSTree::leaf_iterator it = dynRupTree->beginLeaf(initializers::LayerMask(Ghost)); it != dynRupTree->endLeaf(); ++it) {
      real (*strengthData)[numOfPointsPadded]               = it->var(ConcreteLts->strengthData);
      real (*mu)[numOfPointsPadded]                         = it->var(ConcreteLts->mu);
      real  (*initialStressInFaultCS)[numOfPointsPadded][6] = it->var(dynRup->initialStressInFaultCS);

      for (unsigned ltsFace = 0; ltsFace < it->getNumberOfCells(); ++ltsFace) {
        //unsigned meshFace = layerLtsFaceToMeshFace[ltsFace];
        for (unsigned iBndGP = 0; iBndGP < numOfPointsPadded; ++iBndGP) {
          strengthData[ltsFace][iBndGP] =  mu[ltsFace][iBndGP] * initialStressInFaultCS[ltsFace][iBndGP][0];
        }
      }//lts-face loop
      layerLtsFaceToMeshFace += it->getNumberOfCells();
    }//leaf_iterator loop
  }
};



/*
 * should be revisted !
 * parameter could be obtain with m_Param (but initializer in Fortran does currently not suppport this friction law and FL4,7)
 */
class seissol::initializers::Init_RateAndStateFL3 : public seissol::initializers::BaseDrInitializer {
public:
  virtual void initializeFrictionMatrices(seissol::initializers::DynamicRupture *dynRup,
                                          initializers::LTSTree* dynRupTree,
                                          seissol::dr::fr_law::BaseFrictionSolver* FrictionSolver,
                                          std::unordered_map<std::string, double*> faultParameters,
                                          unsigned* ltsFaceToMeshFace,
                                          seissol::Interoperability &e_interoperability) override {
    BaseDrInitializer::initializeFrictionMatrices(dynRup, dynRupTree, FrictionSolver, faultParameters, ltsFaceToMeshFace, e_interoperability);
    seissol::initializers::LTS_RateAndStateFL3 *ConcreteLts = dynamic_cast<seissol::initializers::LTS_RateAndStateFL3 *>(dynRup);
    unsigned* layerLtsFaceToMeshFace = ltsFaceToMeshFace;

    for (initializers::LTSTree::leaf_iterator it = dynRupTree->beginLeaf(initializers::LayerMask(Ghost)); it != dynRupTree->endLeaf(); ++it) {
      real *RS_f0                                               = it->var(ConcreteLts->RS_f0);
      real *RS_a                                                = it->var(ConcreteLts->RS_a);
      real *RS_b                                                = it->var(ConcreteLts->RS_b);
      real *RS_sl0                                              = it->var(ConcreteLts->RS_sl0);
      real *RS_sr0                                              = it->var(ConcreteLts->RS_sr0);
      real (*stateVar)[numOfPointsPadded]                       = it->var(ConcreteLts->stateVar);

      for (unsigned ltsFace = 0; ltsFace < it->getNumberOfCells(); ++ltsFace) {
        unsigned meshFace = layerLtsFaceToMeshFace[ltsFace];

        //get initial values from fortran
        //TODO: RS_a, RS_b, rs_f0, RS_sl0, RS_sr0 could be obtained from paramter file
        e_interoperability.getDynRupFL_3(ltsFace, meshFace, RS_f0, RS_a, RS_b, RS_sl0, RS_sr0);
        e_interoperability.getDynRupStateVar(ltsFace, meshFace, stateVar);

      }//lts-face loop
      layerLtsFaceToMeshFace += it->getNumberOfCells();
    }//leaf_iterator loop
  }
};



#endif //SEISSOL_DR_INITIALIZER_BASE_H
