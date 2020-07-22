//
// Created by adrian on 09.07.20.
//

#ifndef SEISSOL_DR_FRICTION_LAW_H
#define SEISSOL_DR_FRICTION_LAW_H

#include <c++/8.3.0/iostream>
#include "DR_LTS_Base.h"

namespace seissol {
    namespace dr {
        namespace fr_law {
            class Base {
            public:
                //TODO: rename e.g. BaseSolverFL
                virtual ~Base() {}

            protected:
                void precompute(){

                }

                void postcompute(){

                }

            public:
                virtual void evaluate(seissol::initializers::Layer&  layerData,
                                       seissol::initializers::DynamicRupture *dynRup,
                                       real (*QInterpolatedPlus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
                                       real (*QInterpolatedMinus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
                                       unsigned face,
                                        real fullUpdateTime,
                                        real timeWeights[CONVERGENCE_ORDER],
                                        real DeltaT[CONVERGENCE_ORDER]) = 0;
            };

            class FL_2 : public Base {
            public:
                virtual void hook() {}
                virtual void evaluate(seissol::initializers::Layer&  layerData,
                        seissol::initializers::DynamicRupture *dynRup,
                        real (*QInterpolatedPlus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
                        real (*QInterpolatedMinus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
                        unsigned face,
                        real fullUpdateTime,
                        real timeWeights[CONVERGENCE_ORDER],
                        real DeltaT[CONVERGENCE_ORDER]) override {

                    #ifdef _OPENMP
                    #pragma omp parallel for schedule(static) //private(QInterpolatedPlus,QInterpolatedMinus)
                    #endif
                    //TODO: split loop
                    for (unsigned face = 0; face < layerData.getNumberOfCells(); ++face) {

                        //TODO: change later to const_cast
                        seissol::initializers::DR_FL_2 *ConcreteLts = dynamic_cast<seissol::initializers::DR_FL_2 *>(dynRup);
                        //std::cout << "computing DR for FL_2\n";

                        //DRFaceInformation*                    faceInformation                                                   = layerData.var(ConcreteLts->faceInformation);
                        real                                (*imposedStatePlus)[tensor::QInterpolated::size()]                  = layerData.var(ConcreteLts->imposedStatePlus);
                        real                                (*imposedStateMinus)[tensor::QInterpolated::size()]                 = layerData.var(ConcreteLts->imposedStateMinus);
                        seissol::model::IsotropicWaveSpeeds*  waveSpeedsPlus                                                    = layerData.var(ConcreteLts->waveSpeedsPlus);
                        seissol::model::IsotropicWaveSpeeds*  waveSpeedsMinus                                                   = layerData.var(ConcreteLts->waveSpeedsMinus);
                        real*                                   lts_t_0                                                         = layerData.var(ConcreteLts->t_0);
                        real                    (*initialStressInFaultCS)[init::QInterpolated::Stop[0]][6]                      = layerData.var(ConcreteLts->initialStressInFaultCS);
                        real                    (*cohesion)[init::QInterpolated::Stop[0]]                                       = layerData.var(ConcreteLts->cohesion);
                        real                    (*mu)[init::QInterpolated::Stop[0]]                                             = layerData.var(ConcreteLts->mu);
                        real                    (*slip)[init::QInterpolated::Stop[0]]                                           = layerData.var(ConcreteLts->slip);
                        real                    (*slip1)[init::QInterpolated::Stop[0]]                                          = layerData.var(ConcreteLts->slip1);
                        real                    (*slip2)[init::QInterpolated::Stop[0]]                                          = layerData.var(ConcreteLts->slip2);
                        real                    (*d_c)[init::QInterpolated::Stop[0]]                                            = layerData.var(ConcreteLts->d_c);
                        real                    (*mu_S)[init::QInterpolated::Stop[0]]                                           = layerData.var(ConcreteLts->mu_S);
                        real                    (*mu_D)[init::QInterpolated::Stop[0]]                                           = layerData.var(ConcreteLts->mu_D);
                        real                    (*rupture_time)[init::QInterpolated::Stop[0]]                                   = layerData.var(ConcreteLts->rupture_time);
                        bool                    (*RF)[init::QInterpolated::Stop[0]]                                             = layerData.var(ConcreteLts->RF);
                        bool                    (*DS)[init::QInterpolated::Stop[0]]                                             = layerData.var(ConcreteLts->DS);
                        real                    (*peakSR)[init::QInterpolated::Stop[0]]                                         = layerData.var(ConcreteLts->peakSR);
                        real                    (*dynStress_time)[init::QInterpolated::Stop[0]]                                 = layerData.var(ConcreteLts->dynStress_time);
                        real                    (*tracXY)[init::QInterpolated::Stop[0]]                                         = layerData.var(ConcreteLts->tracXY);
                        real                    (*tracXZ)[init::QInterpolated::Stop[0]]                                         = layerData.var(ConcreteLts->tracXZ);
                        real                    (*slipRate1)[init::QInterpolated::Stop[0]]                                      = layerData.var(ConcreteLts->slipRate1);
                        real                    (*slipRate2)[init::QInterpolated::Stop[0]]                                      = layerData.var(ConcreteLts->slipRate2);
                        bool                    (*magnitude_out)                                                                = layerData.var(ConcreteLts->magnitude_out);
                        real                    (*averaged_Slip)                                                                = layerData.var(ConcreteLts->averaged_Slip);


                        precompute();

                        constexpr int numberOfPoints =  tensor::QInterpolated::Shape[0];// DISC%Galerkin%nBndGP

                        //TODO: is start always 0?
                        //TODO: implement padded calculation
                        constexpr int numOfPointsPadded = tensor::QInterpolated::Shape[0];// init::QInterpolated::Stop[0] - init::QInterpolated::Start[0];


                        real TractionGP_XY[numOfPointsPadded][CONVERGENCE_ORDER] = {{}}; // OUT: updated Traction 2D array with size [1:i_numberOfPoints, CONVERGENCE_ORDER]
                        real TractionGP_XZ[numOfPointsPadded][CONVERGENCE_ORDER] = {{}};// OUT: updated Traction 2D array with size [1:i_numberOfPoints, CONVERGENCE_ORDER]
                        real NorStressGP[numOfPointsPadded][CONVERGENCE_ORDER] = {{}};
                        real XYStressGP[numOfPointsPadded][CONVERGENCE_ORDER]= {{}};
                        real XZStressGP[numOfPointsPadded][CONVERGENCE_ORDER]= {{}};
                        real Zp_inv, Zp_neig_inv, Zs_inv, Zs_neig_inv, eta_p, eta_s;

                        //int iFace = static_cast<int>(faceInformation.meshFace);

                        Zp_inv = 1.0 / (waveSpeedsPlus->density * waveSpeedsPlus->pWaveVelocity);
                        Zp_neig_inv = 1.0 / (waveSpeedsMinus->density * waveSpeedsMinus->pWaveVelocity);
                        Zs_inv = 1.0 / (waveSpeedsPlus->density * waveSpeedsPlus->sWaveVelocity);
                        Zs_neig_inv = 1.0 / (waveSpeedsMinus->density * waveSpeedsMinus->sWaveVelocity);
                        eta_p = 1.0 / (Zp_inv + Zp_neig_inv);
                        eta_s = 1.0 / (Zs_inv + Zs_neig_inv);


                        for(int j = 0; j < CONVERGENCE_ORDER; j++){
                            auto QInterpolatedPlusView = init::QInterpolated::view::create(QInterpolatedPlus[face][j]);
                            auto QInterpolatedMinusView = init::QInterpolated::view::create(QInterpolatedMinus[face][j]);
                            //TODO: does QInterpolatedMinusView work with padded access?
                            for(int i = 0; i < numOfPointsPadded; i++){
                                NorStressGP[i][j] = eta_p * (QInterpolatedMinusView(i,6) - QInterpolatedPlusView(i,6) + QInterpolatedPlusView(i,0) * Zp_inv + QInterpolatedMinusView(i,0) * Zp_neig_inv);
                                XYStressGP[i][j]  = eta_s * (QInterpolatedMinusView(i,7) - QInterpolatedPlusView(i,7) + QInterpolatedPlusView(i,3) * Zs_inv + QInterpolatedMinusView(i,3) * Zs_neig_inv);
                                XZStressGP[i][j] = eta_s * (QInterpolatedMinusView(i,8) - QInterpolatedPlusView(i,8) + QInterpolatedPlusView(i,5) * Zs_inv + QInterpolatedMinusView(i,5) * Zs_neig_inv);
                            }
                        }

                        //TODO: is this assert really needed?
                        static_assert(tensor::QInterpolated::Shape[0] == tensor::resample::Shape[0], "Different number of quadrature points?");


                        //required input:
                        auto resampleMatrixView = init::resample::view::create(
                                const_cast<double *>(init::resample::Values));
                        real t_0 = lts_t_0[face]; //= DISC%DynRup%t_0

                        //bool inst_healing = frD.inst_healing;    //DISC%DynRup%inst_healing


                        //declare local variables
                        //TODO: replace arrays with this form: (zero initialized)
                        //std::array<double, numOfPointsPadded> P{};
                        real sum_tmpSlip;
                        real tmpSlip[numOfPointsPadded];
                        real matmul[numOfPointsPadded];

                        real Z = waveSpeedsPlus->density * waveSpeedsPlus->sWaveVelocity;
                        real Z_neig = waveSpeedsMinus->density * waveSpeedsMinus->sWaveVelocity;
                        real eta = Z * Z_neig / (Z + Z_neig);

                        real tn = fullUpdateTime;
                        real time_inc = 0;
                        real P[numOfPointsPadded];
                        real Strength[numOfPointsPadded];
                        real ShTest[numOfPointsPadded];
                        real LocSR[numOfPointsPadded];
                        real LocSR1[numOfPointsPadded];
                        real LocSR2[numOfPointsPadded];
                        real LocTracXY[numOfPointsPadded];
                        real LocTracXZ[numOfPointsPadded];
                        real f1;
                        real f2;
                        real test3[numOfPointsPadded];
                        real test5[numOfPointsPadded];

                        //TODO change this to calloc with free
                        //initialize to 0
                        for (int i = 0; i < numOfPointsPadded; i++) {
                            tmpSlip[i] = 0.0; //D0
                            P[i] = 0.0;
                            Strength[i] = 0.0;
                            ShTest[i] = 0.0;
                            LocSR[i] = 0.0;
                            LocSR1[i] = 0.0;
                            LocSR2[i] = 0.0;
                            LocTracXY[i] = 0.0;
                            LocTracXZ[i] = 0.0;
                            matmul[i] = 0.0;
                            //f1[nBndGP]= 0.0;
                            //f2[nBndGP]= 0.0;
                        }


                        for (int iTimeGP = 0; iTimeGP < CONVERGENCE_ORDER; iTimeGP++) {  //loop over time steps
                            //TODO: test if access works with padded
                            for (int iBndGP = 0; iBndGP < numOfPointsPadded; iBndGP++) {
                                time_inc = DeltaT[iTimeGP];
                                tn = tn + time_inc;

                                P[iBndGP] = initialStressInFaultCS[face][iBndGP][0] + NorStressGP[iBndGP][iTimeGP];
                                Strength[iBndGP] = cohesion[face][iBndGP] - mu[face][iBndGP] * std::min(P[iBndGP], 0.0);


                                ShTest[iBndGP] = std::sqrt(
                                        std::pow(initialStressInFaultCS[face][iBndGP][3] + XYStressGP[iBndGP][iTimeGP],
                                                 2) +
                                        std::pow(initialStressInFaultCS[face][iBndGP][5] + XZStressGP[iBndGP][iTimeGP],
                                                 2));
                                LocSR[iBndGP] = std::max(0.0, (ShTest[iBndGP] - Strength[iBndGP]) / eta);
                                /*
                                LocSR1[iBndGP] =
                                        LocSR[iBndGP] * (frD.getInitialStressInFaultCS(iBndGP, 3, iFace) + XYStressGP[iBndGP][iTimeGP]) /
                                        (Strength[iBndGP] + eta * LocSR[iBndGP]);
                                LocSR2[iBndGP] =
                                        LocSR[iBndGP] * (frD.getInitialStressInFaultCS(iBndGP, 5, iFace) + XZStressGP[iBndGP][iTimeGP]) /
                                        (Strength[iBndGP] + eta * LocSR[iBndGP]);
                                */
                                //TODO: check alternative faster calc??
                                LocSR1[iBndGP] = LocSR[iBndGP] * (initialStressInFaultCS[face][iBndGP][3] +
                                                                  XYStressGP[iBndGP][iTimeGP]) /
                                                 (std::max(ShTest[iBndGP], Strength[iBndGP]));
                                LocSR2[iBndGP] = LocSR[iBndGP] * (initialStressInFaultCS[face][iBndGP][5] +
                                                                  XZStressGP[iBndGP][iTimeGP]) /
                                                 (std::max(ShTest[iBndGP], Strength[iBndGP]));

                                test3[iBndGP] = initialStressInFaultCS[face][iBndGP][3];
                                test5[iBndGP] = initialStressInFaultCS[face][iBndGP][5];

                                LocTracXY[iBndGP] = XYStressGP[iBndGP][iTimeGP] - eta * LocSR1[iBndGP];
                                LocTracXZ[iBndGP] = XZStressGP[iBndGP][iTimeGP] - eta * LocSR2[iBndGP];


                                //Update slip
                                slip1[face][iBndGP] = slip1[face][iBndGP] + LocSR1[iBndGP] * time_inc;
                                slip2[face][iBndGP] = slip2[face][iBndGP] + LocSR2[iBndGP] * time_inc;
                            }
                            //TODO: test if access works with padded
                            for (int iBndGP = 0; iBndGP < numberOfPoints; iBndGP++) {
                                //Resample slip-rate, such that the state (Slip) lies in the same polynomial space as the degrees of freedom
                                //resampleMatrix first projects LocSR on the two-dimensional basis on the reference triangle with
                                //degree less or equal than CONVERGENCE_ORDER-1, and then evaluates the polynomial at the quadrature points
                                matmul[iBndGP] = 0;

                                //TODO: test if access works with padded
                                for (int j = 0; j < numberOfPoints; j++) {
                                    matmul[iBndGP] += resampleMatrixView(iBndGP, j) * LocSR[j];
                                }
                                slip[face][iBndGP] = slip[face][iBndGP] + matmul[iBndGP] * time_inc;
                                tmpSlip[iBndGP] = tmpSlip[iBndGP] + LocSR[iBndGP] * time_inc;


                                //Modif T. Ulrich-> generalisation of tpv16/17 to 30/31

                                f1 = std::min(std::abs(slip[face][iBndGP]) / d_c[face][iBndGP], 1.0);

                                //TODO: implement hook
                                f2 = 0;
                                hook(); //for FL_16

                                mu[face][iBndGP] = mu_S[face][iBndGP] -
                                                   (mu_S[face][iBndGP] - mu_D[face][iBndGP]) * std::max(f1, f2);

                                //instantaneous healing
                                //TODO: hook_insta_healing();
                                //hook_insta_healing();
                                /*
                                if (inst_healing == true) {
                                    if (LocSR[iBndGP] < u_0) {
                                        frD.getMu(iBndGP, iFace) = frictionData.mu_S[iBndGP];
                                        frictionData.slip[iBndGP] = 0.0;
                                    }
                                }
                                 */

                                //TODO: replace all LocTracXY[iBndGP] with TractionGP_XY[iBndGP][iTimeGP]
                                TractionGP_XY[iBndGP][iTimeGP] = LocTracXY[iBndGP];
                                assert(!std::isnan(TractionGP_XY[iBndGP][iTimeGP]));
                                TractionGP_XZ[iBndGP][iTimeGP] = LocTracXZ[iBndGP];
                            }
                        }     //end iTimeGP loop
                        //TODO: test if access works with padded
                        for (int iBndGP = 0; iBndGP < numOfPointsPadded; iBndGP++) {

                            // output rupture front
                            // outside of iTimeGP loop in order to safe an 'if' in a loop
                            // this way, no subtimestep resolution possible
                            if (RF[face][iBndGP] && LocSR[iBndGP] > 0.001) {
                                rupture_time[face][iBndGP] = fullUpdateTime;
                                RF[face][iBndGP] = false;
                            }
                            //output time when shear stress is equal to the dynamic stress after rupture arrived
                            //currently only for linear slip weakening
                            if (rupture_time[face][iBndGP] > 0.0 &&
                                rupture_time[face][iBndGP] <= fullUpdateTime &&
                                DS[face][iBndGP] &&
                                std::abs(slip[face][iBndGP]) >= d_c[face][iBndGP]) {
                                dynStress_time[face][iBndGP] = fullUpdateTime;
                                DS[face][iBndGP] = false;
                            }

                            if (LocSR[iBndGP] > peakSR[face][iBndGP]) {
                                peakSR[face][iBndGP] = LocSR[iBndGP];
                            }
                            tracXY[face][iBndGP] = LocTracXY[iBndGP];
                            tracXZ[face][iBndGP] = LocTracXZ[iBndGP];

                            slipRate1[face][iBndGP] = LocSR1[iBndGP];
                            slipRate2[face][iBndGP] = LocSR2[iBndGP];
                        } //end i < nBndGP loop

                        //---compute and store slip to determine the magnitude of an earthquake ---
                        //    to this end, here the slip is computed and averaged per element
                        //    in calc_seissol.f90 this value will be multiplied by the element surface
                        //    and an output happened once at the end of the simulation
                        sum_tmpSlip = 0;
                        if (magnitude_out[face]) {
                            //TODO: check if padding does not mess up calculation
                            for (int iBndGP = 0; iBndGP < numOfPointsPadded; iBndGP++) {
                                sum_tmpSlip += tmpSlip[iBndGP];
                            }
                            averaged_Slip[face] = averaged_Slip[face] + sum_tmpSlip / numberOfPoints;
                        }

                        auto imposedStatePlusView = init::QInterpolated::view::create(imposedStatePlus[face]);
                        auto imposedStateMinusView = init::QInterpolated::view::create(imposedStateMinus[face]);
                        //initialize to 0
                        imposedStateMinusView.setZero();
                        imposedStatePlusView.setZero();

                        for (int j = 0; j < CONVERGENCE_ORDER; j++) {
                            auto QInterpolatedPlusView = init::QInterpolated::view::create(QInterpolatedPlus[face][j]);
                            auto QInterpolatedMinusView = init::QInterpolated::view::create(QInterpolatedMinus[face][j]);
                            for (int i = 0; i < numberOfPoints; i++) {
                                imposedStateMinusView(i, 0) += timeWeights[j] * NorStressGP[i][j];
                                imposedStateMinusView(i, 3) += timeWeights[j] * TractionGP_XY[i][j];
                                imposedStateMinusView(i, 5) += timeWeights[j] * TractionGP_XZ[i][j];
                                imposedStateMinusView(i, 6) += timeWeights[j] * (QInterpolatedMinusView(i, 6) -
                                                                                 Zp_neig_inv * (NorStressGP[i][j] -
                                                                                                QInterpolatedMinusView(
                                                                                                        i, 0)));
                                imposedStateMinusView(i, 7) += timeWeights[j] * (QInterpolatedMinusView(i, 7) -
                                                                                 Zs_neig_inv * (TractionGP_XY[i][j] -
                                                                                                QInterpolatedMinusView(
                                                                                                        i, 3)));
                                imposedStateMinusView(i, 8) += timeWeights[j] * (QInterpolatedMinusView(i, 8) -
                                                                                 Zs_neig_inv * (TractionGP_XZ[i][j] -
                                                                                                QInterpolatedMinusView(
                                                                                                        i, 5)));

                                imposedStatePlusView(i, 0) += timeWeights[j] * NorStressGP[i][j];
                                imposedStatePlusView(i, 3) += timeWeights[j] * TractionGP_XY[i][j];
                                imposedStatePlusView(i, 5) += timeWeights[j] * TractionGP_XZ[i][j];
                                imposedStatePlusView(i, 6) += timeWeights[j] * (QInterpolatedPlusView(i, 6) + Zp_inv *
                                                                                                              (NorStressGP[i][j] -
                                                                                                               QInterpolatedPlusView(
                                                                                                                       i,
                                                                                                                       0)));
                                imposedStatePlusView(i, 7) += timeWeights[j] * (QInterpolatedPlusView(i, 7) + Zs_inv *
                                                                                                              (TractionGP_XY[i][j] -
                                                                                                               QInterpolatedPlusView(
                                                                                                                       i,
                                                                                                                       3)));
                                imposedStatePlusView(i, 8) += timeWeights[j] * (QInterpolatedPlusView(i, 8) + Zs_inv *
                                                                                                              (TractionGP_XZ[i][j] -
                                                                                                               QInterpolatedPlusView(
                                                                                                                       i,
                                                                                                                       5)));
                            } //End numberOfPoints-loop
                        } //End CONVERGENCE_ORDER-loop

                        /*
                        real XXXimposed[468] = {0};

                        for(int i = 0; i < tensor::QInterpolated::size(); i++){
                            XXXimposed[i] = imposedStateMinus[face][i];
                        }
                        for (int i = 0; i < numberOfPoints; i++) {
                            for(int j = 0; j < 9; j++){
                                assert( !std::isnan( imposedStateMinusView(i,j) ) );
                            }

                        }

                        for(int i = 0; i < tensor::QInterpolated::size(); i++){
                            assert( !std::isnan(imposedStateMinus[face][i]) );
                        }
                        */

                    }

                }//End of Function evaluate
            };//End of Class

            class FL_17 : public Base {
            public:
                virtual void hook() {}

                virtual void evaluate(seissol::initializers::Layer&  layerData,
                                      seissol::initializers::DynamicRupture *dynRup,
                                      real (*QInterpolatedPlus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
                                      real (*QInterpolatedMinus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
                                      unsigned face,
                                      real fullUpdateTime,
                                      real timeWeights[CONVERGENCE_ORDER],
                                      real DeltaT[CONVERGENCE_ORDER]) override {
                    //std::cout << "computing ";
                    hook();
                    //std::cout << " DR for FL_16\n";
                }
            };

            class FL_16 : public FL_17 {
            public:
                virtual void hook() override {
                    //std::cout << "(hook)";
                }
            };

            class FL_33 : public Base {
            public:
                virtual void evaluate(seissol::initializers::Layer&  layerData,
                                      seissol::initializers::DynamicRupture *dynRup,
                                      real (*QInterpolatedPlus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
                                      real (*QInterpolatedMinus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
                                      unsigned face,
                                      real fullUpdateTime,
                                      real timeWeights[CONVERGENCE_ORDER],
                                      real DeltaT[CONVERGENCE_ORDER]) override {
                    seissol::initializers::DR_FL_33 *ConcreteLts = dynamic_cast<seissol::initializers::DR_FL_33 *>(dynRup);
                    std::cout << "computing DR for FL_33\n";
                }
            };
        }
    }
}

#endif //SEISSOL_DR_FRICTION_LAW_H