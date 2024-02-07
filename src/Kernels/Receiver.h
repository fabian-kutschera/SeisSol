/**
 * @file
 * This file is part of SeisSol.
 *
 * @author Carsten Uphoff (c.uphoff AT tum.de, http://www5.in.tum.de/wiki/index.php/Carsten_Uphoff,_M.Sc.)
 *
 * @section LICENSE
 * Copyright (c) 2019, SeisSol Group
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * @section DESCRIPTION
 **/

#ifndef KERNELS_RECEIVER_H_
#define KERNELS_RECEIVER_H_

#include <Eigen/Dense>
#include <Geometry/MeshReader.h>
#include <Initializer/LTS.h>
#include <Initializer/PointMapper.h>
#include <Initializer/tree/Lut.hpp>
#include <Kernels/Interface.hpp>
#include <Kernels/Time.h>
#include <Numerical_aux/BasisFunction.h>
#include <Numerical_aux/Transformation.h>
#include <generated_code/init.h>
#include <vector>

struct GlobalData;
namespace seissol {
  class SeisSol;

  namespace kernels {
    struct Receiver {
      Receiver(unsigned pointId,
               Eigen::Vector3d position,
               double const* elementCoords[4],
               kernels::LocalData data, size_t reserved)
          : pointId(pointId),
            position(std::move(position)),
            data(data) {
        output.reserve(reserved);

        auto xiEtaZeta = seissol::transformations::tetrahedronGlobalToReference(elementCoords[0], elementCoords[1], elementCoords[2], elementCoords[3], position);
        basisFunctions = basisFunction::SampledBasisFunctions<real>(CONVERGENCE_ORDER, xiEtaZeta[0], xiEtaZeta[1], xiEtaZeta[2]);
        basisFunctionDerivatives = basisFunction::SampledBasisFunctionDerivatives<real>(CONVERGENCE_ORDER, xiEtaZeta[0], xiEtaZeta[1], xiEtaZeta[2]);
        basisFunctionDerivatives.transformToGlobalCoordinates(elementCoords);
      }
      unsigned pointId;
      Eigen::Vector3d position;
      basisFunction::SampledBasisFunctions<real> basisFunctions;
      basisFunction::SampledBasisFunctionDerivatives<real> basisFunctionDerivatives;
      kernels::LocalData data;
      std::vector<real> output;
    };

    class ReceiverCluster {
    public:
      ReceiverCluster(seissol::SeisSol& seissolInstance)
        : m_nonZeroFlops(0), m_hardwareFlops(0),
          m_samplingInterval(1.0e99), m_syncPointInterval(0.0),
          seissolInstance(seissolInstance)
      {}

      ReceiverCluster(  GlobalData const*             global,
                        std::vector<unsigned> const&  quantities,
                        double                        samplingInterval,
                        double                        syncPointInterval,
                        bool                          computeRotation,
                        seissol::SeisSol&             seissolInstance)
        : m_quantities(quantities),
          m_samplingInterval(samplingInterval),
          m_syncPointInterval(syncPointInterval),
          m_computeRotation(computeRotation),
          seissolInstance(seissolInstance) {
        m_timeKernel.setHostGlobalData(global);
        m_timeKernel.flopsAder(m_nonZeroFlops, m_hardwareFlops);
      }

      void addReceiver( unsigned          meshId,
                        unsigned          pointId,
                        Eigen::Vector3d   const& point,
                        seissol::geometry::MeshReader const& mesh,
                        seissol::initializer::Lut const& ltsLut,
                        seissol::initializer::LTS const& lts );

      //! Returns new receiver time
      double calcReceivers( double time,
                            double expansionPoint,
                            double timeStepWidth );

      std::vector<Receiver>::iterator begin() {
        return m_receivers.begin();
      }

      std::vector<Receiver>::iterator end() {
        return m_receivers.end();
      }

      size_t ncols() const {
        size_t ncols = m_quantities.size();
        if (m_computeRotation) {
          ncols += 3;
        }
#ifdef MULTIPLE_SIMULATIONS
        ncols *= init::QAtPoint::Stop[0]-init::QAtPoint::Start[0];
#endif
        return 1 + ncols;
      }

    private:
      seissol::SeisSol& seissolInstance;
      std::vector<Receiver> m_receivers;
      seissol::kernels::Time m_timeKernel;
      std::vector<unsigned> m_quantities;
      unsigned m_nonZeroFlops;
      unsigned m_hardwareFlops;
      double m_samplingInterval;
      double m_syncPointInterval;
      bool m_computeRotation;

    };
  }
}

#endif
