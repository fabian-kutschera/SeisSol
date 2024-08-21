/**
 * @file
 * This file is part of SeisSol.
 *
 * @author Carsten Uphoff (c.uphoff AT tum.de,
 *http://www5.in.tum.de/wiki/index.php/Carsten_Uphoff,_M.Sc.)
 *
 * @section LICENSE
 * Copyright (c) 2016, SeisSol Group
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
 * IMPLIED WARRANTIES OF  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * @section DESCRIPTION
 * Dynamic Rupture kernel of SeisSol.
 **/

#include "DynamicRupture.h"

#include <Common/constants.hpp>
#include <DataTypes/ConditionalKey.hpp>
#include <DataTypes/ConditionalTable.hpp>
#include <DataTypes/EncodedConstants.hpp>
#include <Initializer/typedefs.hpp>
#include <Kernels/precision.hpp>
#include <Parallel/Runtime/Stream.hpp>
#include <cassert>
#include <cstring>
#include <stdint.h>
#include <tensor.h>

#include "Numerical/Quadrature.h"
#include "generated_code/kernel.h"
#ifdef ACL_DEVICE
#include "device.h"
#endif
#include <yateto.h>

void seissol::kernels::DynamicRupture::checkGlobalData(const GlobalData* global, size_t alignment) {
#ifndef NDEBUG
  for (unsigned face = 0; face < 4; ++face) {
    for (unsigned h = 0; h < 4; ++h) {
      assert(((const uintptr_t)global->faceToNodalMatrices(face, h)) % alignment == 0);
    }
  }
#endif
}

void seissol::kernels::DynamicRupture::setHostGlobalData(const GlobalData* global) {
  checkGlobalData(global, Alignment);
  m_krnlPrototype.V3mTo2n = global->faceToNodalMatrices;
  m_timeKernel.setHostGlobalData(global);
}

void seissol::kernels::DynamicRupture::setGlobalData(const CompoundGlobalData& global) {
  this->setHostGlobalData(global.onHost);
#ifdef ACL_DEVICE
  assert(global.onDevice != nullptr);
  const auto deviceAlignment = device.api->getGlobMemAlignment();
  checkGlobalData(global.onDevice, deviceAlignment);
  m_gpuKrnlPrototype.V3mTo2n = global.onDevice->faceToNodalMatrices;
  m_timeKernel.setGlobalData(global);
#endif
}

void seissol::kernels::DynamicRupture::setTimeStepWidth(double timestep) {
#ifdef USE_DR_CELLAVERAGE
  static_assert(false, "Cell average currently not supported");
  /*double subIntervalWidth = timestep / ConvergenceOrder;
  for (unsigned timeInterval = 0; timeInterval < ConvergenceOrder; ++timeInterval) {
    double t1 = timeInterval * subIntervalWidth;
    double t2 = t1 + subIntervalWidth;
    /// Compute time-integrated Taylor expansion (at t0=0) weights for interval [t1,t2].
    unsigned factorial = 1;
    for (unsigned derivative = 0; derivative < ConvergenceOrder; ++derivative) {
      m_timeFactors[timeInterval][derivative] = (t2-t1) / (factorial * subIntervalWidth);
      t1 *= t1;
      t2 *= t2;
      factorial *= (derivative+2);
    }
    /// We define the time "point" of the interval as the centre of the interval in order
    /// to be somewhat compatible to legacy code.
    timePoints[timeInterval] = timeInterval * subIntervalWidth + subIntervalWidth / 2.;
    timeWeights[timeInterval] = subIntervalWidth;
  }*/
#else
  // TODO(Lukas) Cache unscaled points/weights to avoid costly recomputation every timestep.
  seissol::quadrature::GaussLegendre(timePoints, timeWeights, ConvergenceOrder);
  for (unsigned point = 0; point < ConvergenceOrder; ++point) {
#ifdef USE_STP
    double tau = timePoints[point];
    timeBasisFunctions[point] =
        std::make_shared<seissol::basisFunction::SampledTimeBasisFunctions<real>>(ConvergenceOrder,
                                                                                  tau);
#endif
    timePoints[point] = 0.5 * (timestep * timePoints[point] + timestep);
    timeWeights[point] = 0.5 * timestep * timeWeights[point];
  }
#endif
}

void seissol::kernels::DynamicRupture::spaceTimeInterpolation(
    const DRFaceInformation& faceInfo,
    const GlobalData* global,
    const DRGodunovData* godunovData,
    DREnergyOutput* drEnergyOutput,
    const real* timeDerivativePlus,
    const real* timeDerivativeMinus,
    real qInterpolatedPlus[ConvergenceOrder][seissol::tensor::QInterpolated::size()],
    real qInterpolatedMinus[ConvergenceOrder][seissol::tensor::QInterpolated::size()],
    const real* timeDerivativePlusPrefetch,
    const real* timeDerivativeMinusPrefetch) {
  // assert alignments
#ifndef NDEBUG
  assert(timeDerivativePlus != nullptr);
  assert(timeDerivativeMinus != nullptr);
  assert(((uintptr_t)timeDerivativePlus) % Alignment == 0);
  assert(((uintptr_t)timeDerivativeMinus) % Alignment == 0);
  assert(((uintptr_t)&qInterpolatedPlus[0]) % Alignment == 0);
  assert(((uintptr_t)&qInterpolatedMinus[0]) % Alignment == 0);
  assert(tensor::Q::size() == tensor::I::size());
#endif

  alignas(PagesizeStack) real degreesOfFreedomPlus[tensor::Q::size()];
  alignas(PagesizeStack) real degreesOfFreedomMinus[tensor::Q::size()];

  dynamicRupture::kernel::evaluateAndRotateQAtInterpolationPoints krnl = m_krnlPrototype;
  for (unsigned timeInterval = 0; timeInterval < ConvergenceOrder; ++timeInterval) {
#ifdef USE_STP
    m_timeKernel.evaluateAtTime(
        timeBasisFunctions[timeInterval], timeDerivativePlus, degreesOfFreedomPlus);
    m_timeKernel.evaluateAtTime(
        timeBasisFunctions[timeInterval], timeDerivativeMinus, degreesOfFreedomMinus);
#else
    m_timeKernel.computeTaylorExpansion(
        timePoints[timeInterval], 0.0, timeDerivativePlus, degreesOfFreedomPlus);
    m_timeKernel.computeTaylorExpansion(
        timePoints[timeInterval], 0.0, timeDerivativeMinus, degreesOfFreedomMinus);
#endif

    const real* plusPrefetch = (timeInterval < ConvergenceOrder - 1)
                                   ? &qInterpolatedPlus[timeInterval + 1][0]
                                   : timeDerivativePlusPrefetch;
    const real* minusPrefetch = (timeInterval < ConvergenceOrder - 1)
                                    ? &qInterpolatedMinus[timeInterval + 1][0]
                                    : timeDerivativeMinusPrefetch;

    krnl.QInterpolated = &qInterpolatedPlus[timeInterval][0];
    krnl.Q = degreesOfFreedomPlus;
    krnl.TinvT = godunovData->TinvT;
    krnl._prefetch.QInterpolated = plusPrefetch;
    krnl.execute(faceInfo.plusSide, 0);

    krnl.QInterpolated = &qInterpolatedMinus[timeInterval][0];
    krnl.Q = degreesOfFreedomMinus;
    krnl.TinvT = godunovData->TinvT;
    krnl._prefetch.QInterpolated = minusPrefetch;
    krnl.execute(faceInfo.minusSide, faceInfo.faceRelation);
  }
}

void seissol::kernels::DynamicRupture::batchedSpaceTimeInterpolation(
    DrConditionalPointersToRealsTable& table, seissol::parallel::runtime::StreamRuntime& runtime) {
#ifdef ACL_DEVICE

  real** degreesOfFreedomPlus{nullptr};
  real** degreesOfFreedomMinus{nullptr};

  auto resetDeviceCurrentState = [this](size_t counter) {
    for (size_t i = 0; i < counter; ++i) {
      this->device.api->popStackMemory();
    }
  };

  device.api->resetCircularStreamCounter();
  for (unsigned timeInterval = 0; timeInterval < ConvergenceOrder; ++timeInterval) {
    ConditionalKey timeIntegrationKey(*KernelNames::DrTime);
    if (table.find(timeIntegrationKey) != table.end()) {
      auto& entry = table[timeIntegrationKey];

      unsigned maxNumElements = (entry.get(inner_keys::Dr::Id::DerivativesPlus))->getSize();
      real** timeDerivativePlus =
          (entry.get(inner_keys::Dr::Id::DerivativesPlus))->getDeviceDataPtr();
      degreesOfFreedomPlus = (entry.get(inner_keys::Dr::Id::IdofsPlus))->getDeviceDataPtr();

      m_timeKernel.computeBatchedTaylorExpansion(timePoints[timeInterval],
                                                 0.0,
                                                 timeDerivativePlus,
                                                 degreesOfFreedomPlus,
                                                 maxNumElements,
                                                 runtime);

      real** timeDerivativeMinus =
          (entry.get(inner_keys::Dr::Id::DerivativesMinus))->getDeviceDataPtr();
      degreesOfFreedomMinus = (entry.get(inner_keys::Dr::Id::IdofsMinus))->getDeviceDataPtr();
      m_timeKernel.computeBatchedTaylorExpansion(timePoints[timeInterval],
                                                 0.0,
                                                 timeDerivativeMinus,
                                                 degreesOfFreedomMinus,
                                                 maxNumElements,
                                                 runtime);
    }

    // finish all previous work in the default stream
    size_t streamCounter{0};
    runtime.envMany(20, [&](void* stream, size_t i) {
      unsigned side = i / 5;
      unsigned faceRelation = i % 5;
      if (faceRelation == 4) {
        ConditionalKey plusSideKey(*KernelNames::DrSpaceMap, side);
        if (table.find(plusSideKey) != table.end()) {
          auto& entry = table[plusSideKey];
          const size_t numElements = (entry.get(inner_keys::Dr::Id::IdofsPlus))->getSize();

          auto krnl = m_gpuKrnlPrototype;
          real* tmpMem =
              (real*)(device.api->getStackMemory(krnl.TmpMaxMemRequiredInBytes * numElements));
          ++streamCounter;
          krnl.linearAllocator.initialize(tmpMem);
          krnl.streamPtr = stream;
          krnl.numElements = numElements;

          krnl.QInterpolated =
              (entry.get(inner_keys::Dr::Id::QInterpolatedPlus))->getDeviceDataPtr();
          krnl.extraOffset_QInterpolated = timeInterval * tensor::QInterpolated::size();
          krnl.Q = const_cast<const real**>(
              (entry.get(inner_keys::Dr::Id::IdofsPlus))->getDeviceDataPtr());
          krnl.TinvT =
              const_cast<const real**>((entry.get(inner_keys::Dr::Id::TinvT))->getDeviceDataPtr());
          krnl.execute(side, 0);
        }
      } else {
        ConditionalKey minusSideKey(*KernelNames::DrSpaceMap, side, faceRelation);
        if (table.find(minusSideKey) != table.end()) {
          auto& entry = table[minusSideKey];
          const size_t numElements = (entry.get(inner_keys::Dr::Id::IdofsMinus))->getSize();

          auto krnl = m_gpuKrnlPrototype;
          real* tmpMem =
              (real*)(device.api->getStackMemory(krnl.TmpMaxMemRequiredInBytes * numElements));
          ++streamCounter;
          krnl.linearAllocator.initialize(tmpMem);
          krnl.streamPtr = stream;
          krnl.numElements = numElements;

          krnl.QInterpolated =
              (entry.get(inner_keys::Dr::Id::QInterpolatedMinus))->getDeviceDataPtr();
          krnl.extraOffset_QInterpolated = timeInterval * tensor::QInterpolated::size();
          krnl.Q = const_cast<const real**>(
              (entry.get(inner_keys::Dr::Id::IdofsMinus))->getDeviceDataPtr());
          krnl.TinvT =
              const_cast<const real**>((entry.get(inner_keys::Dr::Id::TinvT))->getDeviceDataPtr());
          krnl.execute(side, faceRelation);
        }
      }
    });
    resetDeviceCurrentState(streamCounter);
  }
#else
  assert(false && "no implementation provided");
#endif
}

void seissol::kernels::DynamicRupture::flopsGodunovState(const DRFaceInformation& faceInfo,
                                                         long long& nonZeroFlops,
                                                         long long& hardwareFlops) {
  m_timeKernel.flopsTaylorExpansion(nonZeroFlops, hardwareFlops);

  // 2x evaluateTaylorExpansion
  nonZeroFlops *= 2;
  hardwareFlops *= 2;

  nonZeroFlops += dynamicRupture::kernel::evaluateAndRotateQAtInterpolationPoints::nonZeroFlops(
      faceInfo.plusSide, 0);
  hardwareFlops += dynamicRupture::kernel::evaluateAndRotateQAtInterpolationPoints::hardwareFlops(
      faceInfo.plusSide, 0);

  nonZeroFlops += dynamicRupture::kernel::evaluateAndRotateQAtInterpolationPoints::nonZeroFlops(
      faceInfo.minusSide, faceInfo.faceRelation);
  hardwareFlops += dynamicRupture::kernel::evaluateAndRotateQAtInterpolationPoints::hardwareFlops(
      faceInfo.minusSide, faceInfo.faceRelation);

  nonZeroFlops *= ConvergenceOrder;
  hardwareFlops *= ConvergenceOrder;
}
