#include <Kernels/DeviceAux/PlasticityAux.h>
#include "hip/hip_runtime.h"
#include <init.h>
#include <cmath>

// NOTE: using c++14 because of cuda@10 which acts as a HIP backend for CI
namespace seissol {
namespace kernels {
namespace device {
namespace aux {
namespace plasticity {

template<typename T>
__forceinline__ __device__ typename std::enable_if<std::is_floating_point<T>::value,T>::type
squareRoot(T x) {
  return std::is_same<T, double>::value ? sqrt(x) : sqrtf(x);
}

template<typename T>
__forceinline__ __device__ typename std::enable_if<std::is_floating_point<T>::value,T>::type
maxValue(T x, T y) {
  return std::is_same<T, double>::value ? fmax(x, y) : fmaxf(x, y);
}

//--------------------------------------------------------------------------------------------------
__global__ void kernel_saveFirstMode(real *firstModes,
                                     const real **modalStressTensors) {
  constexpr unsigned numModesPerElement = tensor::Q::Shape[0];
  firstModes[hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x] = modalStressTensors[hipBlockIdx_x][hipThreadIdx_x * numModesPerElement];
}

void saveFirstModes(real *firstModes,
                    const real **modalStressTensors,
                    const size_t numElements,
                    void *streamPtr) {
  dim3 block(NUM_STRESS_COMPONENTS, 1, 1);
  dim3 grid(numElements, 1, 1);
  auto stream = reinterpret_cast<hipStream_t>(streamPtr);
  hipLaunchKernelGGL(kernel_saveFirstMode, grid, block, 0, stream, firstModes, modalStressTensors);
}


//--------------------------------------------------------------------------------------------------
__global__ void kernel_adjustDeviatoricTensors(real **nodalStressTensors,
                                               int *isAdjustableVector,
                                               const PlasticityData *plasticity,
                                               const double oneMinusIntegratingFactor) {
  real *elementTensors = nodalStressTensors[hipBlockIdx_x];
  real localStresses[NUM_STRESS_COMPONENTS];


  // NOTE: hipBlockDim_x == tensor::QStressNodal::Shape[0] i.e., num nodes
  constexpr unsigned numNodesPerElement = tensor::QStressNodal::Shape[0];
  #pragma unroll
  for (int i = 0; i < NUM_STRESS_COMPONENTS; ++i) {
    localStresses[i] = elementTensors[hipThreadIdx_x + numNodesPerElement * i];
  }

  // 2. Compute the mean stress for each node
  real meanStress = (localStresses[0] + localStresses[1] + localStresses[2]) / 3.0f;

  // 3. Compute deviatoric stress tensor
  #pragma unroll
  for (int i = 0; i < 3; ++i) {
    localStresses[i] -= meanStress;
  }

  // 4. Compute the second invariant for each node
  real tau = 0.5 * (localStresses[0] * localStresses[0] +
                    localStresses[1] * localStresses[1] +
                    localStresses[2] * localStresses[2]);
  tau += (localStresses[3] * localStresses[3] +
          localStresses[4] * localStresses[4] +
          localStresses[5] * localStresses[5]);
  tau = squareRoot(tau);

  // 5. Compute the plasticity criteria
  const real cohesionTimesCosAngularFriction = plasticity[hipBlockIdx_x].cohesionTimesCosAngularFriction;
  const real sinAngularFriction = plasticity[hipBlockIdx_x].sinAngularFriction;
  real taulim = cohesionTimesCosAngularFriction - meanStress * sinAngularFriction;
  taulim = maxValue(static_cast<real>(0.0), taulim);

  __shared__ int isAdjusted;
  if (hipThreadIdx_x == 0) {isAdjusted = static_cast<int>(false);}
  __syncthreads();

  // 6. Compute the yield factor
  real factor = 0.0;
  if (tau > taulim) {
    isAdjusted = static_cast<int>(true);
    factor = ((taulim / tau) - 1.0) * oneMinusIntegratingFactor;
  }

  // 7. Adjust deviatoric stress tensor if a node within a node exceeds the elasticity region
  __syncthreads();
  if (isAdjusted) {
    #pragma unroll
    for (int i = 0; i < NUM_STRESS_COMPONENTS; ++i) {
      elementTensors[hipThreadIdx_x + hipBlockIdx_x * i] = localStresses[i] * factor;
    }
  }

  if (hipThreadIdx_x == 0) {
    isAdjustableVector[hipBlockIdx_x] = isAdjusted;
  }
}

void adjustDeviatoricTensors(real **nodalStressTensors,
                             int *isAdjustableVector,
                             const PlasticityData *plasticity,
                             const double oneMinusIntegratingFactor,
                             const size_t numElements,
                             void *streamPtr) {
  constexpr unsigned numNodesPerElement = tensor::QStressNodal::Shape[0];
  dim3 block(numNodesPerElement, 1, 1);
  dim3 grid(numElements, 1, 1);
  auto stream = reinterpret_cast<hipStream_t>(streamPtr);
  hipLaunchKernelGGL(kernel_adjustDeviatoricTensors,
                     grid,
                     block,
                     0,
                     stream,
                     nodalStressTensors,
                     isAdjustableVector,
                     plasticity,
                     oneMinusIntegratingFactor);
}

//--------------------------------------------------------------------------------------------------
__global__ void kernel_adjustModalStresses(real** modalStressTensors,
                                           const real** nodalStressTensors,
                                           const real * inverseVandermondeMatrix,
                                           const int* isAdjustableVector) {
  if (isAdjustableVector[hipBlockIdx_x]) {

    // NOTE: hipBlockDim_x == init::QStressNodal::Shape[0]
    constexpr int numNodes = init::QStressNodal::Shape[0];
    constexpr size_t nodalTensorSize = numNodes * NUM_STRESS_COMPONENTS;
    __shared__ real shrMem[nodalTensorSize];

    real *modalTensor = modalStressTensors[hipBlockIdx_x];
    const real *nodalTensor = nodalStressTensors[hipBlockIdx_x];

    for (int n = 0; n < NUM_STRESS_COMPONENTS; ++n) {
      shrMem[hipThreadIdx_x + numNodes * n] = nodalTensor[hipThreadIdx_x + numNodes * n];
    }
    __syncthreads();

    // matrix multiply: (numNodes x numNodes) * (numNodes x NUM_STRESS_COMPONENTS)
    real accumulator[NUM_STRESS_COMPONENTS] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    real value = 0.0;
    // inverseVandermondeMatrix - square matrix
    for (int k = 0; k < numNodes; ++k) {
      value = inverseVandermondeMatrix[hipThreadIdx_x + numNodes * k];

      #pragma unroll
      for (int n = 0; n < NUM_STRESS_COMPONENTS; ++n) {
        accumulator[n] += value * shrMem[k + numNodes * n];
      }
    }

    constexpr unsigned numModesPerElement = init::Q::Shape[0];
    #pragma unroll
    for (int n = 0; n < NUM_STRESS_COMPONENTS; ++n) {
      modalTensor[hipThreadIdx_x + numModesPerElement * n] += accumulator[n];
    }
  }
}

void adjustModalStresses(real **modalStressTensors,
                         const real **nodalStressTensors,
                         const real *inverseVandermondeMatrix,
                         const int *isAdjustableVector,
                         const size_t numElements,
                         void *streamPtr) {
  constexpr unsigned numNodesPerElement = init::vInv::Shape[0];
  dim3 block(numNodesPerElement, 1, 1);
  dim3 grid(numElements, 1, 1);
  auto stream = reinterpret_cast<hipStream_t>(streamPtr);
  hipLaunchKernelGGL(kernel_adjustModalStresses,
                     grid,
                     block,
                     0,
                     stream,
                     modalStressTensors,
                     nodalStressTensors,
                     inverseVandermondeMatrix,
                     isAdjustableVector);
}

//--------------------------------------------------------------------------------------------------
__global__ void kernel_computePstrains(real **pstrains,
                                       const int* isAdjustableVector,
                                       const real** modalStressTensors,
                                       const real* firsModes,
                                       const PlasticityData* plasticity,
                                       const double oneMinusIntegratingFactor,
                                       const double timeStepWidth,
                                       const double T_v,
                                       const size_t numElements) {
  // compute element id
  size_t index = hipThreadIdx_y + hipBlockDim_x * hipBlockDim_y;
  if ((isAdjustableVector[index]) && (index < numElements)) {
    // NOTE: Six threads (x-dimension) work on the same element.

    // get local data
    real *localPstrains = pstrains[index];
    const real *localModalTensor = modalStressTensors[index];
    const real *localFirstMode = &firsModes[NUM_STRESS_COMPONENTS * index];
    const PlasticityData *localData = &plasticity[index];

    constexpr unsigned numModesPerElement = init::Q::Shape[0];
    real factor = localData->mufactor / (T_v * oneMinusIntegratingFactor);
    real duDtPstrain = factor * (localFirstMode[hipThreadIdx_x] - localModalTensor[hipThreadIdx_x * numModesPerElement]);
    localPstrains[hipThreadIdx_x] += timeStepWidth * duDtPstrain;

    __shared__ real squaredDuDtPstrains[NUM_STRESS_COMPONENTS];
    real coefficient = hipThreadIdx_x < 3 ? 0.5f : 1.0f;
    squaredDuDtPstrains[hipThreadIdx_x] = coefficient * duDtPstrain * duDtPstrain;
    __syncthreads();

    if (hipThreadIdx_x == 0) {
      real sum = 0.0;

      #pragma unroll
      for (int i = 0; i < NUM_STRESS_COMPONENTS; ++i) {
        sum += squaredDuDtPstrains[i];
      }
      localPstrains[6] += (timeStepWidth * squareRoot(duDtPstrain));
    }
  }
}


void computePstrains(real **pstrains,
                     const int *isAdjustableVector,
                     const real **modalStressTensors,
                     const real *firsModes,
                     const PlasticityData *plasticity,
                     const double oneMinusIntegratingFactor,
                     const double timeStepWidth,
                     const double T_v,
                     const size_t numElements,
                     void *streamPtr) {
  dim3 block(NUM_STRESS_COMPONENTS, 32, 1);
  size_t numBlocks = (numElements + block.y - 1) / block.y;
  dim3 grid(numBlocks, 1, 1);
  auto stream = reinterpret_cast<hipStream_t>(streamPtr);

  hipLaunchKernelGGL(kernel_computePstrains,
                     grid,
                     block,
                     0,
                     stream,
                     pstrains,
                     isAdjustableVector,
                     modalStressTensors,
                     firsModes,
                     plasticity,
                     oneMinusIntegratingFactor,
                     timeStepWidth,
                     T_v,
                     numElements);
}

} // namespace plasticity
} // namespace aux
} // namespace device
} // namespace kernels
} // namespace seissol