// SPDX-FileCopyrightText: 2022-2024 SeisSol Group
//
// SPDX-License-Identifier: BSD-3-Clause
// SPDX-LicenseComments: Full text under /LICENSE and /LICENSES/
//
// SPDX-FileContributor: Author lists in /AUTHORS and /CITATION.cff

#include "Parallel/AcceleratorDevice.h"
#include "Parallel/MPI.h"
#include "utils/logger.h"
#include <sstream>
#include <string>

namespace seissol {
void AcceleratorDevice::bindSyclDevice(int deviceId) {
  try {
#ifdef __DPCPP_COMPILER
    syclDevice = sycl::device(sycl::gpu_selector_v);
#elif defined(SYCL_PLATFORM_NVIDIA)
    syclDevice = sycl::make_device<sycl::backend::cuda>(deviceId);
#elif defined(SYCL_PLATFORM_AMD)
    syclDevice = sycl::make_device<sycl::backend::hip>(deviceId);
#else
    syclDevice = sycl::device(sycl::cpu_selector());
#endif // __DPCPP_COMPILER
  } catch (const sycl::exception& err) {
    {
      std::ostringstream warn;
      warn << "SYCL error: " << err.what();
      warnMessages.push_back(warn.str());
    }
#ifdef __DPCPP_COMPILER
    syclDevice = sycl::device(sycl::cpu_selector_v);
#else
    syclDevice = sycl::device(sycl::cpu_selector());
#endif
  }

  std::ostringstream info;
  info << "SYCL device for Dynamic Rupture and Point sources (GPU: " << std::boolalpha
       << syclDevice.is_gpu() << "): " << syclDevice.get_info<sycl::info::device::name>();
  infoMessages.push_back(info.str());

#if (defined(ACPP_EXT_COARSE_GRAINED_EVENTS) || defined(SYCL_EXT_ACPP_COARSE_GRAINED_EVENTS)) &&   \
    !defined(SEISSOL_KERNELS_SYCL)
  const sycl::property_list property{sycl::property::queue::in_order(),
                                     sycl::property::queue::AdaptiveCpp_coarse_grained_events()};
#elif defined(HIPSYCL_EXT_COARSE_GRAINED_EVENTS) && !defined(SEISSOL_KERNELS_SYCL)
  const sycl::property_list property{sycl::property::queue::in_order(),
                                     sycl::property::queue::hipSYCL_coarse_grained_events()};
#else
  const sycl::property_list property{sycl::property::queue::in_order()};
#endif

  syclDefaultQueue = sycl::queue(syclDevice, property);
}

void AcceleratorDevice::bindNativeDevice(int deviceId) {
  device::DeviceInstance& device = device::DeviceInstance::getInstance();
  {
    std::ostringstream info;
    info << "Device API: " << device.api->getApiName();
    infoMessages.push_back(info.str());
  }
  {
    std::ostringstream info;
    info << "Device (rank=0): " << device.api->getDeviceName(deviceId);
    infoMessages.push_back(info.str());
  }
  device.api->setDevice(deviceId);
}

void AcceleratorDevice::printInfo() {
  for (const auto& warn : warnMessages) {
    logWarning() << warn.c_str();
  }
  for (const auto& info : infoMessages) {
    logInfo() << info.c_str();
  }
}
} // namespace seissol
