# SPDX-FileCopyrightText: 2019-2024 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause


# FIXME: forced to be on
# option(HDF5 "Use HDF5 library for data output" ON)
set(HDF5 ON)

option(NETCDF "Use netcdf library for mesh input" ON)

set(GRAPH_PARTITIONING_LIBS "parmetis" CACHE STRING "Graph partitioning library for mesh partitioning")
set(GRAPH_PARTITIONING_LIB_OPTIONS none parmetis parhip ptscotch)
set_property(CACHE GRAPH_PARTITIONING_LIBS PROPERTY STRINGS ${GRAPH_PARTITIONING_LIB_OPTIONS})

# FIXME: forced to be on
# option(MPI "Use MPI parallelization" ON)
set(MPI ON)

# FIXME: forced to be on
# option(OPENMP "Use OpenMP parallelization" ON)
set(OPENMP ON)

option(ASAGI "Use asagi for material input" OFF)
option(MEMKIND "Use memkind library for hbw memory support" OFF)
option(LIKWID "Link with the likwid marker interface for proxy" OFF)

option(INTEGRATE_QUANTITIES "Compute cell-averaged integrated velocity and stress components (currently breaks compilation)" OFF)
option(ADDRESS_SANITIZER_DEBUG "Use address sanitzer in debug mode" OFF)

option(TESTING "Compile unit tests" OFF)
option(TESTING_GENERATED "Include kernel tests generated by yateto" OFF)
option(COVERAGE "Generate targed for code coverage using lcob" OFF)
set(TESTING_COMMAND "" CACHE STRING "Prefix to the test binary, so that CMake can list the tests")

#Seissol specific
set(ORDER 6 CACHE STRING "Convergence order")  # must be INT type, by cmake-3.16 accepts only STRING
set(ORDER_OPTIONS 2 3 4 5 6 7 8)
set_property(CACHE ORDER PROPERTY STRINGS ${ORDER_OPTIONS})

set(NUMBER_OF_MECHANISMS 0 CACHE STRING "Number of mechanisms")

set(EQUATIONS "elastic" CACHE STRING "Equation set used")
set(EQUATIONS_OPTIONS elastic anisotropic viscoelastic viscoelastic2 poroelastic acoustic)
set_property(CACHE EQUATIONS PROPERTY STRINGS ${EQUATIONS_OPTIONS})


set(HOST_ARCH "hsw" CACHE STRING "Type of host architecture")
set(HOST_ARCH_OPTIONS noarch wsm snb hsw knc knl skx naples rome milan bergamo thunderx2t99 power9 a64fx neon sve128 sve256 sve512 sve1024 sve2048 apple-m1 apple-m2)
# size of a vector registers in bytes for a given architecture
set(HOST_ARCH_ALIGNMENT   16  16  32  32  64  64  64     32   32    32      64       16     16     256     16     16     32     64     128     256      128      128)
set(HOST_ARCH_VECTORSIZE  16  16  32  32  64  64  64     32   32    32      64       16     16      64     16     16     32     64     128     256       16       16)
set_property(CACHE HOST_ARCH PROPERTY STRINGS ${HOST_ARCH_OPTIONS})


set(DEVICE_BACKEND "none" CACHE STRING "Type of GPU backend")
set(DEVICE_BACKEND_OPTIONS none cuda hip hipsycl oneapi)
set_property(CACHE DEVICE_BACKEND PROPERTY STRINGS ${DEVICE_BACKEND_OPTIONS})


set(DEVICE_ARCH "none" CACHE STRING "Type of GPU architecture")
set(DEVICE_ARCH_OPTIONS none
        sm_60 sm_61 sm_62 sm_70 sm_71 sm_75 sm_80 sm_86 sm_87 sm_89 sm_90 sm_100   # Nvidia
        gfx900 gfx906 gfx908 gfx90a gfx942 gfx1010 gfx1030 gfx1100 gfx1101 gfx1102 # AMD
        bdw skl dg1 acm_g10 acm_g11 acm_g12 pvc Gen8 Gen9 Gen11 Gen12LP)           # Intel
set_property(CACHE DEVICE_ARCH PROPERTY STRINGS ${DEVICE_ARCH_OPTIONS})

set(PRECISION "double" CACHE STRING "Type of floating point precision, namely: double/single")
set(PRECISION_OPTIONS single double)
set_property(CACHE PRECISION PROPERTY STRINGS ${PRECISION_OPTIONS})


set(PLASTICITY_METHOD "nb" CACHE STRING "Plasticity method: nb (nodal basis) is faster, ip (interpolation points) possibly more accurate. Recommended: nb")
set(PLASTICITY_OPTIONS nb ip)
set_property(CACHE PLASTICITY_METHOD PROPERTY STRINGS ${PLASTICITY_OPTIONS})


set(DR_QUAD_RULE "stroud" CACHE STRING "Dynamic Rupture quadrature rule")
set(DR_QUAD_RULE_OPTIONS stroud dunavant)
set_property(CACHE DR_QUAD_RULE PROPERTY STRINGS ${DR_QUAD_RULE_OPTIONS})


set(NUMBER_OF_FUSED_SIMULATIONS 1 CACHE STRING "A number of fused simulations")

set(MEMORY_LAYOUT "auto" CACHE FILEPATH "A file with a specific memory layout or auto")

option(NUMA_AWARE_PINNING "Use libnuma to pin threads to correct NUMA nodes" ON)

option(SHARED "Build SeisSol as shared library" OFF)

option(PROXY_PYBINDING "Enable pybind11 for proxy (everything will be compiled with -fPIC)" OFF)

# FIXME: currently unused
#set(LOG_LEVEL "warning" CACHE STRING "Log level for the code")
#set(LOG_LEVEL_OPTIONS "debug" "info" "warning" "error")
#set_property(CACHE LOG_LEVEL PROPERTY STRINGS ${LOG_LEVEL_OPTIONS})

set(LOG_LEVEL_MASTER "info" CACHE STRING "Log level for the code")
set(LOG_LEVEL_MASTER_OPTIONS "debug" "info" "warning" "error")
set_property(CACHE LOG_LEVEL_MASTER PROPERTY STRINGS ${LOG_LEVEL_MASTER_OPTIONS})


set(GEMM_TOOLS_LIST "auto" CACHE STRING "GEMM tool(s) used for CPU code generation")
set(GEMM_TOOLS_OPTIONS "auto" "none"
        "LIBXSMM,PSpaMM" "LIBXSMM" "MKL" "OpenBLAS" "BLIS" "PSpaMM" "Eigen" "LIBXSMM,PSpaMM,GemmForge" "Eigen,GemmForge"
        "LIBXSMM_JIT,PSpaMM" "LIBXSMM_JIT" "LIBXSMM_JIT,PSpaMM,GemmForge")
set_property(CACHE GEMM_TOOLS_LIST PROPERTY STRINGS ${GEMM_TOOLS_OPTIONS})

#-------------------------------------------------------------------------------
# ------------------------------- ERROR CHECKING -------------------------------
#-------------------------------------------------------------------------------
function(check_parameter parameter_name value options)

    list(FIND options ${value} INDEX)

    set(WRONG_PARAMETER -1)
    if (${INDEX} EQUAL ${WRONG_PARAMETER})
        message(FATAL_ERROR "${parameter_name} is wrong. Specified \"${value}\". Allowed: ${options}")
    endif()

endfunction()


check_parameter("ORDER" ${ORDER} "${ORDER_OPTIONS}")
check_parameter("HOST_ARCH" ${HOST_ARCH} "${HOST_ARCH_OPTIONS}")
check_parameter("DEVICE_BACKEND" ${DEVICE_BACKEND} "${DEVICE_BACKEND_OPTIONS}")
check_parameter("DEVICE_ARCH" ${DEVICE_ARCH} "${DEVICE_ARCH_OPTIONS}")
check_parameter("EQUATIONS" ${EQUATIONS} "${EQUATIONS_OPTIONS}")
check_parameter("PRECISION" ${PRECISION} "${PRECISION_OPTIONS}")
check_parameter("PLASTICITY_METHOD" ${PLASTICITY_METHOD} "${PLASTICITY_OPTIONS}")
# check_parameter("LOG_LEVEL" ${LOG_LEVEL} "${LOG_LEVEL_OPTIONS}")
check_parameter("LOG_LEVEL_MASTER" ${LOG_LEVEL_MASTER} "${LOG_LEVEL_MASTER_OPTIONS}")

# deduce GEMM_TOOLS_LIST based on the host arch
if (GEMM_TOOLS_LIST STREQUAL "auto")
    message(STATUS "Inferring GEMM_TOOLS_LIST for ${HOST_ARCH}")

    set(SUPPORT_PSPAMM hsw knl skx naples rome milan bergamo thunderx2t99 a64fx neon sve128 sve256 sve512 sve1024 sve2048 apple-m1)
    set(SUPPORT_LIBXSMM wsm snb hsw knc knl skx naples rome milan bergamo)
    set(SUPPORT_LIBXSMM_JIT wsm snb hsw knc knl skx naples rome milan bergamo thunderx2t99 power9 a64fx neon sve128 sve256 sve512 apple-m1)

    set(AUTO_GEMM_TOOLS_LIST)

    if (${HOST_ARCH} IN_LIST SUPPORT_LIBXSMM_JIT)
        find_package(LIBXSMM 1.17 QUIET)
        find_package(BLAS QUIET)
        if (LIBXSMM_FOUND AND BLAS_FOUND)
            message(STATUS "Found LIBXSMM_JIT, and it is supported")
            list(APPEND AUTO_GEMM_TOOLS_LIST "LIBXSMM_JIT")
        else()
            message(STATUS "LIBXSMM_JIT would be supported, but it was not found")
        endif()
    endif()
    if (${HOST_ARCH} IN_LIST SUPPORT_LIBXSMM)
        find_package(Libxsmm_executable QUIET)
        if (Libxsmm_executable_FOUND)
            message(STATUS "Found LIBXSMM, and it is supported")
            list(APPEND AUTO_GEMM_TOOLS_LIST "LIBXSMM")
        else()
            message(STATUS "LIBXSMM would be supported, but it was not found")
        endif()
    endif()
    if (${HOST_ARCH} IN_LIST SUPPORT_PSPAMM)
        find_package(PSpaMM QUIET)
        if (PSpaMM_FOUND)
            message(STATUS "Found PSpaMM, and it is supported")
            list(APPEND AUTO_GEMM_TOOLS_LIST "PSpaMM")
        else()
            message(STATUS "PSpaMM would be supported, but it was not found")
        endif()
    endif()

    # Eigen will always be there
    # But it won't give us much, if there's any of the above generators found
    if (NOT AUTO_GEMM_TOOLS_LIST)
        message(STATUS "Adding Eigen as last resort option")
        list(APPEND AUTO_GEMM_TOOLS_LIST "Eigen")
    endif()

    list(JOIN AUTO_GEMM_TOOLS_LIST "," GEMM_TOOLS_LIST)
endif()

if (NOT ${DEVICE_BACKEND} STREQUAL "none")
    message(STATUS "GPUs are enabled, adding GemmForge (and ChainForge, if available)")
    set(GEMM_TOOLS_LIST "${GEMM_TOOLS_LIST},GemmForge")
    set(WITH_GPU on)
    # note: GPU builds currently don't support CPU-only execution (hence the following message)
    message(STATUS "Compiling SeisSol for GPU")
else()
    set(WITH_GPU off)
    message(STATUS "Compiling SeisSol for CPU")
endif()
message(STATUS "GEMM_TOOLS are: ${GEMM_TOOLS_LIST}")

if (DEVICE_ARCH MATCHES "sm_*")
    set(DEVICE_VENDOR "nvidia")
    set(IS_NVIDIA_OR_AMD ON)
elseif(DEVICE_ARCH MATCHES "gfx*")
    set(DEVICE_VENDOR "amd")
    set(IS_NVIDIA_OR_AMD ON)
else()
    # TODO(David): adjust as soon as we add support for more vendors
    set(DEVICE_VENDOR "intel")
    set(IS_NVIDIA_OR_AMD OFF)
endif()

if (WITH_GPU)
    # the premultiplication was only so far demonstrated to be efficient on AMD+NVIDIA HW; enable on others by demand
    option(PREMULTIPLY_FLUX "Merge device flux matrices (recommended for AMD and Nvidia GPUs)" ${IS_NVIDIA_OR_AMD})

    # experimental kernels should stay experimental; they've only be sort of tested on NV+AMD hardware for now
    option(DEVICE_EXPERIMENTAL_EXPLICIT_KERNELS "Enable experimental explicitly-written kernels" ${IS_NVIDIA_OR_AMD})
endif()


# check compute sub architecture (relevant only for GPU)
if (NOT ${DEVICE_ARCH} STREQUAL "none")
    if (${DEVICE_BACKEND} STREQUAL "none")
        message(FATAL_ERROR "DEVICE_BACKEND is not provided for ${DEVICE_ARCH}")
    endif()

    if (${DEVICE_ARCH} MATCHES "sm_*")
        set(ALIGNMENT  64)
        set(VECTORSIZE 128)
    elseif(${DEVICE_ARCH} MATCHES "gfx*")
        set(ALIGNMENT 128)
        set(VECTORSIZE 256)
    else()
        set(ALIGNMENT 128)
        set(VECTORSIZE 32)
        message(STATUS "Assume device alignment = 128, for DEVICE_ARCH=${DEVICE_ARCH}")
    endif()

    # for now
    set(VECTORSIZE ${ALIGNMENT})
else()
    list(FIND HOST_ARCH_OPTIONS ${HOST_ARCH} INDEX)
    list(GET HOST_ARCH_ALIGNMENT ${INDEX} ALIGNMENT)
    list(GET HOST_ARCH_VECTORSIZE ${INDEX} VECTORSIZE)
    set(DEVICE_BACKEND "none")
endif()

message(STATUS "Memory alignment has been set to ${ALIGNMENT} B.")
message(STATUS "Vector size has been set to ${VECTORSIZE} B.")

# check NUMBER_OF_MECHANISMS
if ((NOT "${EQUATIONS}" MATCHES "viscoelastic.?") AND ${NUMBER_OF_MECHANISMS} GREATER 0)
    message(FATAL_ERROR "${EQUATIONS} does not support a NUMBER_OF_MECHANISMS > 0.")
endif()

if ("${EQUATIONS}" MATCHES "viscoelastic.?" AND ${NUMBER_OF_MECHANISMS} LESS 1)
    message(FATAL_ERROR "${EQUATIONS} needs a NUMBER_OF_MECHANISMS > 0.")
endif()


# derive a byte representation of real numbers
if ("${PRECISION}" STREQUAL "double")
    set(REAL_SIZE_IN_BYTES 8)
elseif ("${PRECISION}" STREQUAL "single")
    set(REAL_SIZE_IN_BYTES 4)
endif()


# check NUMBER_OF_FUSED_SIMULATIONS
math(EXPR IS_ALIGNED_MULT_SIMULATIONS 
        "${NUMBER_OF_FUSED_SIMULATIONS} % (${ALIGNMENT} / ${REAL_SIZE_IN_BYTES})")

if (NOT ${NUMBER_OF_FUSED_SIMULATIONS} EQUAL 1 AND NOT ${IS_ALIGNED_MULT_SIMULATIONS} EQUAL 0)
    math(EXPR FACTOR "${ALIGNMENT} / ${REAL_SIZE_IN_BYTES}")
    message(FATAL_ERROR "a number of fused simulations must be multiple of ${FACTOR}")
endif()

#-------------------------------------------------------------------------------
# -------------------- COMPUTE/ADJUST ADDITIONAL PARAMETERS --------------------
#-------------------------------------------------------------------------------

# generate an internal representation of an architecture type which is used in seissol
string(SUBSTRING ${PRECISION} 0 1 PRECISION_PREFIX)
if (${PRECISION} STREQUAL "double")
    set(HOST_ARCH_STR "d${HOST_ARCH}")
    set(DEVICE_ARCH_STR "d${DEVICE_ARCH}")
elseif(${PRECISION} STREQUAL "single")
    set(HOST_ARCH_STR "s${HOST_ARCH}")
    set(DEVICE_ARCH_STR "s${DEVICE_ARCH}")
endif()

if (${DEVICE_ARCH} STREQUAL "none")
    set(DEVICE_ARCH_STR "none")
endif()


function(cast_log_level_to_int log_level_str log_level_int)
  if (${log_level_str} STREQUAL "debug")
    set(${log_level_int} 3 PARENT_SCOPE)
  elseif (${log_level_str} STREQUAL "info")
    set(${log_level_int} 2 PARENT_SCOPE)
  elseif (${log_level_str} STREQUAL "warning")
    set(${log_level_int} 1 PARENT_SCOPE)
  elseif (${log_level_str} STREQUAL "error")
    set(${log_level_int} 0 PARENT_SCOPE)
  endif()
endfunction()

cast_log_level_to_int(LOG_LEVEL LOG_LEVEL)
cast_log_level_to_int(LOG_LEVEL_MASTER LOG_LEVEL_MASTER)

if (PROXY_PYBINDING)
    set(EXTRA_CXX_FLAGS -fPIC)

    # Note: ENABLE_PIC_COMPILATION can be used to signal other sub-modules
    # generate position independent code
    set(ENABLE_PIC_COMPILATION ON)
endif()
