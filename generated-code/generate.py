#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2019-2024 SeisSol Group
#
# SPDX-License-Identifier: BSD-3-Clause


# @file
# This file is part of SeisSol.
#
# @author Carsten Uphoff (c.uphoff AT tum.de)
# @author Sebastian Wolf (wolf.sebastian AT tum.de)
#

import argparse
import importlib.util
import os
import sys

import kernels.dynamic_rupture
import kernels.general
import kernels.memlayout
import kernels.nodalbc
import kernels.plasticity
import kernels.point
import kernels.surface_displacement
import kernels.vtkproject
from yateto import (Generator, NamespacedGenerator, gemm_configuration,
                    useArchitectureIdentifiedBy)
from yateto.ast.cost import (BoundingBoxCostEstimator,
                             FusedGemmsBoundingBoxCostEstimator)
from yateto.gemm_configuration import Eigen, GeneratorCollection


def main():

    cmdLineParser = argparse.ArgumentParser()
    cmdLineParser.add_argument("--equations")
    cmdLineParser.add_argument("--matricesDir")
    cmdLineParser.add_argument("--outputDir")
    cmdLineParser.add_argument("--host_arch")
    cmdLineParser.add_argument("--device_backend", default=None)
    cmdLineParser.add_argument("--device_arch", default=None)
    cmdLineParser.add_argument("--order", type=int)
    cmdLineParser.add_argument("--numberOfMechanisms", type=int)
    cmdLineParser.add_argument("--memLayout")
    cmdLineParser.add_argument("--multipleSimulations", type=int)
    cmdLineParser.add_argument("--PlasticityMethod")
    cmdLineParser.add_argument("--gemm_tools")
    cmdLineParser.add_argument("--drQuadRule")
    cmdLineParser.add_argument("--enable_premultiply_flux", action="store_true")
    cmdLineParser.add_argument(
        "--disable_premultiply_flux",
        dest="enable_premultiply_flux",
        action="store_false",
    )
    cmdLineParser.add_argument("--executable_libxsmm", default="")
    cmdLineParser.add_argument("--executable_pspamm", default="")
    cmdLineParser.set_defaults(enable_premultiply_flux=False)
    cmdLineArgs = cmdLineParser.parse_args()

    # derive the compute platform
    gpu_platforms = ["cuda", "hip", "hipsycl", "acpp", "oneapi"]
    targets = ["gpu", "cpu"] if cmdLineArgs.device_backend in gpu_platforms else ["cpu"]

    if cmdLineArgs.device_backend == "none":
        arch = useArchitectureIdentifiedBy(cmdLineArgs.host_arch)
    else:
        arch = useArchitectureIdentifiedBy(
            cmdLineArgs.host_arch,
            cmdLineArgs.device_arch,
            cmdLineArgs.device_backend,
        )

    # pick up the gemm tools defined by the user
    gemm_tool_list = cmdLineArgs.gemm_tools.replace(" ", "").split(",")
    gemm_generators = []

    for tool in gemm_tool_list:
        if hasattr(gemm_configuration, tool):
            specific_gemm_class = getattr(gemm_configuration, tool)
            # take executable arguments, but only if they are not empty
            if (
                specific_gemm_class is gemm_configuration.LIBXSMM
                and cmdLineArgs.executable_libxsmm != ""
            ):
                gemm_generators.append(
                    specific_gemm_class(arch, cmdLineArgs.executable_libxsmm)
                )
            elif (
                specific_gemm_class is gemm_configuration.PSpaMM
                and cmdLineArgs.executable_pspamm != ""
            ):
                gemm_generators.append(
                    specific_gemm_class(arch, cmdLineArgs.executable_pspamm)
                )
            else:
                gemm_generators.append(specific_gemm_class(arch))
        elif tool.strip().lower() != "none":
            print(
                'YATETO::ERROR: unknown "{}" GEMM tool. '
                "Please, refer to the documentation".format(tool)
            )
            sys.exit("failure")

    cost_estimators = BoundingBoxCostEstimator
    if "gpu" in targets:
        chainforge_spec = importlib.util.find_spec("chainforge")
        if chainforge_spec is not None:
            chainforge_spec.loader.load_module()
            cost_estimators = FusedGemmsBoundingBoxCostEstimator
        else:
            print("WARNING: ChainForge was not found. Falling back to GemmForge.")

    subfolders = []

    equationsSpec = importlib.util.find_spec(
        f"kernels.equations.{cmdLineArgs.equations}"
    )
    if equationsSpec is None:
        raise RuntimeError("Could not find kernels for " + cmdLineArgs.equations)
    equations = equationsSpec.loader.load_module()

    equation_class = equations.EQUATION_CLASS

    def generate_equation(subfolders, equation, order):
        precision = "double" if cmdLineArgs.host_arch[0] == "d" else "single"

        if cmdLineArgs.memLayout == "auto":
            # TODO(Lukas) Don't hardcode this
            env = {
                "equations": cmdLineArgs.equations,
                "order": order,
                "arch": cmdLineArgs.host_arch,
                "device_arch": cmdLineArgs.device_arch,
                "multipleSimulations": cmdLineArgs.multipleSimulations,
                "targets": targets,
            }
            mem_layout = kernels.memlayout.guessMemoryLayout(env)
        else:
            mem_layout = cmdLineArgs.memLayout

        cmdArgsDict = vars(cmdLineArgs)
        cmdArgsDict["memLayout"] = mem_layout

        adg = equation(**cmdArgsDict)

        include_tensors = set()
        generator = Generator(arch)

        # Equation-specific kernels
        adg.addInit(generator)
        adg.addLocal(generator, targets)
        adg.addNeighbor(generator, targets)
        adg.addTime(generator, targets)
        adg.add_include_tensors(include_tensors)

        kernels.vtkproject.addKernels(generator, adg, cmdLineArgs.matricesDir, targets)
        kernels.vtkproject.includeTensors(cmdLineArgs.matricesDir, include_tensors)

        # Common kernels
        include_tensors.update(
            kernels.dynamic_rupture.addKernels(
                NamespacedGenerator(generator, namespace="dynamicRupture"),
                adg,
                cmdLineArgs.matricesDir,
                cmdLineArgs.drQuadRule,
                targets,
            )
        )

        kernels.plasticity.addKernels(
            generator,
            adg,
            cmdLineArgs.matricesDir,
            cmdLineArgs.PlasticityMethod,
            targets,
        )
        kernels.nodalbc.addKernels(
            generator,
            adg,
            include_tensors,
            cmdLineArgs.matricesDir,
            cmdLineArgs,
            targets,
        )
        kernels.surface_displacement.addKernels(
            generator, adg, include_tensors, targets
        )
        kernels.point.addKernels(generator, adg)

        outputDirName = f"equation-{adg.name()}-{order}-{precision}"
        trueOutputDir = os.path.join(cmdLineArgs.outputDir, outputDirName)
        if not os.path.exists(trueOutputDir):
            os.mkdir(trueOutputDir)

        subfolders += [outputDirName]

        # Generate code
        gemmTools = GeneratorCollection(gemm_generators)
        generator.generate(
            outputDir=trueOutputDir,
            namespace="seissol",
            gemm_cfg=gemmTools,
            cost_estimator=cost_estimators,
            include_tensors=include_tensors,
        )

    def generate_general(subfolders):
        # we use always use double here,
        # since these kernels are only used in the initialization
        arch = useArchitectureIdentifiedBy("d" + cmdLineArgs.host_arch[1:])

        outputDir = os.path.join(cmdLineArgs.outputDir, "general")
        if not os.path.exists(outputDir):
            os.mkdir(outputDir)

        subfolders += ["general"]

        # for now, enforce Eigen as a code generator here...
        # ...until we have a shared subroutine cache
        generator = Generator(arch)
        kernels.general.addStiffnessTensor(generator)
        generator.generate(
            outputDir=outputDir,
            namespace="seissol_general",
            gemm_cfg=GeneratorCollection([Eigen(arch)]),
            cost_estimator=cost_estimators,
            include_tensors=kernels.general.includeMatrices(cmdLineArgs.matricesDir),
        )

    def forward_files(filename):
        with open(os.path.join(cmdLineArgs.outputDir, filename), "w") as file:
            file.writelines(["// IWYU pragma: begin_exports\n"])
            file.writelines(
                [
                    f'#include "{os.path.join(folder, filename)}"\n'
                    for folder in subfolders
                ]
            )
            file.writelines(["// IWYU pragma: end_exports\n"])

    generate_equation(subfolders, equation_class, cmdLineArgs.order)
    generate_general(subfolders)

    forward_files("init.h")
    forward_files("kernel.h")
    forward_files("subroutine.h")
    forward_files("tensor.h")
    forward_files("init.cpp")
    forward_files("kernel.cpp")
    forward_files("subroutine.cpp")
    forward_files("tensor.cpp")
    forward_files("test-kernel.cpp")
    forward_files("gpulike_subroutine.cpp")


if __name__ == "__main__":
    main()
