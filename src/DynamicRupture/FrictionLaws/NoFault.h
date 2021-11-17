#ifndef SEISSOL_NOFAULT_H
#define SEISSOL_NOFAULT_H

#include "BaseFrictionLaw.h"

namespace seissol::dr::friction_law {
/**
 * No friction computation
 * input stress XYStressGP, XZStressGP equals output XYTractionResultGP, XZTractionResultGP
 */
class NoFault : public BaseFrictionLaw {
  public:
  using BaseFrictionLaw::BaseFrictionLaw;

  virtual void
      evaluate(seissol::initializers::Layer& layerData,
               seissol::initializers::DynamicRupture* dynRup,
               real (*QInterpolatedPlus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
               real (*QInterpolatedMinus)[CONVERGENCE_ORDER][tensor::QInterpolated::size()],
               real fullUpdateTime,
               double timeWeights[CONVERGENCE_ORDER]) override;
};
} // namespace seissol::dr::friction_law
#endif // SEISSOL_NOFAULT_H