#include "SlipLaw.h"

namespace seissol::dr::friction_law {

real SlipLaw::calcStateVariableHook(real sv0, real tmp, real timeIncrement, real sl0) {
  return sl0 / tmp * std::pow(tmp * sv0 / sl0, exp(-tmp * timeIncrement / sl0));
}
} // namespace seissol::dr::friction_law
