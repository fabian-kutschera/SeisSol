#ifndef SEISSOL_MATH_H
#define SEISSOL_MATH_H

#include <stdexcept>
#include <cmath>

namespace seissol::dr::aux {
constexpr int numGaussPoints2d(const int Order) {
  return std::pow(Order + 1, 2);
}

template <class TupleT, class F, std::size_t... I>
constexpr F forEachImpl(TupleT&& tuple, F&& functor, std::index_sequence<I...>) {
  return (void)std::initializer_list<int>{
             (std::forward<F>(functor)(std::get<I>(std::forward<TupleT>(tuple)), I), 0)...},
         functor;
}

template <typename TupleT, typename F>
constexpr F forEach(TupleT&& tuple, F&& functor) {
  return forEachImpl(
      std::forward<TupleT>(tuple),
      std::forward<F>(functor),
      std::make_index_sequence<std::tuple_size<std::remove_reference_t<TupleT>>::value>{});
}
} // namespace seissol::dr::aux

#endif // SEISSOL_MATH_H
