#include "inner_outer_common.hpp"

namespace methods::inner_outer {

common::OutputResult RunMinimalResidual(const common::InputParams& params, const int64_t num_threads) {
  return Run(params, num_threads, false);
}

common::OutputResult RunLocallyOptimalMinimalResidual(const common::InputParams& params, const int64_t num_threads) {
  return Run(params, num_threads, true);
}

}  // namespace methods::inner_outer
