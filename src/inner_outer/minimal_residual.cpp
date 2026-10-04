#include "inner_outer_common.hpp"

namespace methods::inner_outer {

common::OutputResult RunMinimalResidual(const common::InputParams& params, const int64_t num_threads, const int64_t inner_iterations) {
  return Run(params, num_threads, false, inner_iterations);
}

common::OutputResult RunLocallyOptimalMinimalResidual(const common::InputParams& params, const int64_t num_threads, const int64_t inner_iterations) {
  return Run(params, num_threads, true, inner_iterations);
}

}  // namespace methods::inner_outer
