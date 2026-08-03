#include "methods_common.hpp"

#include <Eigen/QR>

#include <cmath>
#include <stdexcept>

namespace methods::common {

bool IsFiniteNonZero(const double value) {
  return std::isfinite(value) && std::fabs(value) > kPruneValueThreshold;
}

double GetSafeQuotient(const double numerator, const double denominator) {
  if (!IsFiniteNonZero(denominator)) {
    return 0.0;
  }
  const auto quotient = numerator / denominator;
  return std::isfinite(quotient) ? quotient : 0.0;
}

void LinearCombination(const SparseMatrix& x, const double a, SparseMatrix& y, const double b) {
  y *= b;
  y += a * x;
  y.prune(1.0, kPruneValueThreshold);
}

void MulScalarColumnwise(SparseMatrix& x, const std::vector<double>& a) {
  if (static_cast<int64_t>(a.size()) != x.cols()) {
    throw std::invalid_argument("Column coefficients do not match the sparse matrix x.cols != a.cols");
  }

  for (int64_t i = 0; i < x.cols(); ++i) {
    for (SparseMatrix::InnerIterator entry(x, i); entry; ++entry) {
      entry.valueRef() *= a[i];
    }
  }
  x.prune(1.0, kPruneValueThreshold);
}

void AddScaled(SparseMatrix& target, const SparseMatrix& direction, const double coefficient) {
  if (target.rows() != direction.rows() || target.cols() != direction.cols()) {
    throw std::invalid_argument("Sparse matrices must have the same shape");
  }
  if (std::fabs(coefficient) > kPruneValueThreshold) {
    target += coefficient * direction;
  }
  target.prune(1.0, kPruneValueThreshold);
}

double FrobeniusDot(const SparseMatrix& x, const SparseMatrix& y) {
  if (x.rows() != y.rows() || x.cols() != y.cols()) {
    throw std::invalid_argument("Sparse matrices must have the same shape");
  }
  return x.cwiseProduct(y).sum();
}

Eigen::Vector2d SolveLeastSquares2x2(const Eigen::Matrix2d& matrix, const Eigen::Vector2d& x) {
  if (!matrix.allFinite() || !x.allFinite()) {
    return Eigen::Vector2d::Zero();
  }
  Eigen::Vector2d solution = matrix.completeOrthogonalDecomposition().solve(x);
  if (!solution.allFinite()) {
    solution.setZero();
  }
  return solution;
}

}  // namespace methods::common
