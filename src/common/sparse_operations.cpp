#include "methods_common.hpp"

#include <Eigen/QR>

#include <cmath>
#include <stdexcept>

namespace methods::common {

bool IsFiniteNonZero(const double value, const double threshold) {
  return std::isfinite(value) && std::fabs(value) > threshold;
}

double GetSafeQuotient(const double numerator, const double denominator) {
  if (!IsFiniteNonZero(denominator, 0.0)) {
    return 0.0;
  }
  const auto quotient = numerator / denominator;
  return std::isfinite(quotient) ? quotient : 0.0;
}

void LinearCombination(const SparseMatrix& x, const double a, SparseMatrix& y, const double b, const bool prune) {
  y *= b;
  y += a * x;
  y.prune(prune ? y.norm() : 0.0, prune ? kPruneValueThreshold : 0.0);
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
  x.prune(0.0);
}

void AddScaled(SparseMatrix& target, const SparseMatrix& direction, const double coefficient, const bool prune) {
  if (target.rows() != direction.rows() || target.cols() != direction.cols()) {
    throw std::invalid_argument("Sparse matrices must have the same shape");
  }
  if (coefficient != 0.0) {
    target += coefficient * direction;
  }
  target.prune(prune ? target.norm() : 0.0, prune ? kPruneValueThreshold : 0.0);
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
  const Eigen::Vector2d scaling = matrix.diagonal().cwiseAbs().cwiseSqrt().unaryExpr(
    [](const double value) { return value > 0.0 ? 1.0 / value : 1.0; }
  );
  const Eigen::Matrix2d scaled_matrix = scaling.asDiagonal() * matrix * scaling.asDiagonal();
  Eigen::Vector2d solution = scaling.asDiagonal() * scaled_matrix.completeOrthogonalDecomposition().solve(
    scaling.asDiagonal() * x
  );
  if (!solution.allFinite()) {
    solution.setZero();
  }
  return solution;
}

}  // namespace methods::common
