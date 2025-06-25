#pragma once

#include <Eigen/Dense>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <limits>
#include "solve.hpp"
#include "convex_bodies/hpolytope.h"
#include "cartesian_geom/cartesian_kernel.h"


std::pair<Eigen::VectorXd,double> solve_dual(const HPOLYTOPE& h, double eps0);
Eigen::VectorXd getInteriorPoint(const HPOLYTOPE& h);
std::pair<Eigen::VectorXd,double> approximateMinEnclosingSphere(const std::vector<Eigen::VectorXd>& P, Eigen::VectorXd init_c, double eps);
std::pair<Eigen::VectorXd,double> primitiveMoveWithCoreset(const std::vector<Eigen::VectorXd>& P, Eigen::VectorXd init_c, double eps);
Eigen::VectorXd liftAndFindGoodStart(const std::vector<Eigen::VectorXd>& P, double eps, double alpha);

std::pair<Eigen::VectorXd,double> solve_optimization(const HPOLYTOPE& h, double eps);
double smoothMinAt(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& x, double mu);
double computeL(const Eigen::MatrixXd& A, double mu);
Eigen::VectorXd gradient(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& x, double mu);
Eigen::VectorXd maxIBPre(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& x0, double mu, int T);
double refineBeta(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, double eps);
