#pragma once

#include <Eigen/Dense>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>
#include "solve.hpp"
#include "convex_bodies/hpolytope.h"
#include "cartesian_geom/cartesian_kernel.h"
#include "metrics.hpp"


std::tuple<Eigen::VectorXd,double,int> solve_Xie(const HPOLYTOPE& h, double eps0);
std::pair<Eigen::VectorXd,double> approximateMinEnclosingSphere(const std::vector<Eigen::VectorXd>& P, Eigen::VectorXd init_c, double eps ,int& iter);
std::pair<Eigen::VectorXd,double> primitiveMoveWithCoreset(const std::vector<Eigen::VectorXd>& P, Eigen::VectorXd init_c, double eps,int& iter);


std::tuple<Eigen::VectorXd,double,int> solve_Zhu(const HPOLYTOPE& h, double eps);
Eigen::VectorXd maxIBPre(const Eigen::MatrixXd& A,const Eigen::VectorXd& b,const Eigen::VectorXd& x0,double mu, int T,int& iter);

std::tuple<Eigen::VectorXd,double,int> solve_FW(const HPOLYTOPE& h, double eps);



double estimateMuLocal(
    const Eigen::VectorXd& y_prev,
    const Eigen::VectorXd& g_prev,
    const Eigen::VectorXd& y_curr,
    const Eigen::VectorXd& g_curr);

double estimateLBacktracking(
    const std::function<double(const Eigen::VectorXd&)>& f,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& grad,
    const Eigen::VectorXd& y,
    double L,
    double tau);