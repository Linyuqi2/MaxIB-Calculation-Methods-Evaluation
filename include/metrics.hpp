#pragma once
#include "convex_bodies/hpolytope.h"
#include <cartesian_geom/cartesian_kernel.h>

typedef double NT;
typedef Cartesian <NT> Kernel;
typedef typename Kernel::Point Point;
typedef HPolytope <Point> HPOLYTOPE;

double calculate_rmse(const HPOLYTOPE& P, const std::pair<Point,double>& solution);
double calculate_rmse(const HPOLYTOPE& P, const std::pair<Eigen::VectorXd, double>& solution);
