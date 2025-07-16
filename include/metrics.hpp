#pragma once
#include "convex_bodies/hpolytope.h"
#include <cartesian_geom/cartesian_kernel.h>

typedef double NT;
typedef Cartesian <NT> Kernel;
typedef typename Kernel::Point Point;
typedef HPolytope <Point> HPOLYTOPE;

double calculate_rmse(const Point& sol, const Point& pt);
double calculate_rmse(const Point& sol, const Eigen::VectorXd& center);
double optimalr(const HPOLYTOPE& P, const Point& pt, double radius);
double optimalr(const HPOLYTOPE& P, const Eigen::VectorXd& center, double radius);

