#pragma once

#include "convex_bodies/hpolytope.h"
#include "cartesian_geom/cartesian_kernel.h"
#include <utility>

typedef double NT;
typedef Cartesian <NT> Kernel;
typedef typename Kernel::Point Point;
typedef HPolytope <Point> HPOLYTOPE;

std::pair<Point, double> solve_lpsolve(const HPOLYTOPE& P);
std::pair<Point,double> solve_glpk_simplex(const HPOLYTOPE& P);
std::pair<Point, double> solve_glpk_interior(const HPOLYTOPE& P);
std::pair<Point, double> solve_interior(const HPOLYTOPE& P);

