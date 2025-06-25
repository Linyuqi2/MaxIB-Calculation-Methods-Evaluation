#pragma once
#include "convex_bodies/hpolytope.h"
#include "generators/h_polytopes_generator.h"
#include "cartesian_geom/cartesian_kernel.h"
#include <chrono>
#include <random>

typedef double NT;
typedef Cartesian <NT> Kernel;
typedef typename Kernel::Point Point;
typedef HPolytope <Point> HPOLYTOPE;


HPOLYTOPE generate_random_polytope(int dim, int num_facets);

HPOLYTOPE generate_skinny_random_polytope(int dim, int num_facets,double ratio);