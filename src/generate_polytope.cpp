#include "generate_polytope.hpp"
#include "random_walks/random_walks.hpp"

#include <boost/random/mersenne_twister.hpp>

//typedef BoostRandomNumberGenerator<boost::mt19937, NT, 3> RNGType;
typedef boost::mt19937 RNGType;

HPOLYTOPE generate_random_polytope(int dim, int num_facets) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);
    int seed = dist(gen);
    return random_hpoly<HPOLYTOPE,RNGType>(dim,num_facets, seed);
}

HPOLYTOPE generate_skinny_random_polytope(int dim, int num_facets,double ratio) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);
    int seed = dist(gen);
    return skinny_random_hpoly<HPOLYTOPE,double, RNGType>(dim,num_facets,false,NT(ratio),seed);
}

