#include <Eigen/Dense>
#include <numeric>
#include <vector>
#include <iostream>
#include <chrono>
#include "generators/known_polytope_generators.h"
#include "generate_polytope.hpp"
#include "solve.hpp"
#include "solve_mosek.hpp"
#include "metrics.hpp"
#include "solve_other.hpp"


typedef double NT;
typedef Cartesian<NT>     Kernel;
typedef typename Kernel::Point  Point;
typedef HPolytope<Point>       HPOLYTOPE;

int main() {
    std::vector<int> dimensions = {900,1000};
    int trials = 10;

    for (int dim : dimensions) {
        std::vector<double> rmses_simplex;
        rmses_simplex.reserve(trials);

        std::vector<double> rmses_other;
        rmses_other.reserve(trials);
        
        auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < trials; ++t) {
            auto P = generate_skinny_random_polytope(dim, dim * 4,5);
            //auto P = generate_random_polytope(dim,dim*4);
            //auto P = generate_cube<HPOLYTOPE>(5,false);

            /*
            std::cout << "[Trial " << t << "] Polytope: " <<std::endl;
            P.print();
            */
            
            /*
            Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> A = P.get_mat();
            Eigen::Matrix<NT, Eigen::Dynamic, 1>            b = P.get_vec();
            int m = A.rows(), d = A.cols();
            for(int i = 0; i < m; ++i) {
                double ni = b[i];
                if (ni > 0) {
                    A.row(i) /= ni;
                    b[i]     /= ni;
                }
            }
            P.set_mat(A);
            P.set_vec(b);
            */


            auto sol = solve_mosek_primal_simplex(P);
            if (sol.second >= 0) {
                double rmse = calculate_rmse(P, sol);
                rmses_other.push_back(rmse);

                const Point& pt = sol.first;
                int d = pt.dimension();
                Eigen::VectorXd center(d);
                for (int i = 0; i < d; ++i) {
                    center(i) = pt[i];
                }

                std::cout 
                //<< "[Trial " << t << "] HighDimension: center: "<< center << "radius: " << sol.second 
                <<"rmse: "<<rmse<<std::endl;
            } else {
                std::cout << "[Trial " << t << "] Method failed. ";
            }
            
            /* For comparison
            auto sol1 = solve_lpsolve(P);
            if (sol1.second >= 0) {
                double rmse1 = calculate_rmse(P, sol1);
                rmses_simplex.push_back(rmse1);
                std::cout << "[Trial " << t << "] LPSolve: radius: " << sol1.second <<"rmse: "<<rmse1<<std::endl;
            } else {
                std::cerr << "[Trial " << t << "] Primal Simplex LP failed. ";
            }*/

        }
        auto t1 = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        auto avg = [&](const std::vector<double>& v){
            if (v.empty()) return std::nan("");
            double sum = std::accumulate(v.begin(), v.end(), 0.0);
            return sum / static_cast<double>(v.size());
        };

        double avg_simp = avg(rmses_simplex);
        double avg_rmse = avg(rmses_other);


        std::cout << "Dimension:"<<dim <<"RMSE = " << avg_rmse <<"Avg Time Comsumption: "<< double(duration/trials)<<"ms"<<std::endl;
    }

    return 0;
}
