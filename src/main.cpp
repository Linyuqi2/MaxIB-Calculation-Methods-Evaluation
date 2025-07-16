#include <Eigen/Dense>
#include <numeric>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
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

double stdev(const std::vector<double>& v) {
    const size_t n = v.size();
    if (n == 0) return 0.0;
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
    double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double variance = sq_sum / n - mean * mean;
    return std::sqrt(variance);
}

double randn() {
    static std::mt19937_64 gen{std::random_device{}()};
    static std::normal_distribution<> dist;
    return dist(gen);
}

int main() {
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double>;

    std::vector<int> dimensions = {100};//,,100,150,200,300,400,500,600,700,800,900,1000
    int trials = 2;
    double abratio = 3;

    std::vector<std::pair<std::string,std::vector<double>>> data = {
        {"a/b:", {abratio}},
        {"Dimension:", std::vector<double>(dimensions.begin(), dimensions.end())},
        {"RMSE:",   {}},
        {"SD: RMSE:",   {}},
        {"Time consumption:", {}},
        {"radius%:", {}},
        {"SD: radius%:", {}},
        {"# of iteration:", {}},
        {"SD: iteration:", {}},
        {"rmses1: ",{}},
        {"rmses2: ",{}}
    };

    std::vector<double> rmse1;
            std::vector<double> rmse2;
    for (int dim : dimensions) {
        std::vector<double> rmses_other;
        rmses_other.reserve(trials);

        std::vector<double> optimalrs;
        optimalrs.reserve(trials);

        std::vector<double> percentr;
        percentr.reserve(trials);

        std::vector<int>iterations;
        iterations.reserve(trials);
        
        Duration t_total(0),t_total2(0);
        
        for (int t = 0; t < trials; ++t) {
            auto P = generate_skinny_random_polytope(dim, dim * 4,abratio);
            //auto P = generate_random_polytope(dim,dim*3);
            //auto P = generate_cube<HPOLYTOPE>(5,false);

            /*
            std::cout << "[Trial " << t << "] Polytope: " <<std::endl;
            P.print();
            */
            
            
            Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> A = P.get_mat();
            Eigen::Matrix<NT, Eigen::Dynamic, 1>            b = P.get_vec();

            std::vector<Eigen::VectorXd> vertices;
    
            

            auto sol = solve_mosek_interior(P);

            auto t0 = Clock::now();
            auto[c,r,iter] =solve_FW(P,0.05);// solve_nesterov_AGS(P,0.01,rmse2,sol.first);
            t_total += std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0);
            auto t1 = Clock::now();
            auto[c2,r2,iter2] = solve_Zhu(P,0.05);
            t_total2 += std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t1);
    
            if (r >= 0) {
                
                std::cout<<"difference: "<<calculate_rmse(c,c2)<<std::endl;
 
                std::cout<<"Difference to 0: "<<calculate_rmse(c2,Eigen::VectorXd::Zero(dim))<<std::endl;
                std::cout<<"Correct d to 0: "<<calculate_rmse(sol.first,Eigen::VectorXd::Zero(dim))<<std::endl;
                std::cout<<"0: "<<optimalr(P,Eigen::VectorXd::Zero(dim),100)/sol.second<<std::endl;

                double rmse = calculate_rmse(sol.first,c);
                double rmse2 = calculate_rmse(sol.first,c2);
                rmses_other.push_back(rmse);

                
                double opt_r = optimalr(P,c,r);
                double opt_r2 = optimalr(P,c2,r2);
                optimalrs.push_back(opt_r);

                double percent = opt_r/sol.second;///
                double percent2 = opt_r2/sol.second;
                percentr.push_back(percent);

                //iterations.push_back(iter);

                const Point& pt = c;
                int d = pt.dimension();
                Eigen::VectorXd center(d);
                for (int i = 0; i < d; ++i) {
                    center(i) = pt[i];
                }

                std::cout 
                << "[Trial " << t <<"] "
                << " center: "<< center[0] 
                << " radius: " << r 
                << " rmse: "<<rmse
                //<< " optimal radius: "<<opt_r
                << " %r: "<<percent
                //<<" # of iterations: "<<iter
                <<std::endl;

                std::cout 
                << "[Trial " << t <<"]* "
                << " center: "<< c2[0] 
                << " radius: " << r2 
                << " rmse: "<<rmse2
                //<< " optimal radius: "<<opt_r
                << " %r: "<<percent2
                //<<" # of iterations: "<<iter2
                <<std::endl;

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
        
        auto avg = [&](const std::vector<double>& v){
            if (v.empty()) return std::nan("");
            double sum = std::accumulate(v.begin(), v.end(), 0.0);
            return sum / static_cast<double>(v.size());
        };

        double avg_rmse = avg(rmses_other);
        double avg_opt_r = avg(optimalrs);
        double avg_percent =avg(percentr);


        std::cout << "[ DIMENSION:"<<dim <<"] RMSE =" << avg_rmse << " Avg optimal radius= " << avg_opt_r <<" Avg r% = "<< avg_percent
        <<" Avg Time Comsumption:"<< (std::chrono::duration_cast<std::chrono::milliseconds>(t_total).count()/trials)<<"ms"
        <<" Avg Time2:"<< (std::chrono::duration_cast<std::chrono::milliseconds>(t_total2).count()/trials)<<"ms"
        <<std::endl;

        data[2].second.push_back(avg_rmse);
        data[4].second.push_back((std::chrono::duration_cast<std::chrono::milliseconds>(t_total).count()/trials));
        data[5].second.push_back(avg_percent);
        data[7].second.push_back(avg(std::vector<double>(iterations.begin(),iterations.end())));
        data[3].second.push_back(stdev(rmses_other));
        data[6].second.push_back(stdev(percentr));
        data[8].second.push_back(stdev(std::vector<double>(iterations.begin(),iterations.end())));
        data[9].second = rmse1;
        data[10].second = rmse2;

    }

    std::ofstream ofs("output.csv");
    if (!ofs) {
        std::cerr << "fail to open 'output.csv' \n";
        return 1;
    }
        for (const auto &entry : data) {
        const std::string &name = entry.first;
        const std::vector<double> &vals = entry.second;

        ofs << name;

        for (double v : vals) {
            ofs << "," << v;
        }
        ofs << "\n";
    }

    ofs.close();
    std::cout << "output.csv generated\n";

    return 0;
}
