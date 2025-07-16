#include "metrics.hpp"
#include <cmath>
#include <numeric>

#include <Eigen/Dense>


double calculate_rmse(const Point& solution, const Point& pt){
   
    int d = pt.dimension();
    Eigen::VectorXd center(d);
    for (int i = 0; i < d; ++i) {
        center(i) = pt[i];
    }

    return calculate_rmse(solution,center);
}

double calculate_rmse(const Point& solution, const Eigen::VectorXd& center){


    int d = solution.dimension();
    Eigen::VectorXd sol(d);
    for (int i = 0; i < d; ++i) {
        sol(i) = solution[i];
    }

    Eigen::VectorXd diff = sol - center;
    return std::sqrt(diff.squaredNorm()/diff.size());
}

double optimalr(const HPOLYTOPE& P, const Point& pt, double radius) {

    int d = pt.dimension();
    Eigen::VectorXd center(d);
    for (int i = 0; i < d; ++i) {
        center(i) = pt[i];
    }


    return optimalr(P,center,radius);
}

double optimalr(const HPOLYTOPE& P, const Eigen::VectorXd& center, double radius){
    Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> A = P.get_mat();
    Eigen::Matrix<NT, Eigen::Dynamic, 1> b_orig = P.get_vec();
    int m = static_cast<int>(A.rows());
    int d = static_cast<int>(A.cols());  


    double externalr = 0.0;

    double b_diff;
    for (int i = 0; i < m; ++i) {
        double dot_ai_xc = A.row(i).dot(center);
        double norm_ai = A.row(i).norm();
        b_diff = (dot_ai_xc + norm_ai * radius - b_orig(i))/norm_ai;
        if(b_diff > 0 && b_diff > externalr) {
            externalr = b_diff;
        }

    }

    return radius - externalr;
}
