#include "metrics.hpp"
#include <cmath>
#include <numeric>

#include <Eigen/Dense>


double calculate_rmse(const HPOLYTOPE& P, const std::pair<Point, double>& solution){
   
    const Point& pt = solution.first;
    
    int d = pt.dimension();
    Eigen::VectorXd center(d);
    for (int i = 0; i < d; ++i) {
        center(i) = pt[i];
    }

    double radius = solution.second;
    return calculate_rmse(P,std::pair<Eigen::VectorXd,double>(center,radius));
}

double calculate_rmse(const HPOLYTOPE& P, const std::pair<Eigen::VectorXd, double>& solution){
    Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> A     = P.get_mat();
    Eigen::Matrix<NT, Eigen::Dynamic, 1>            b_orig = P.get_vec();
    int m = static_cast<int>(A.rows());
    int d = static_cast<int>(A.cols());  

    Eigen::VectorXd center = solution.first;
    double radius = solution.second;


    Eigen::VectorXd b_pred(m);
    for (int i = 0; i < m; ++i) {
        double dot_ai_xc = A.row(i).dot(center);
        double norm_ai   = A.row(i).norm();
        b_pred(i)        = dot_ai_xc + norm_ai * radius;
    }


    Eigen::VectorXd diff  = b_orig - b_pred;
    double sum_sq         = diff.squaredNorm();
    return std::sqrt(sum_sq / m);
}
