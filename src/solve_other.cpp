#include "solve_other.hpp"
#include <chrono>
#include <set>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

std::pair<Eigen::VectorXd,double> approximateMinEnclosingSphere(
    const std::vector<Eigen::VectorXd>& P,
    Eigen::VectorXd center,
    double eps,
    int& iter)
{
    int subiter = 0;
    const int d = center.size();
    double radius = 0;
    int t = 0;
    std::vector<Eigen::VectorXd> coreset;
    while (true) {
        subiter++;

        int    farIdx = -1;
        double maxDist = 0;
        for (int i = 0; i < (int)P.size(); ++i) {
            double dist = (P[i] - center).squaredNorm();
            if (dist > maxDist) {
                maxDist = dist;
                farIdx  = i;
            }
        }
        maxDist = std::sqrt(maxDist);

        if (maxDist <= (1 + eps) * radius && maxDist >= (1 - eps) * radius) {
            break;
        }

        double eta = 1.0 / (t + 1);
        center = (1 - eta) * center + eta * P[farIdx];
        radius = maxDist;
        ++t;
        }

    for (const auto& p : P) {
        double dist = (p - center).norm();
        if (dist > radius) radius = dist;
    }

    iter+= subiter;
    return {center, radius};
}

std::pair<Eigen::VectorXd,double> primitiveMoveWithCoreset(
    const std::vector<Eigen::VectorXd>& P,
    Eigen::VectorXd center,
    double eps,
    int& iter)
{
    int subiter = 0;
    const int d = center.size();
    const int n = static_cast<int>(P.size());
    double radius = 0.0;

    int T = static_cast<int>(std::ceil(1.0/(eps*eps)));
    for (int t = 1; t < T; ++t) {
        subiter++;

        double maxDist = -1.0;
        int farthestIdx = -1;

        for (int i = 0; i < n; ++i) {
            double dist = (P[i] - center).squaredNorm();
            if (dist > maxDist) {
                maxDist = dist;
                farthestIdx = i;
            }
        }
        if (maxDist-radius <=  eps * radius && maxDist-radius >=  -eps * radius) {
            break;
        }
        const Eigen::VectorXd& p_f = P[farthestIdx];
        double eta = 1.0 / (t + 1.0);
        center = (1.0 - eta) * center + eta * p_f;
        radius = maxDist;
    }
    
    for (const auto& p : P) {
        double dist = (p - center).norm();
        if (dist > radius) radius = dist;
    }
    iter += subiter;
    return {center, radius};
}


std::tuple<Eigen::VectorXd,double,int> solve_Xie(const HPOLYTOPE& h, double eps0) {
    int iter = 0;

    Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> A = h.get_mat();
    Eigen::Matrix<NT, Eigen::Dynamic, 1>            b = h.get_vec();

    int n = A.rows(), d = A.cols();

    for (int i = 0; i < n; ++i) {
        double normi = A.row(i).norm();
        if (normi > 0) {
            A.row(i) /= normi;
            b[i]     /= normi;
        }
    }

    std::vector<Eigen::VectorXd> P;
    P.reserve(n);
    for(int i = 0; i < n; ++i) {
        P.emplace_back( A.row(i).transpose() / b[i] );
    }

    Eigen::VectorXd init_c = Eigen::VectorXd::Zero(d);
    auto [c_dual, r_dual] = approximateMinEnclosingSphere(P, init_c, 0.1,iter);
    Eigen::VectorXd O = c_dual / r_dual;

    double hmin = std::numeric_limits<double>::infinity();
    double hmax = 0;
    for (int i = 0; i < n; ++i) {
        double hn = A.row(i).norm();
        double dist = b[i]/hn;
        hmin = std::min(hmin, dist);
        hmax = std::max(hmax, dist);
    }
    double alpha = hmax / hmin;

    double eps = eps0 / (8.0*alpha);

    double u = hmax / eps;
    for (auto& p : P) p /= u;
    O /= u;


    double lo = eps / (2.0 * alpha);
    double hi = eps;
    Eigen::VectorXd start = Eigen::VectorXd::Zero(d);


    int maxIter = static_cast<int>(std::ceil(std::log2(alpha)));
    for (int i = 0; i < maxIter; ++i) {
        double h = 0.5 * (lo + hi);

            std::vector<Eigen::VectorXd> P1;
            P1.reserve(P.size() * 2);
            for (const auto& p : P) {
                Eigen::VectorXd q1(d+1), q2(d+1);
                q1.head(d) = p; q1[d] =  +h;
                q2.head(d) = p; q2[d] =  -h;
                P1.push_back(q1);
                P1.push_back(q2);
            }

                 Eigen::VectorXd phi(d+1);
        phi.setZero();
        phi[d] = 1.0 / eps;      
        P1.push_back(phi);
        P1.push_back(-phi);

        Eigen::VectorXd init = Eigen::VectorXd::Zero(d+1);
        auto [c1, r1] = primitiveMoveWithCoreset(P1, init, eps,iter);
        double w = c1[d];

        if (std::abs(w) < 1e-12) {
            lo = h;
            continue;
        }

        Eigen::VectorXd proj = c1.head(d) / w;

        double mindist = std::numeric_limits<double>::infinity();
        for (const auto &p : P) {
            mindist = std::min(mindist, (proj - p).norm());
        }

        if (mindist >= h) {
            start = proj;
            hi = h;
        } else {
            lo = h;
        }
    }
    //good start found

    auto [c_star, r_star] = primitiveMoveWithCoreset(P, start, eps,iter);

    Eigen::VectorXd center = O + c_star / r_star;
    double radius = 1.0 / r_star/u;

    return { center, radius, iter };
}








Eigen::VectorXd maxIBPre(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& x0,
    double mu,
    int T, 
    int& iter
) 
    {
    
    const int m = A.rows();
    const int d = A.cols();


    double maxRow2 = 0.0;
    for (int i = 0; i < m; ++i)
        maxRow2 = std::max(maxRow2, A.row(i).squaredNorm());
    const double L = maxRow2 / mu;
    const double step = 1.0 / L;

    Eigen::VectorXd x = x0;
    Eigen::VectorXd y = x0;
    Eigen::VectorXd best_x = x0;
    double best_val = std::numeric_limits<double>::infinity();
    double prev_best = best_val; 

    double t_prev = 1.0;

    Eigen::VectorXd Ay(m), grad(d), Ax(m), fvec(m);

    for (int k = 0; k < T; ++k) {
        iter++;

        Ay.noalias() = A * y;


        Eigen::VectorXd v = -(Ay + b) / mu;
        double vmax = v.maxCoeff();  
        v = (v.array() - vmax).exp();  
        v /= v.sum();  

        // 3. grad = A^T * v
        grad.noalias() = A.transpose() * v;

        // Nesterov update
        Eigen::VectorXd x_new = y + step * grad;
        double t_new = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t_prev * t_prev));
        y = x_new + ((t_prev - 1.0) / t_new) * (x_new - x);
        x = x_new;
        t_prev = t_new;

        Ax.noalias() = A * x;
        fvec.noalias() = -(Ax + b) / mu;
        double fmax = fvec.maxCoeff();
        fvec = (fvec.array() - fmax).exp();
        double Z = fvec.sum();
        double f_mu = mu * std::log(m) - mu * (std::log(Z) + fmax);

        if (f_mu < best_val) {
            best_val = f_mu;
            best_x = x;
        }

        prev_best = best_val;
    }

    Ax.noalias() = A * x;
    double r_cur = std::numeric_limits<double>::infinity();
    for (int i = 0; i < m; ++i)
        r_cur = std::min(r_cur, b[i] - Ax[i]);


    return best_x;
    
}

std::tuple<Eigen::VectorXd,double,int> solve_Zhu(
    const HPOLYTOPE& h,
    double eps
    )
{
    int iter = 0;    
    
    Eigen::MatrixXd A = h.get_mat();
    Eigen::VectorXd b = h.get_vec();
    const int m = A.rows(), d = A.cols();

    for (int i = 0; i < m; ++i) {
        double normi = A.row(i).norm();
        if (normi > 0) {
            A.row(i) /= normi;
            b[i]     /= normi;
        }
    }

    std::vector<double> rho(m);
    for (int i = 0; i < m; ++i)
        rho[i] = b[i] / A.row(i).norm();

    double beta  = *std::max_element(rho.begin(), rho.end());
    double hmin  = *std::min_element(rho.begin(), rho.end());
    double alpha = beta / hmin;

    const double logm = std::log(m);
    const double sqrtlogm = std::sqrt(logm);
    const int T0 = static_cast<int>(std::ceil(4 *alpha* sqrtlogm));
    const Eigen::VectorXd x0 = Eigen::VectorXd::Zero(d);

    
    while (true) {
        double mu = beta / (8.0 * logm);
        int T = T0;

        Eigen::VectorXd y_star = maxIBPre(A, b, x0, mu, T,iter);
        Eigen::VectorXd Ay;
        Ay.noalias() = A * y_star;
        Eigen::VectorXd val = -(Ay + b) / mu;

        double vmax = val.maxCoeff();
        val = (val.array() - vmax).exp();
        double Z = val.sum();
        double f_mu = mu * logm - mu * (std::log(Z) + vmax);

        //std::cerr << "[debug] beta = " << beta << "   f_mu = " << f_mu << std::endl;

        if (f_mu <= beta / 4.0)
            beta *= 0.5;
        else
            break;
    }

    double mu = (eps * beta) / (16 * logm);
    int T = static_cast<int>(std::ceil((8 * std::sqrt(2.0* logm) *alpha  ) / eps));
    std::cout<<"enter final y"<<std::endl;

    Eigen::VectorXd final_y = maxIBPre(A, b, x0, mu, T,iter);

    double radius = std::numeric_limits<double>::infinity();
    for (int i = 0; i < m; ++i)
        radius = std::min(radius, b[i] - A.row(i).dot(final_y));

    return { final_y, radius, iter };
}


double estimateMuLocal(
    const Eigen::VectorXd& y_prev,
    const Eigen::VectorXd& g_prev,
    const Eigen::VectorXd& y_curr,
    const Eigen::VectorXd& g_curr)
{
    Eigen::VectorXd dy   = y_curr - y_prev;
    Eigen::VectorXd dg   = g_curr - g_prev;
    double denom = dy.squaredNorm();
    return (denom > 0.0 ? dg.dot(dy) / denom : 0.0);
}

double estimateLBacktracking(
    const std::function<double(const Eigen::VectorXd&)>& f,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& grad,
    const Eigen::VectorXd& y,
    double L,
    double tau = 1.5)
{
    double fy = f(y);
    Eigen::VectorXd gy = grad(y);
    double g2 = gy.squaredNorm();
    while (true) {
        double t = 1.0 / L;
        Eigen::VectorXd y_new = y - t * gy;
        if (f(y_new) <= fy - 0.5 * t * g2) break;
        L *= tau;
    }
    return L;
}

std::tuple<Eigen::VectorXd,double,int> solve_FW(const HPOLYTOPE& h, double eps) {


    Eigen::MatrixXd A = h.get_mat();  
    Eigen::VectorXd b = h.get_vec();  
    int m = A.rows(), d = A.cols();
    for (int i = 0; i < m; ++i) {
        double ni = A.row(i).norm();
        if (ni > 0.0) {
            A.row(i) /= ni;
            b(i)     /= ni;
        }
    }



    std::vector<double> rho(m);
    for (int i = 0; i < m; ++i)
        rho[i] = b[i] / A.row(i).norm();

    double tau =*std::max_element(rho.begin(), rho.end()) / *std::min_element(rho.begin(), rho.end());

    double eta = 1.0 / eps;
        auto f = [&](const Eigen::VectorXd& y) {
        Eigen::VectorXd Aty = A.transpose() * y;
        return b.dot(y) + 0.5 * eta * Aty.squaredNorm();
    };
    auto gradF = [&](const Eigen::VectorXd& y) {
        return b + eta * (A * (A.transpose() * y));
    };


    Eigen::VectorXd y = Eigen::VectorXd::Zero(m);
    {
        int best = 0;
        for (int i = 1; i < m; ++i) {
            if (b(i) < b(best)) best = i;
        }
        y(best) = 1.0;
    }

  
    Eigen::VectorXd y_prev = y;
    Eigen::VectorXd g_prev = gradF(y);

    double L = 1.0;       
    double mu = 0.0;        
    L  = estimateLBacktracking(f, gradF, y, L);
    mu = estimateMuLocal(y_prev, g_prev, y, gradF(y));

    double r0 = b.dot(y);
    Eigen::VectorXd Aty0 = A.transpose() * y;          
    Eigen::VectorXd g0   = b + eta * (A * Aty0);      
    int j_FW0 = 0;
    for (int i = 1; i < m; ++i) {
        if (g0(i) < g0(j_FW0)) j_FW0 = i;
    }
    int j_AW0 = -1;
    for (int i = 0; i < m; ++i) {
        if (y(i) > 0.0) { j_AW0 = i; break; }
    }
    for (int i = j_AW0+1; i < m; ++i) {
        if (y(i) > 0.0 && g0(i) > g0(j_AW0)) j_AW0 = i;
    }
    double gap0 = g0(j_AW0) - g0(j_FW0);   
    double target_gap = eps * r0;

    int K_max = 0;
    if (target_gap > 0.0 && gap0 > target_gap) {
        K_max = static_cast<int>(
            std::ceil((tau*tau) * std::log(gap0/target_gap))
        );
    }
    if (K_max < 1) {
        K_max = static_cast<int>(2.0/eps);
    }

    int iter = 0;
    int k = 0;
    std::vector<int> S; S.reserve(m);
    
    while (k < K_max) {
        k++;
        iter++;
    
        //  g = b + η·A·(Aᵀy)
        Eigen::VectorXd Aty = A.transpose() * y;    // O(m·d)
        Eigen::VectorXd g   = b + eta * (A * Aty);  // O(m·d)

        L = estimateLBacktracking(f, gradF, y, L);
        mu = estimateMuLocal(y_prev, g_prev, y, g);
        y_prev = y;
        g_prev = g;

        int j_FW = 0;
        for (int i = 1; i < m; ++i) {
            if (g(i) < g(j_FW)) j_FW = i;
        }

        S.clear();
        for (int i = 0; i < m; ++i) {
            if (y(i) > 0.0) S.push_back(i);
        }
        if (S.empty()) break;
        int j_AW = S[0];
        for (size_t idx = 1; idx < S.size(); ++idx) {
            int i = S[idx];
            if (g(i) > g(j_AW)) j_AW = i;
        }
        
        double gap = g(j_AW) - g(j_FW);
        double r_cur = b.dot(y);
        //std::cout<<"gap: "<<gap<<"eps*r: "<<eps * r_cur<<std::endl;
        if (gap <= eps * r_cur) {
            std::cout<<"break ";
            break;
        }

        Eigen::VectorXd d = Eigen::VectorXd::Zero(m);
        d(j_FW) = +1.0;
        d(j_AW) = -1.0;

        double numer = gap;
        Eigen::VectorXd Atd = A.transpose() * d;    // O(m·d)
        double denom = eta * Atd.squaredNorm();
        //std::cout<<"L: "<<L<<"mu: "<<mu<<"alpha: "<<numer/denom<<std::endl;
        double alpha = 0.0;
        if (denom > 0.0) alpha = numer/denom;
        if (alpha < 0.0) alpha = 0.0;
        if (alpha > y(j_AW)) alpha = y(j_AW);

        y(j_FW) += alpha;
        y(j_AW) -= alpha;

        
        if (mu > 0 && gap > eps * r_cur) {
            K_max = std::ceil(tau*tau*(L/mu) * std::log(gap0/(eps)));
            
        }
    }

    double r = b.dot(y);            
    Eigen::VectorXd c = A.transpose() * y;  

    return {-c, r,iter};
}
