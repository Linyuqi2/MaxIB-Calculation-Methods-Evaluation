#include "solve_other.hpp"

Eigen::VectorXd getInteriorPoint(const HPOLYTOPE& h) {
    Eigen::MatrixXd A = h.get_mat();   // n×d
    Eigen::VectorXd b = h.get_vec();   // n
    int n = A.rows(), d = A.cols();

    std::vector<Eigen::VectorXd> P;
    P.reserve(n);
    for(int i = 0; i < n; ++i) {
        P.emplace_back( A.row(i).transpose() / b[i] );
    }

    Eigen::VectorXd init_c = Eigen::VectorXd::Zero(d);
    auto [c_dual, r_dual] = approximateMinEnclosingSphere(P, init_c, 0.1);

    std::cout<<"c dual: "<< c_dual<< "    r dual: "<<r_dual<<std::endl;

    return c_dual / r_dual;
}

std::pair<Eigen::VectorXd,double> approximateMinEnclosingSphere(
    const std::vector<Eigen::VectorXd>& P,
    Eigen::VectorXd center,
    double eps)
{
    const int d = center.size();
    double radius = 0;
    std::vector<Eigen::VectorXd> coreset;
    while (true) {
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

        if (maxDist <= (1 + eps) * radius) {
            break;
        }

        coreset.push_back(P[farIdx]);

        Eigen::VectorXd y = Eigen::VectorXd::Zero(d);
        for (auto& q : coreset) y += q;
        y /= coreset.size();
        center = y;
        radius = 0;
        for (auto& q : coreset) {
        radius = std::max(radius, (q - center).norm());
        }
    }
    return {center, radius};
}

std::pair<Eigen::VectorXd,double> primitiveMoveWithCoreset(
    const std::vector<Eigen::VectorXd>& P,
    Eigen::VectorXd init_c,
    double eps)
{
    //std::cout<<"enter movewithcoreset function"<<std::endl;
    Eigen::VectorXd c = init_c;
    double r, delta;

    int T = static_cast<int>(std::ceil(1.0/(eps)));
    for (int t = 0; t < T; ++t) {
        //std::cout<<"enter loop inside movewithcoreset function"<<std::endl;
        std::tie(c, r) = approximateMinEnclosingSphere(P, c, eps);
        delta = c.norm();
        if (delta <= eps * r) break;  
        
        double step = delta / std::sqrt(r*r - delta*delta);
        c += step * (c / delta);
    }
    std::cout<<"exit movewithcoreset function"<<std::endl;
    return {c, r};
}

Eigen::VectorXd liftAndFindGoodStart(
    const std::vector<Eigen::VectorXd>& P,
    double eps,
    double alpha)
{
    int d = static_cast<int>(P[0].size());
    double lo = eps / (2.0 * alpha);
    double hi = eps;
    Eigen::VectorXd bestStart = Eigen::VectorXd::Zero(d);

    int maxIter = static_cast<int>(std::ceil(std::log2(alpha)));
    for (int iter = 0; iter < maxIter; ++iter) {
        double h = 0.5 * (lo + hi);

        std::vector<Eigen::VectorXd> P1;
        P1.reserve(P.size() * 2);
        for (const auto &p : P) {
            Eigen::VectorXd q1(d+1), q2(d+1);
            q1.head(d) = p;  q1[d] =  h;
            q2.head(d) = p;  q2[d] = -h;
            P1.push_back(q1);
            P1.push_back(q2);
        }

        Eigen::VectorXd init = Eigen::VectorXd::Zero(d+1);
        auto [c1, r1] = primitiveMoveWithCoreset(P1, init, eps);
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
            bestStart = proj;
            hi = h;
        } else {
            lo = h;
        }
    }

    return bestStart;
}

std::pair<Eigen::VectorXd,double> solve_dual(const HPOLYTOPE& h, double eps0) {

    Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> A = h.get_mat();
    Eigen::Matrix<NT, Eigen::Dynamic, 1>            b = h.get_vec();
    int n = A.rows(), d = A.cols();

    Eigen::VectorXd O = getInteriorPoint(h);
    //std::cout<<"Interior point O: "<< O <<std::endl;

    double rmin = std::numeric_limits<double>::infinity();
    double rmax = 0;
    for (int i = 0; i < n; ++i) {
        double hn = A.row(i).norm();
        double dist = b[i]/hn;
        rmin = std::min(rmin, dist);
        rmax = std::max(rmax, dist);
    }
    double alpha = rmax / rmin;

    double eps = eps0 / (8.0*alpha);

    b = b - A*O;

    //double scale = eps / rmax;
    //A   *= scale;
    //b   *= scale;

    std::vector<Eigen::VectorXd> P;
    P.reserve(n);
    for (int i = 0; i < n; ++i) {
        P.push_back(A.row(i).transpose()/b[i]);
    }

    
    //std::cout << "First 5 dual pts:\n";
    //for (int i = 0; i < 5; ++i) {
        //std::cout << P[i].transpose() << "\n";
    //}

    Eigen::VectorXd start = liftAndFindGoodStart(P, eps, alpha);

    std::cout << "primitiveMovewithCoreset" << std::endl;
    auto [c_star, r_star] = primitiveMoveWithCoreset(P, start, eps);

    //Eigen::VectorXd center = O + c_star/(r_star*scale);
    //double radius = r_star/scale;
    std::cout << "function finished" << std::endl;
    Eigen::VectorXd center = O + c_star / r_star;
    double radius = 1.0 / r_star;

    return { center, radius };
}


double smoothMinAt(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& x,
    double mu)
{
    int m = A.rows();
    Eigen::VectorXd v(m);
    for(int i = 0; i < m; ++i) {
        v[i] = -(A.row(i).dot(x) - b[i]) / mu;
    }
    double vmax = v.maxCoeff();
    double sum = (v.array() - vmax).exp().sum();
    return -mu * (std::log(sum) + vmax);
}

double computeL(const Eigen::MatrixXd& A, double mu) {
    int m = A.rows();
    double maxRow2 = 0;
    for(int i = 0; i < m; ++i) {
        double row2 = A.row(i).squaredNorm();
        maxRow2 = std::max(maxRow2, row2);
    }
    return maxRow2 / mu;
}


Eigen::VectorXd gradient(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& x,
    double mu)
{
    int m = A.rows(), d = A.cols();
    Eigen::VectorXd v(m);
    for(int i = 0; i < m; ++i) {
        v[i] = -(A.row(i).dot(x) - b[i]) / mu;
    }
    double vmax = v.maxCoeff();
    v = (v.array() - vmax).exp();
    v /= v.sum();
    return A.transpose() * v;
}

// Nesterov-accelerated gradient ascent on f_μ
// maximize f_μ starting from x0, for T iterations
Eigen::VectorXd maxIBPre(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& x0,
    double mu,
    int T)
{
    int d = A.cols();
    double L = computeL(A, mu);
    double step = 1.0 / L;

    Eigen::VectorXd x = x0;
    Eigen::VectorXd y = x0;
    double t_prev = 1.0;

    for(int k = 0; k < T; ++k) {
        
        Eigen::VectorXd grad = gradient(A, b, y, mu);
        Eigen::VectorXd x_new = y + step * grad;  // ← 用 step

        double t_new = 0.5*(1.0 + std::sqrt(1.0 + 4*t_prev*t_prev));
        y = x_new + ((t_prev - 1.0)/t_new)*(x_new - x);

        x = x_new;
        t_prev = t_new;

        /*
        if (k % 50 == 0) {
            double fval = smoothMinAt(A, b, x, mu);
            std::cerr<<"[Nesterov] k="<<k<<" f_mu="<<fval<<"\n";
        }*/
    }
    return x;
}


double refineBeta(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    double eps)
{
    int m = A.rows();
    std::vector<double> rho(m);
    for(int i = 0; i < m; ++i) {
        rho[i] = b[i] / A.row(i).norm();
    }
    double beta = *std::max_element(rho.begin(), rho.end());
    double alpha = beta / *std::min_element(rho.begin(), rho.end());

    int maxIter = static_cast<int>(std::ceil(std::log2(alpha)));
    for(int it = 0; it < maxIter; ++it) {
        double mu_test = beta / std::log(m);
        double f0 = smoothMinAt(A, b, Eigen::VectorXd::Zero(A.cols()), mu_test);
        if (f0 >= beta / 2.0) {
            beta *= 0.5;
        } else {
            break;
        }
    }
    return beta;
}


std::pair<Eigen::VectorXd,double> solve_optimization(
    const HPOLYTOPE& h,
    double eps)
{
    Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> A = h.get_mat();
    Eigen::Matrix<NT, Eigen::Dynamic, 1>            b = h.get_vec();

    int m = A.rows(), d = A.cols();

    for(int i = 0; i < m; ++i) {
        double ni = A.row(i).norm();
        if (ni > 0) {
            A.row(i) /= ni;
            b[i]     /= ni;
        }
    }
    double beta = refineBeta(A, b, eps);
    //std::cerr<<"[debug] refined beta="<<beta<<"\n";

    double mu = eps * beta / std::log(m);
    //std::cerr<<"[debug] use mu="<<mu<<"\n";


    int T = static_cast<int>(std::ceil(std::sqrt(std::log(m)) * beta / eps));
    //std::cerr<<"[debug] Nesterov Iters T="<<T<<"\n";
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(d);
    Eigen::VectorXd x_star = maxIBPre(A, b, x0, mu, T);

    
    double radius = std::numeric_limits<double>::infinity();
    for(int i = 0; i < m; ++i) {
        radius = std::min(radius, b[i] - A.row(i).dot(x_star));
    }

    return { x_star, radius };
}