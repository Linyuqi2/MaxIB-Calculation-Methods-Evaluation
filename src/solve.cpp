#include "solve.hpp"
#include "lp_oracles/solve_lp.h"

#include <lp_lib.h>
#include <glpk.h> 
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>

std::pair<Point, double> solve_lpsolve(const HPOLYTOPE& P) {
    std::pair<Point,double> exception_pair(Point(1), -1.0);

    lprec* lp = nullptr;
    int *colno = nullptr;
    REAL *row_vals = nullptr;

    try {
        Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> A = P.get_mat();
        Eigen::Matrix<NT, Eigen::Dynamic, 1>            b = P.get_vec();
        int m = static_cast<int>(A.rows());   
        int d = static_cast<int>(A.cols());   
        int Ncol = d + 1;                     

        lp = make_lp(m, Ncol);
        if (lp == nullptr) throw false;

        REAL infinite = get_infinite(lp);

        colno   = static_cast<int*>(malloc(Ncol * sizeof(int)));
        row_vals = static_cast<REAL*>(malloc(Ncol * sizeof(REAL)));
        if (!colno || !row_vals) throw false;

        set_add_rowmode(lp, TRUE);

        for (int i = 0; i < m; ++i) {
            NT norm_ai = 0.0;
            for (int j = 0; j < d; ++j) {
                norm_ai += A(i, j) * A(i, j);
            }
            norm_ai = std::sqrt(norm_ai);

            for (int j = 0; j < d; ++j) {
                colno[j]   = j + 1;                     
                row_vals[j] = static_cast<REAL>(A(i, j));
            }
            colno[d]    = d + 1;                        
            row_vals[d] = static_cast<REAL>(norm_ai);

            // constraint a_i^T x + ||a_i|| r ≤ b_i
            if (!add_constraintex(lp, Ncol, row_vals, colno, LE, static_cast<REAL>(b(i)))) {
                throw false;
            }
        }

        set_add_rowmode(lp, FALSE);

        for (int j = 0; j < d; ++j) {
            colno[j]   = j + 1;
            row_vals[j] = 0.0;                       
            set_bounds(lp, j + 1, -infinite, infinite);
        }
        colno[d]   = d + 1;
        row_vals[d] = 1.0;                         
        set_bounds(lp, d + 1, 0.0, infinite);

        //max r
        for (int j = 0; j < d; ++j) {
            colno[j]   = j + 1;
            row_vals[j] = 0.0;
        }
        colno[d]   = d + 1;
        row_vals[d] = 1.0;
        if (!set_obj_fnex(lp, Ncol, row_vals, colno)) {
            throw false;
        }

        set_maxim(lp);
        set_verbose(lp, NEUTRAL);

        if (solve(lp) != OPTIMAL) {
            int st = get_status(lp);
            if (st == UNBOUNDED) {
                std::cerr << "[solve_lpsolve] LP is UNBOUNDED.\n";
            } else if (st == INFEASIBLE) {
                std::cerr << "[solve_lpsolve] LP is INFEASIBLE.\n";
            } else {
                std::cerr << "[solve_lpsolve] LP_solve failed: returned status=" << st << "\n";
            }
            throw false;
        }

        if (!get_variables(lp, row_vals)) {
            //std::cout<<"failed to get variables"<<std::endl;
            throw false;
        }

        std::vector<NT> temp_p(d, 0.0);
        for (int j = 0; j < d; ++j) {
            temp_p[j] = static_cast<NT>(row_vals[j]);
        }
        Point xc(d, temp_p.begin(), temp_p.end());

        NT radius = static_cast<NT>(get_objective(lp));

        delete_lp(lp);
        free(colno);
        free(row_vals);

        return { xc, static_cast<double>(radius) };
    }
    catch (bool e) {
        if (lp) delete_lp(lp);
        if (colno) free(colno);
        if (row_vals) free(row_vals);
        return exception_pair;
    }
}

std::pair<Point,double> solve_glpk_simplex(const HPOLYTOPE& P) {
    
    std::pair<Point,double> exception_pair(Point(1), -1.0);

    Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> A = P.get_mat();
    Eigen::Matrix<NT, Eigen::Dynamic, 1>            b = P.get_vec();
    int m = static_cast<int>(A.rows());
    int d = static_cast<int>(A.cols());
    int Ncol = d + 1;  

    glp_prob* lp = glp_create_prob();
    glp_set_prob_name(lp, "chebyshev_ball_simplex");
    glp_set_obj_dir(lp, GLP_MAX);  
    
    glp_add_rows(lp, m);
    for (int i = 1; i <= m; ++i) {
        double norm_ai = 0.0;
        for (int j = 0; j < d; ++j) {
            norm_ai += A(i-1, j) * A(i-1, j);
        }
        norm_ai = std::sqrt(norm_ai);

        glp_set_row_bnds(lp, i, GLP_UP, 0.0, static_cast<double>(b(i-1)));
    }

    glp_add_cols(lp, Ncol);
    for (int j = 1; j <= d; ++j) {
        glp_set_col_bnds(lp, j, GLP_FR, 0.0, 0.0);
        glp_set_obj_coef(lp, j, 0.0);  
    }

    glp_set_col_bnds(lp, d+1, GLP_LO, 0.0, 0.0);
    glp_set_obj_coef(lp, d+1, 1.0);  

    int total_nz = m * (d + 1);
    int* ia = new int[total_nz + 1];
    int* ja = new int[total_nz + 1];
    double* ar = new double[total_nz + 1];

    int idx = 1;
    for (int i = 1; i <= m; ++i) {
        double norm_ai = 0.0;
        for (int j = 0; j < d; ++j) {
            norm_ai += A(i-1, j) * A(i-1, j);
        }
        norm_ai = std::sqrt(norm_ai);

        // a_i^T x
        for (int j = 1; j <= d; ++j) {
            ia[idx] = i;          
            ja[idx] = j;          
            ar[idx] = static_cast<double>(A(i-1, j-1));
            ++idx;
        }
        // ||a_i|| * r 
        ia[idx] = i;
        ja[idx] = d + 1;         
        ar[idx] = norm_ai;
        ++idx;
    }

    glp_load_matrix(lp, total_nz, ia, ja, ar);

    glp_smcp parm;
    glp_init_smcp(&parm);
    parm.msg_lev = GLP_MSG_OFF;  
    int ret = glp_simplex(lp, &parm);
    if (ret != 0) {
        glp_delete_prob(lp);
        delete[] ia;
        delete[] ja;
        delete[] ar;
        return exception_pair;
    }

    int status = glp_get_status(lp);
    if (status != GLP_OPT) {
        double st = glp_get_status(lp);
        if (st == GLP_UNBND) {
            std::cerr << "[solve_glpk_simplex] LP is UNBOUNDED.\n";
        } else if (st == GLP_INFEAS) {
            std::cerr << "[solve_glpk_simplex] LP is INFEASIBLE.\n";
        } else {
            std::cerr << "[solve_glpk_simplex] GLPK simplex failed: status=" << st << "\n";
        }
        glp_delete_prob(lp);
        delete[] ia;
        delete[] ja;
        delete[] ar;
        return exception_pair;
    }
    std::vector<NT> temp_p(d, 0.0);
    for (int j = 1; j <= d; ++j) {
        double xj = glp_get_col_prim(lp, j);
        temp_p[j-1] = static_cast<NT>(xj);
    }
    Point xc(d, temp_p.begin(), temp_p.end());
    double radius = glp_get_obj_val(lp);

    glp_delete_prob(lp);
    delete[] ia;
    delete[] ja;
    delete[] ar;

    return { xc, radius };
}



std::pair<Point, double> solve_glpk_interior(const HPOLYTOPE& P) {

    //glp_term_out(GLP_OFF);

    Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> A = P.get_mat();
    Eigen::Matrix<NT, Eigen::Dynamic, 1>            b = P.get_vec();
    int m = static_cast<int>(A.rows());
    int d = static_cast<int>(A.cols());

    std::pair<Point,double> fail = { Point(1), -1.0 };

    glp_prob *lp = glp_create_prob();
    if (!lp) return fail;
    glp_set_prob_name(lp, "ChebyshevBall");
    glp_set_obj_dir(lp, GLP_MAX);

    glp_add_rows(lp, m);
    for (int i = 1; i <= m; ++i) {
        glp_set_row_bnds(lp, i, GLP_UP, 0.0, static_cast<double>(b(i-1)));
    }

    glp_add_cols(lp, d + 1);

    for (int j = 1; j <= d; ++j) {
        glp_set_col_bnds(lp, j, GLP_FR, 0.0, 0.0);
        glp_set_obj_coef(lp, j, 0.0);
    }
    glp_set_col_bnds(lp, d + 1, GLP_LO, 0.0, 0.0);
    glp_set_obj_coef(lp, d + 1, 1.0);

    int nz = m * (d + 1);
    std::vector<int> ia(1 + nz), ja(1 + nz);
    std::vector<double> ar(1 + nz);
    int idx = 1;
    for (int i = 0; i < m; ++i) {
        double norm_ai = A.row(i).norm();
        for (int j = 0; j < d; ++j) {
            ia[idx] = i + 1;       
            ja[idx] = j + 1;      
            ar[idx] = static_cast<double>(A(i, j));
            ++idx;
        }
        ia[idx] = i + 1;
        ja[idx] = d + 1;
        ar[idx] = norm_ai;
        ++idx;
    }
    glp_load_matrix(lp, nz, ia.data(), ja.data(), ar.data());

    
    {
        glp_smcp smcp;
        glp_init_smcp(&smcp);
        smcp.msg_lev = GLP_MSG_OFF;   
        smcp.meth    = GLP_PRIMAL;    
        int sret = glp_simplex(lp, &smcp);
        if (sret != 0) {
            glp_delete_prob(lp);
            return fail;
        }
        int sstatus = glp_get_status(lp);
        if (sstatus != GLP_OPT && sstatus != GLP_FEAS) {
            glp_delete_prob(lp);
            return fail;
        }
    }

    {
        glp_iptcp iparm;
        glp_init_iptcp(&iparm);

        iparm.msg_lev = GLP_MSG_OFF;
        //iparm.it_lim = 100;     
        int iret = glp_interior(lp, &iparm);
        if (iret != 0) {
            glp_delete_prob(lp);
            return fail;
        }

        glp_exact(lp, NULL);

        int istatus = glp_get_status(lp);
        /*if (istatus != GLP_OPT) {
            glp_delete_prob(lp);
            return fail;
        }*/
    }

    std::vector<NT> sol_x(d);
    for (int j = 0; j < d; ++j) {
        sol_x[j] = static_cast<NT>(glp_get_col_prim(lp, j + 1));
    }
    double radius = glp_get_col_prim(lp, d + 1);

    glp_delete_prob(lp);

    Point xc(d, sol_x.begin(), sol_x.end());
    return { xc, radius };
}

