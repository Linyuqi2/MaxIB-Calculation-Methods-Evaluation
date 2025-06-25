#include "solve_mosek.hpp"
#include "lp_oracles/solve_lp.h"

#include <lp_lib.h>
#include <glpk.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>
#include <mosek.h>


static std::pair<Point, double> fail()
{
    return std::make_pair(Point(1), -1.0);
}


static bool build_mosek_task(MSKenv_t& env, MSKtask_t& task, const HPOLYTOPE& P)
{
    // 1. 取出 A, b
    Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> A = P.get_mat();
    Eigen::Matrix<NT, Eigen::Dynamic, 1>               b = P.get_vec();
    int m = static_cast<int>(A.rows());
    int d = static_cast<int>(A.cols());
    int n = d + 1;  
    
    if (MSK_makeenv(&env, nullptr) != MSK_RES_OK) return false;
    if (MSK_maketask(env, m, n, &task) != MSK_RES_OK) {
        MSK_deleteenv(&env);
        return false;
    }

   
    if (MSK_appendvars(task, n) != MSK_RES_OK) {
        MSK_deletetask(&task);
        MSK_deleteenv(&env);
        return false;
    }
    for (int j = 0; j < d; ++j) {
        MSK_putvarname(task, j, ("x" + std::to_string(j)).c_str());
        MSK_putvarbound(task, j, MSK_BK_FR, -MSK_INFINITY, MSK_INFINITY);
        MSK_putcj(task, j, 0.0);
    }
    
    MSK_putvarname(task, d, "r");
    MSK_putvarbound(task, d, MSK_BK_LO, 0.0, MSK_INFINITY);
    MSK_putcj(task, d, 1.0);

    
    if (MSK_appendcons(task, m) != MSK_RES_OK) {
        MSK_deletetask(&task);
        MSK_deleteenv(&env);
        return false;
    }
    for (int i = 0; i < m; ++i) {
        MSK_putconname(task, i, ("con_" + std::to_string(i)).c_str());
        MSK_putconbound(task, i, MSK_BK_UP, -MSK_INFINITY, b(i));
    }

    for (int i = 0; i < m; ++i) {
        double norm_ai = A.row(i).norm();
        for (int j = 0; j < d; ++j) {
            MSK_putaij(task,
                       static_cast<MSKint32t>(i),   
                       static_cast<MSKint32t>(j),   
                       A(i,j));                     
        }
        MSK_putaij(task,
                   static_cast<MSKint32t>(i),
                   static_cast<MSKint32t>(d),
                   norm_ai);
    }

    return true;
}


std::pair<Point, double> solve_mosek_primal_simplex(const HPOLYTOPE& P)
{
    MSKenv_t  env  = nullptr;
    MSKtask_t task = nullptr;
    if (!build_mosek_task(env, task, P)) {
        std::cout<<"failed to create the task"<<std::endl;
        return fail();
    }

    int d = static_cast<int>(P.get_mat().cols());

    MSK_putobjsense(task, MSK_OBJECTIVE_SENSE_MAXIMIZE);
    MSK_putintparam(task, MSK_IPAR_OPTIMIZER, MSK_OPTIMIZER_PRIMAL_SIMPLEX);
    MSK_putintparam(task, MSK_IPAR_LOG, 0);

    MSKrescodee r = MSK_optimizetrm(task, nullptr);
    if (r != MSK_RES_OK) {
        char errmsg[MSK_MAX_STR_LEN];
        MSK_rescodetostr(r, errmsg);
        std::cerr << "[MOSEK] MSK_optimizetrm return error, code=" << r
                << "  msg=\"" << errmsg << "\"\n";

        MSK_deletetask(&task);
        MSK_deleteenv(&env);
        return fail();
    }

    MSKsolsta_enum solsta;
    MSK_getsolsta(task, MSK_SOL_BAS, &solsta);
    if (solsta != MSK_SOL_STA_OPTIMAL) {
        std::cerr << "[Primal] Not optimal, solsta=" << solsta << ")\n";
        MSK_deletetask(&task);
        MSK_deleteenv(&env);
        return fail();
    }

    int32_t nvar;
    MSK_getnumvar(task, &nvar);
    std::vector<double> xx(nvar);
    MSK_getxxslice(task, MSK_SOL_BAS, 0, nvar, xx.data());

    
    std::vector<NT> tmp(d);
    for (int j = 0; j < d; ++j) {
        tmp[j] = xx[j];
    }

    Point center(d, tmp.begin(), tmp.end());
    double radius = xx[d];

    MSK_deletetask(&task);
    MSK_deleteenv(&env);
    return std::make_pair(center, radius);
}


std::pair<Point, double> solve_mosek_dual_simplex(const HPOLYTOPE& P)
{
    MSKenv_t  env  = nullptr;
    MSKtask_t task = nullptr;
    if (!build_mosek_task(env, task, P)) {
        return fail();
    }

    int d = static_cast<int>(P.get_mat().cols());

    MSK_putobjsense(task, MSK_OBJECTIVE_SENSE_MAXIMIZE);
    MSK_putintparam(task, MSK_IPAR_OPTIMIZER, MSK_OPTIMIZER_DUAL_SIMPLEX);
    MSK_putintparam(task, MSK_IPAR_LOG, 0);

    if (MSK_optimizetrm(task, nullptr) != MSK_RES_OK) {
        MSK_deletetask(&task);
        MSK_deleteenv(&env);
        return fail();
    }

    MSKsolsta_enum solsta;
    MSK_getsolsta(task, MSK_SOL_BAS, &solsta);
    if (solsta != MSK_SOL_STA_OPTIMAL) {
        std::cerr << "[Dual] Not optimal, solsta=" << solsta << ")\n";
        MSK_deletetask(&task);
        MSK_deleteenv(&env);
        return fail();
    }

    int32_t nvar;
    MSK_getnumvar(task, &nvar);
    std::vector<double> xx(nvar);
    MSK_getxxslice(task, MSK_SOL_BAS, 0, nvar, xx.data());

    std::vector<NT> tmp(d);
    for (int j = 0; j < d; ++j) {
        tmp[j] = xx[j];
    }
    Point center(d, tmp.begin(), tmp.end());
    double radius = xx[d];

    MSK_deletetask(&task);
    MSK_deleteenv(&env);
    return std::make_pair(center, radius);
}


std::pair<Point, double> solve_mosek_interior(const HPOLYTOPE& P)
{
    MSKenv_t  env  = nullptr;
    MSKtask_t task = nullptr;
    if (!build_mosek_task(env, task, P)) {
        return fail();
    }

    int d = static_cast<int>(P.get_mat().cols());

    MSK_putobjsense(task, MSK_OBJECTIVE_SENSE_MAXIMIZE);
    MSK_putintparam(task, MSK_IPAR_OPTIMIZER, MSK_OPTIMIZER_INTPNT);
    MSK_putintparam(task, MSK_IPAR_LOG, 0);
    MSK_putdouparam(task, MSK_DPAR_INTPNT_CO_TOL_REL_GAP, 1e-8);

    if (MSK_optimizetrm(task, nullptr) != MSK_RES_OK) {
        MSK_deletetask(&task);
        MSK_deleteenv(&env);
        return fail();
    }

    MSKsolsta_enum solsta;
    MSK_getsolsta(task, MSK_SOL_ITR, &solsta);
    if (solsta != MSK_SOL_STA_OPTIMAL) {
        MSK_deletetask(&task);
        MSK_deleteenv(&env);
        return fail();
    }

    int32_t nvar;
    MSK_getnumvar(task, &nvar);
    std::vector<double> xx(nvar);
    MSK_getxxslice(task, MSK_SOL_ITR, 0, nvar, xx.data());

    std::vector<NT> tmp(d);
    for (int j = 0; j < d; ++j) {
        tmp[j] = xx[j];
    }
    Point center(d, tmp.begin(), tmp.end());
    double radius = xx[d];

    MSK_deletetask(&task);
    MSK_deleteenv(&env);
    return std::make_pair(center, radius);
}
