// Solver.h —— 线性求解：稠密直接法（小系统基准）+ CG（可带 Jacobi）+ BiCGStab
// 迭代法都留残差历史，方便看收敛过程和停滞。

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include "Assembler.h"

namespace femsolver
{

inline double dot(const std::vector<double>& a, const std::vector<double>& b)
{
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

inline double norm2(const std::vector<double>& a) { return std::sqrt(dot(a, a)); }

// 部分主元高斯消元，只用来给小系统当"真值"
inline std::vector<double> denseSolve(std::vector<std::vector<double>> A,
                                      std::vector<double> b)
{
    const int n = (int)b.size();
    for (int k = 0; k < n; ++k) {
        int piv = k;
        for (int i = k + 1; i < n; ++i)
            if (std::abs(A[i][k]) > std::abs(A[piv][k])) piv = i;
        std::swap(A[piv], A[k]);
        std::swap(b[piv], b[k]);
        assert(std::abs(A[k][k]) > 1e-14);
        for (int i = k + 1; i < n; ++i) {
            const double f = A[i][k] / A[k][k];
            for (int j = k; j < n; ++j) A[i][j] -= f * A[k][j];
            b[i] -= f * b[k];
        }
    }
    std::vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < n; ++j) s -= A[i][j] * x[j];
        x[i] = s / A[i][i];
    }
    return x;
}

inline std::vector<std::vector<double>> toDense(const CSRMatrix& A)
{
    std::vector<std::vector<double>> D(A.n, std::vector<double>(A.n, 0.0));
    for (int i = 0; i < A.n; ++i)
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k)
            D[i][A.col[k]] = A.values[k];
    return D;
}

// CG。jacobi 时 M = diag(A)。返回迭代数，history 存相对残差序列。
// 要求 A 对称正定，非对称矩阵上 CG 不保证收敛（见 main 里的停滞实验）。
inline int cg(const CSRMatrix& A, const std::vector<double>& b,
              std::vector<double>& x, double tol, int maxIter,
              bool jacobi, std::vector<double>& history)
{
    const int n = A.n;
    assert((int)b.size() == n);
    x.assign(n, 0.0);

    std::vector<double> Minv(n, 1.0);
    if (jacobi)
        for (int i = 0; i < n; ++i)
            Minv[i] = 1.0 / A.get(i, i);   // Dirichlet 处理后对角是 1，安全

    std::vector<double> r(b), z(n), p(n), Ap(n);
    A.matvec(x, Ap);
    for (int i = 0; i < n; ++i) r[i] -= Ap[i];

    auto prec = [&](const std::vector<double>& v, std::vector<double>& out) {
        if (jacobi) for (int i = 0; i < n; ++i) out[i] = Minv[i] * v[i];
        else        out = v;
    };

    prec(r, z);
    p = z;
    double rz = dot(r, z);
    const double bnorm = norm2(b);
    history.push_back(norm2(r) / bnorm);

    for (int it = 0; it < maxIter; ++it) {
        A.matvec(p, Ap);
        const double pAp = dot(p, Ap);
        if (std::abs(pAp) < 1e-300) break;
        const double alpha = rz / pAp;
        for (int i = 0; i < n; ++i) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }
        const double relres = norm2(r) / bnorm;
        history.push_back(relres);
        if (relres < tol) return it + 1;
        prec(r, z);
        const double rzNew = dot(r, z);
        const double beta = rzNew / rz;
        rz = rzNew;
        for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
    }
    return maxIter;   // 没收敛
}

// BiCGStab，非对称系统（对流项不为零）用这个。同样可带 Jacobi。
inline int bicgstab(const CSRMatrix& A, const std::vector<double>& b,
                    std::vector<double>& x, double tol, int maxIter,
                    bool jacobi, std::vector<double>& history)
{
    const int n = A.n;
    x.assign(n, 0.0);
    std::vector<double> Minv(n, 1.0);
    if (jacobi)
        for (int i = 0; i < n; ++i) Minv[i] = 1.0 / A.get(i, i);
    auto prec = [&](const std::vector<double>& v, std::vector<double>& out) {
        for (int i = 0; i < n; ++i) out[i] = jacobi ? Minv[i] * v[i] : v[i];
    };

    std::vector<double> r(b), r0(b), p(n), v(n), s(n), t(n), y(n), z(n);
    const double bnorm = norm2(b);
    history.push_back(norm2(r) / bnorm);
    double rho = 1.0, alpha = 1.0, omega = 1.0;

    for (int it = 0; it < maxIter; ++it) {
        const double rhoNew = dot(r, r0);
        if (std::abs(rhoNew) < 1e-300) break;   // rho 除零保护
        const double beta = (rhoNew / rho) * (alpha / omega);
        rho = rhoNew;
        for (int i = 0; i < n; ++i) p[i] = r[i] + beta * (p[i] - omega * v[i]);

        prec(p, y);
        A.matvec(y, v);
        alpha = rho / dot(r0, v);
        for (int i = 0; i < n; ++i) s[i] = r[i] - alpha * v[i];

        const double sNorm = norm2(s);
        history.push_back(sNorm / bnorm);
        if (sNorm / bnorm < tol) {
            for (int i = 0; i < n; ++i) x[i] += alpha * y[i];
            return it + 1;
        }
        prec(s, z);
        A.matvec(z, t);
        const double tt = dot(t, t);
        if (std::abs(tt) < 1e-300) break;       // omega 除零保护
        omega = dot(t, s) / tt;
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * y[i] + omega * z[i];
            r[i]  = s[i] - omega * t[i];
        }
        const double relres = norm2(r) / bnorm;
        history.push_back(relres);
        if (relres < tol) return it + 1;
        if (std::abs(omega) < 1e-300) break;
    }
    return maxIter;   // 没收敛
}

} // namespace femsolver
