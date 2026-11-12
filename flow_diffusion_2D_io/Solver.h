// Solver.h —— 线性求解：稀疏 LU（CSR 上直接分解，部分主元）+ CG（可带 Jacobi）+ BiCGStab

#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
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

    // 稀疏 LU：右看式逐行消元，行存 std::map，fill-in 来了现插，省得先做符号分解。
    // 没做排序（AMD/嵌套剖分以后想要再说），fill 偏大——正好留给 main 里的统计当现象看。
    // 用法：SparseLU lu(A); auto x = lu.solve(b);
    // 只配当裁判：n 上几万，fill 和时间都顶不住，那种规模回头用上面的迭代法。
    class SparseLU
    {
    public:
        explicit SparseLU(const CSRMatrix& A) { factor(A); }

        int nnzA()  const { return nnzA_; }
        int nnzL()  const { return nnzL_; }          // 严格下三角部分
        int nnzU()  const { return nnzU_; }         // 含对角
        double fillRatio() const { return (double)(nnzL_ + nnzU_) / std::max(1, nnzA_); }

        // Ly = b 前代 + Ux = y 回代
        std::vector<double> solve(const std::vector<double>& b) const
        {
            const int n = n_;
            assert((int)b.size() == n);

            std::vector<double> y(n), x(n);
            for (int i = 0; i < n; ++i) {                       // L 单位对角，前代
                double s = b[i];
                for (int k = rowPtrL_[i]; k < rowPtrL_[i + 1]; ++k)
                    s -= valL_[k] * y[colL_[k]];
                y[i] = s;
            }
            for (int i = n - 1; i >= 0; --i) {                  // U 回代
                double s = y[i];
                for (int k = rowPtrU_[i]; k < rowPtrU_[i + 1]; ++k)
                    s -= valU_[k] * x[colU_[k]];
                x[i] = s / diagU_[i];
            }
            return x;
        }

    private:
        void factor(const CSRMatrix& A)
        {
            const int n = A.n;
            n_ = n;
            nnzA_ = A.nnz();

            // CSR 先拷成逐行 map，列号自带升序，后面所有消元都在 map 上原地做
            std::vector<std::map<int, double>> row(n);
            for (int i = 0; i < n; ++i)
                for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k)
                    row[i][A.col[k]] = A.values[k];

            // 右看式消元：第 k 步把 row[k] 钉成 U 的第 k 行，
            // 列 k 在 k+1..n-1 行里的值变成 L 因子，剩下的往右下传播——fill 就在这一步冒出来
            for (int k = 0; k < n; ++k) {
                // 部分主元：行 k..n-1 里列 k 谁绝对值大用谁
                int piv = k;
                double best = 0.0;
                for (int i = k; i < n; ++i) {
                    const auto it = row[i].find(k);
                    if (it != row[i].end() && std::abs(it->second) > best) {
                        best = std::abs(it->second);
                        piv = i;
                    }
                }
                assert(best > 1e-30);   // 零主元：矩阵奇异（纯 Neumann 不钉点就会踩这里）
                if (piv != k) std::swap(row[piv], row[k]);

                const double d = row[k][k];
                // U 第 k 行先拍快照（列 > k 的部分），等下所有行的更新都拿它当驱动
                const std::vector<std::pair<int, double>> uRow(row[k].lower_bound(k + 1), row[k].end());

                for (int i = k + 1; i < n; ++i) {
                    const auto it = row[i].find(k);
                    if (it == row[i].end()) continue;   // 本行不在当前消元列上，跳过
                    const double lik = it->second / d;
                    row[i].erase(it);
                    if (lik == 0.0) continue;
                    row[i][k] = lik;                    // L 因子留在严格下三角位
                    for (const auto& [j, ukj] : uRow)
                        row[i][j] -= lik * ukj;         // map 没有的列会自动插进去——fill-in 就是这么来的
                }
            }

            // 收摊：列号比行号小的进 L，其余的进 U（含对角）
            rowPtrL_.assign(n + 1, 0);
            rowPtrU_.assign(n + 1, 0);
            diagU_.assign(n, 0.0);
            for (int i = 0; i < n; ++i) {
                for (const auto& [j, v] : row[i]) {
                    if (v == 0.0) continue;             // 消元产生的显式零，不存
                    if (j < i) { colL_.push_back(j); valL_.push_back(v); }
                    else { colU_.push_back(j); valU_.push_back(v); if (j == i) diagU_[i] = v; }
                }
                rowPtrL_[i + 1] = (int)colL_.size();
                rowPtrU_[i + 1] = (int)colU_.size();
            }
            nnzL_ = (int)colL_.size();
            nnzU_ = (int)colU_.size();
            for (int i = 0; i < n; ++i)
                assert(diagU_[i] != 0.0);
        }

        int n_ = 0;
        int nnzA_ = 0, nnzL_ = 0, nnzU_ = 0;
        std::vector<int>    rowPtrL_, colL_;
        std::vector<double> valL_;
        std::vector<int>    rowPtrU_, colU_;
        std::vector<double> valU_, diagU_;
    };

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
                r[i] = s[i] - omega * t[i];
            }
            const double relres = norm2(r) / bnorm;
            history.push_back(relres);
            if (relres < tol) return it + 1;
            if (std::abs(omega) < 1e-300) break;
        }
        return maxIter;   // 没收敛
    }

}