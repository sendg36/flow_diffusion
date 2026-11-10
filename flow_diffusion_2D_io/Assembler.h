// Assembler.h —— CSR 装配 + Dirichlet 处理
// 稀疏模式：共过单元的结点才有非零。std::set 顺便给出升序，
// addTo 靠行内升序做二分定位，换容器的话这条不变量要保住。

#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <set>
#include <vector>
#include "Mesh.h"
#include "Element.h"

struct CSRMatrix
{
    int n = 0;
    std::vector<int>    row_ptr;   // n+1
    std::vector<int>    col;       // nnz，行内升序
    std::vector<double> values;

    int nnz() const { return (int)values.size(); }

    void matvec(const std::vector<double>& x, std::vector<double>& y) const
    {
        assert((int)x.size() == n);
        y.assign(n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
                y[i] += values[k] * x[col[k]];
    }

    // 调试用，热路径走 matvec
    double get(int i, int j) const
    {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k)
            if (col[k] == j) return values[k];
        return 0.0;
    }
};

class Assembler2D
{
public:
    // 常系数入口，包一层转发到变系数版本
    template <typename Func>
    static CSRMatrix assemble(const Mesh2D& mesh, double eps, double bx, double by,
                              double c, Func f, std::vector<double>& rhs)
    {
        return assemble(mesh,
                        [eps](double, double){ return eps; },
                        [bx](double, double){ return bx; },
                        [by](double, double){ return by; },
                        [c](double, double){ return c; }, f, rhs);
    }

    // 变系数版：系数函数在单元质心取值，所以单元内仍是常数
    template <typename FEps, typename FBx, typename FBy, typename FC, typename Func>
    static CSRMatrix assemble(const Mesh2D& mesh, FEps eps, FBx bx, FBy by,
                              FC c, Func f, std::vector<double>& rhs)
    {
        const int n = mesh.numNodes();
        CSRMatrix A;
        A.n = n;

        // 1. 建稀疏模式（含对角）
        std::vector<std::set<int>> nbr(n);
        for (int e = 0; e < mesh.numElements(); ++e) {
            const auto& tri = mesh.element(e);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    nbr[tri[i]].insert(tri[j]);
        }

        A.row_ptr.assign(n + 1, 0);
        for (int i = 0; i < n; ++i) {
            A.col.insert(A.col.end(), nbr[i].begin(), nbr[i].end());
            A.row_ptr[i + 1] = (int)A.col.size();
        }
        A.values.assign(A.col.size(), 0.0);
        assert((int)A.row_ptr.size() == n + 1);
        assert(A.row_ptr[n] == (int)A.col.size());

        // 2. 逐单元装配
        rhs.assign(n, 0.0);
        for (int e = 0; e < mesh.numElements(); ++e) {
            Element2D el(mesh, e);
            const auto& tri = el.vertexIndices();
            double cx = 0.0, cy = 0.0;
            for (int i = 0; i < 3; ++i) {
                const auto& p = mesh.node(tri[i]);
                cx += p[0] / 3.0; cy += p[1] / 3.0;
            }
            const auto Ke = el.elementMatrix(eps(cx, cy), bx(cx, cy), by(cx, cy), c(cx, cy));
            const auto Fe = el.loadVector(f);
            for (int i = 0; i < 3; ++i) {
                rhs[tri[i]] += Fe[i];
                for (int j = 0; j < 3; ++j)
                    addTo(A, tri[i], tri[j], Ke[i][j]);
            }
        }
        return A;
    }

    // Dirichlet：列移项 + 划行划列 + 对角置 1（非齐次也适用）
    static void applyDirichlet(CSRMatrix& A, std::vector<double>& rhs,
                               const Mesh2D& mesh,
                               const std::function<bool(int)>& isEssential,
                               const std::function<double(const std::array<double, 2>&)>& g)
    {
        const int n = mesh.numNodes();

        // 第一遍：非本质方程的 rhs 移项、清列。本质行跳过，
        // 反正第二遍要整行重写成 u_j = g_j。
        for (int j = 0; j < n; ++j) {
            if (isEssential(mesh.boundaryFlag(j))) continue;
            for (int k = A.row_ptr[j]; k < A.row_ptr[j + 1]; ++k) {
                const int colIdx = A.col[k];
                if (colIdx != j && isEssential(mesh.boundaryFlag(colIdx))) {
                    rhs[j] -= A.values[k] * g(mesh.node(colIdx));
                    A.values[k] = 0.0;
                }
            }
        }
        // 第二遍：本质行划行置一
        for (int k = 0; k < n; ++k) {
            if (!isEssential(mesh.boundaryFlag(k))) continue;
            for (int j = A.row_ptr[k]; j < A.row_ptr[k + 1]; ++j) {
                if (A.col[j] == k) A.values[j] = 1.0;
                else               A.values[j] = 0.0;
            }
            rhs[k] = g(mesh.node(k));
        }
    }

    // Neumann：ε∂u/∂n = g。分部积分多出来的 ∫_Γ g·v ds 装右端，矩阵不动。
    // 边的认法：两个端点的 flag 都过谓词才算这条边。
    template <typename Func>
    static void applyNeumann(std::vector<double>& rhs, const Mesh2D& mesh,
                             const std::function<bool(int)>& isNatural, Func g)
    {
        for (int e = 0; e < mesh.numElements(); ++e) {
            const auto& tri = mesh.element(e);
            Element2D el(mesh, e);
            for (int m = 0; m < 3; ++m) {                 // 边 m 对向顶点 m
                const int ga = tri[(m + 1) % 3], gb = tri[(m + 2) % 3];
                if (!isNatural(mesh.boundaryFlag(ga)) ||
                    !isNatural(mesh.boundaryFlag(gb))) continue;
                const auto fe = el.edgeLoad(g, m);
                rhs[ga] += fe[0];
                rhs[gb] += fe[1];
            }
        }
    }

    // Robin：ε∂u/∂n + αu = g。∫(g-αu)v ds 拆两份：∫gv 进右端（edgeLoad 复用），
    // α∫uv 进矩阵（2x2 边块往 CSR 里累加，非零模式不变）。α 取边中点值。
    // 先装 Neumann/Robin，最后 applyDirichlet 封行。
    template <typename FuncAlpha, typename FuncG>
    static void applyRobin(CSRMatrix& A, std::vector<double>& rhs,
                           const Mesh2D& mesh,
                           const std::function<bool(int)>& isRobin,
                           FuncAlpha alpha, FuncG g)
    {
        for (int e = 0; e < mesh.numElements(); ++e) {
            const auto& tri = mesh.element(e);
            Element2D el(mesh, e);
            for (int m = 0; m < 3; ++m) {
                const int ga = tri[(m + 1) % 3], gb = tri[(m + 2) % 3];
                if (!isRobin(mesh.boundaryFlag(ga)) ||
                    !isRobin(mesh.boundaryFlag(gb))) continue;

                const auto& P = mesh.node(ga);
                const auto& Q = mesh.node(gb);
                const double mx = 0.5 * (P[0] + Q[0]), my = 0.5 * (P[1] + Q[1]);

                const auto Me = el.edgeRobinMatrix(alpha(mx, my), m);   // α∫λᵢλⱼ
                addTo(A, ga, ga, Me[0][0]);
                addTo(A, ga, gb, Me[0][1]);
                addTo(A, gb, ga, Me[1][0]);
                addTo(A, gb, gb, Me[1][1]);

                const auto fe = el.edgeLoad(g, m);                      // ∫g·v
                rhs[ga] += fe[0];
                rhs[gb] += fe[1];
            }
        }
    }

private:
    static void addTo(CSRMatrix& A, int i, int j, double v)
    {
        const auto beg = A.row_ptr[i], end = A.row_ptr[i + 1];
        auto it = std::lower_bound(A.col.begin() + beg, A.col.begin() + end, j);
        assert(it != A.col.begin() + end && *it == j);
        A.values[it - A.col.begin()] += v;
    }
};
