// Element.h —— P1 三角形单元
// 适配方程 -eps*Δu + b·∇u + c*u = f，单元内系数视为常数（闭式公式的前提）。
//
// 全部积分在面积坐标 (λ1,λ2) 下做，换元 Jacobi = 2Δ：
//   ∫λ1^a1 λ2^a2 λ3^a3 dxdy = 2Δ·a1!a2!a3!/(a1+a2+a3+2)!   （讲义 4.1.55）
// 载荷走三边中点求积：对二次多项式精确，f 线性时 F 是精确值。

#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include "Mesh.h"

class Element2D
{
public:
    Element2D(const Mesh2D& mesh, int e)
    {
        const auto& tri = mesh.element(e);
        for (int i = 0; i < 3; ++i) {
            vertIdx_[i] = tri[i];
            const auto& p = mesh.node(tri[i]);
            x_[i] = p[0];
            y_[i] = p[1];
        }
        area2_ = signedArea2();          // 2Δ = 换元 Jacobi
        assert(area2_ > 0.0);            // Mesh 已保证逆时针
        area_ = 0.5 * area2_;
        computeGradLambda();
    }

    double area()  const { return area_; }
    double area2() const { return area2_; }
    const std::array<int, 3>& vertexIndices() const { return vertIdx_; }

    // 扩散阵 K = eps·Δ·BᵀB，B = [∇λ1 ∇λ2 ∇λ3]
    // Gram 矩阵 ⇒ 对称；Poisson 总矩阵因此对称正定。
    std::array<std::array<double, 3>, 3> diffusionMatrix(double eps) const
    {
        std::array<std::array<double, 3>, 3> K;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                K[i][j] = eps * area_ * (gradLambda_[i][0] * gradLambda_[j][0]
                                       + gradLambda_[i][1] * gradLambda_[j][1]);
        return K;
    }

    // 对流阵 C_ij = (b·∇λ_j)·2Δ·∫λ_i = (b·∇λ_j)·Δ/3
    // 每列相同 ⇒ 外积结构；行和 = Δ/3·(b·B1) = 0。
    // 非对称：b≠0 时 CG 的前提没了，得用 BiCGStab。
    std::array<std::array<double, 3>, 3> convectionMatrix(double bx, double by) const
    {
        std::array<std::array<double, 3>, 3> C;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                const double bgrad = bx * gradLambda_[j][0] + by * gradLambda_[j][1];
                C[i][j] = bgrad * area_ / 3.0;
            }
            assert(std::abs(C[i][0] + C[i][1] + C[i][2]) < 1e-12);
        }
        return C;
    }

    // 质量阵：M_ij = c·2Δ·∫λ_iλ_j，对角 cΔ/6、非对角 cΔ/12
    std::array<std::array<double, 3>, 3> reactionMatrix(double c) const
    {
        const double diag = c * area_ / 6.0;
        const double offd = c * area_ / 12.0;
        return {{ {diag, offd, offd},
                  {offd, diag, offd},
                  {offd, offd, diag} }};
    }

    std::array<std::array<double, 3>, 3>
    elementMatrix(double eps, double bx, double by, double c) const
    {
        auto K = diffusionMatrix(eps);
        const auto C = convectionMatrix(bx, by);
        const auto M = reactionMatrix(c);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                K[i][j] += C[i][j] + M[i][j];
        return K;
    }

    // 载荷：三个边中点求积，权 Δ/3，x(P) 由换元 x = Σλ_i x_i 反算
    template <typename Func>
    std::array<double, 3> loadVector(Func f) const
    {
        static const double LAM[3][3] = {
            { 0.0, 0.5, 0.5 },   // 边 P1P2 中点
            { 0.5, 0.0, 0.5 },   // 边 P2P0 中点
            { 0.5, 0.5, 0.0 }    // 边 P0P1 中点
        };
        static const double RHO = 1.0 / 6.0;   // 参考单元上的权

        std::array<double, 3> F = { 0.0, 0.0, 0.0 };
        for (int m = 0; m < 3; ++m) {
            double px = 0.0, py = 0.0;
            for (int i = 0; i < 3; ++i) {
                px += LAM[m][i] * x_[i];
                py += LAM[m][i] * y_[i];
            }
            const double w = area2_ * RHO * f(px, py);
            for (int i = 0; i < 3; ++i)
                F[i] += w * LAM[m][i];
        }
        return F;
    }

    // 边通量 ∫_edge g·λ_i ds（Neumann、Robin 的右端都要）
    // 中点求积；边上 λ 就是一元帽子函数，∫λ_a ds = |e|/2，常数/线性 g 都是准的
    template <typename Func>
    std::array<double, 2> edgeLoad(Func g, int edge) const
    {
        const int a = (edge + 1) % 3, b = (edge + 2) % 3;   // 边 edge 对向顶点 edge
        const double len = std::hypot(x_[b] - x_[a], y_[b] - y_[a]);
        const double gv = g(0.5 * (x_[a] + x_[b]), 0.5 * (y_[a] + y_[b]));
        return { gv * len / 2.0, gv * len / 2.0 };
    }

    // Robin 的 α∫_edge λ_iλ_j ds：跟体质量阵一个套路，一维闭式 |e|/6·[[2,1],[1,2]]
    //（∫N_a²=1/3，∫N_aN_b=1/6，乘 |e| 乘 α）
    std::array<std::array<double, 2>, 2> edgeRobinMatrix(double alpha, int edge) const
    {
        const int a = (edge + 1) % 3, b = (edge + 2) % 3;
        const double len = std::hypot(x_[b] - x_[a], y_[b] - y_[a]);
        const double diag = alpha * len / 3.0;   // α|e|·(1/3)
        const double offd = alpha * len / 6.0;   // α|e|·(1/6)
        return {{ { diag, offd }, { offd, diag } }};
    }

    // cot 公式重算扩散阵，独立路径，用来和 diffusionMatrix 对拍：
    // K_ii = eps(cotθ_j + cotθ_k)/2,  K_ij = -eps·cotθ_k/2（k 为 ij 对顶点）
    std::array<std::array<double, 3>, 3> diffusionMatrixCot(double eps) const
    {
        std::array<double, 3> cot;
        for (int i = 0; i < 3; ++i) {
            const int j = (i + 1) % 3, k = (i + 2) % 3;
            const double u1 = x_[j] - x_[i], v1 = y_[j] - y_[i];
            const double u2 = x_[k] - x_[i], v2 = y_[k] - y_[i];
            cot[i] = (u1 * u2 + v1 * v2) / (u1 * v2 - v1 * u2);
        }

        std::array<std::array<double, 3>, 3> K;
        for (int i = 0; i < 3; ++i) {
            const int j = (i + 1) % 3, k = (i + 2) % 3;
            K[i][i] = eps * 0.5 * (cot[j] + cot[k]);
            K[i][j] = -eps * 0.5 * cot[k];
            K[i][k] = -eps * 0.5 * cot[j];
        }
        return K;
    }

private:
    double signedArea2() const
    {
        return (x_[1] - x_[0]) * (y_[2] - y_[0])
             - (x_[2] - x_[0]) * (y_[1] - y_[0]);
    }

    // ∇λ_i = 对边法向 / (2Δ)（讲义 p.91 性质 5）
    void computeGradLambda()
    {
        gradLambda_[0] = { (y_[1] - y_[2]) / area2_, (x_[2] - x_[1]) / area2_ };
        gradLambda_[1] = { (y_[2] - y_[0]) / area2_, (x_[0] - x_[2]) / area2_ };
        gradLambda_[2] = { (y_[0] - y_[1]) / area2_, (x_[1] - x_[0]) / area2_ };

        // B1 = 0（由 Σλ_i = 1 两边求梯度）
        const double s0 = gradLambda_[0][0] + gradLambda_[1][0] + gradLambda_[2][0];
        const double s1 = gradLambda_[0][1] + gradLambda_[1][1] + gradLambda_[2][1];
        assert(std::abs(s0) < 1e-12 && std::abs(s1) < 1e-12);
    }

    std::array<int, 3>    vertIdx_;
    std::array<double, 3> x_, y_;
    double area2_, area_;
    std::array<std::array<double, 2>, 3> gradLambda_;
};
