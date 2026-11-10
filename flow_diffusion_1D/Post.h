#pragma once
#include <vector>
#include <cmath>
#include "Mesh.h"

// ============================================================
// Post.h —— 后处理：误差与收敛阶
//
// L2 误差：
//   E_L2^2 = Σ_e ∫ (u_h - u)^2 dx
//
// H1 误差：
//   E_H1^2 = Σ_e ∫ (u_h' - u')^2 dx
//
// 单元上 u_h' = (u_1 - u_0) / h 是常数
//
// 收敛阶：
//   rate = log2(E(h) / E(2h))
// ============================================================

struct Errors { double L2, H1; };

template <typename U, typename DU>
[[nodiscard]] Errors computeErrors(const Mesh1D& mesh,
    const std::vector<double>& u,
    U exactU, DU exactDU) {
    const double g1 = 0.5 - 0.28867513459481287;
    const double g2 = 0.5 + 0.28867513459481287;
    const double gw = 0.5;
    double eL2 = 0.0, eH1 = 0.0;

    for (int e = 0; e < mesh.N; ++e) {
        const double x0 = mesh.x[e];
        const double h = mesh.h;
        const double u0 = u[e], u1 = u[e + 1];
        const double duh = (u1 - u0) / h;   // u_h' 在单元上是常数

        const double xiG[2] = { g1, g2 };
        for (int gp = 0; gp < 2; ++gp) {
            const double xi = xiG[gp];
            const double x = x0 + h * xi;
            const double uh = u0 * (1.0 - xi) + u1 * xi;
            eL2 += gw * h * (uh - exactU(x)) * (uh - exactU(x));
            eH1 += gw * h * (duh - exactDU(x)) * (duh - exactDU(x));
        }
    }
    return { std::sqrt(eL2), std::sqrt(eH1) };
}
