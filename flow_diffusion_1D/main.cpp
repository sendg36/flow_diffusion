#include <cstdio>
#include <windows.h>
#include <cmath>
#include <vector>
#include <functional>
#include <array>
#include "Mesh.h"
#include "Element.h"
#include "Assembler.h"
#include "Solver.h"
#include "Post.h"

// ============================================================
// main.cpp —— 两个算例
//
// 算例一：扩散-反应
//   -(1·u')' + 1·u = x^2 + x - 1,  精确解 u = x^2 + x + 1
//   边界：左 Neumann u'(0)=1，右 Robin u'(1)+u(1)=6
//   （通量形式：左端 -u'(0)=-1；右端 u'(1) = 6 - u(1)）
//
// 算例二：对流-扩散
//   -0.01·u'' + 1·u' = 0
//   精确解 u = (e^(x/0.01) - 1)/(e^(100) - 1)
//   边界：两端 Dirichlet u(0)=0, u(1)=1
// ============================================================

using Fn = std::function<double(double)>;

struct Problem {
    Fn p = [](double) { return 1.0; };
    Fn q = [](double) { return 0.0; };
    Fn f = [](double) { return 0.0; };
    double b = 0.0;
    bool convection = false;
    ConvScheme scheme = ConvScheme::center;
};

struct DirichletBC { int node; double value; };
struct NaturalBC { int node; double g; double beta; };

[[nodiscard]] inline std::vector<double> solve(
    int N, const Problem& prob,
    const std::vector<DirichletBC>& essential = {},
    const std::vector<NaturalBC>& natural = {}) {
    Mesh1D mesh(N, 0.0, 1.0);
    Assembler asb(N + 1);

    for (int e = 0; e < N; ++e) {
        std::array<std::array<double, 2>, 2> Ke, Ce;
        std::array<double, 2> Fe;
        Element::compute(mesh.h, mesh.x[e], prob.p, prob.q, prob.f, Ke, Fe);
        if (prob.convection) {
            Element::convection(prob.scheme, prob.b, Ce);
            for (int m = 0; m < 2; ++m)
                for (int k = 0; k < 2; ++k) Ke[m][k] += Ce[m][k];
        }
        asb.addElement({ e, e + 1 }, Ke, Fe);
    }

    for (const auto& bc : natural)
        asb.addNaturalBC(bc.node, bc.g, bc.beta);

    for (const auto& bc : essential)
        asb.addDirichletBC(bc.node, bc.value);

    return gaussSolve(asb.A, asb.F);
}

// ------------------------------------------------------------
// 算例一：扩散-反应
//   左 Neumann：-u'(0) = -1  →  addNaturalBC(0, g=-1, beta=0)
//   右 Robin：u'(1) = 6 - u(1)  ⇔  u'(1)+u(1)=6
//             →  addNaturalBC(N, g=6, beta=1)
// ------------------------------------------------------------
void runDiffusion() {
    printf("===== 算例一：扩散反应 =====\n");
    printf("-(1*u')' + 1*u = x^2 + x - 1,  exact u = x^2 + x + 1\n");
    printf("BC: left Neumann u'(0)=1, right Robin u'(1)+u(1)=6\n\n");

    Problem prob;
    prob.p = [](double) { return 1.0; };
    prob.q = [](double) { return 1.0; };
    prob.f = [](double x) { return x * x + x - 1.0; };

    printf("N        E_L2        rateL2      E_H1        rateH1\n");
    double pL2 = 0.0, pH1 = 0.0;
    for (int N : { 4, 8, 16, 32 }) {
        std::vector<NaturalBC> nbc = {
            { 0, -1.0, 0.0 },      // 左 Neumann：-u'(0) = -1（外法向 n=-1）
            { N,  6.0, 1.0 }       // 右 Robin：u'(1) = 6 - u(1)
        };
        const auto u = solve(N, prob, {}, nbc);
        Mesh1D mesh(N, 0.0, 1.0);
        const auto [eL2, eH1] = computeErrors(mesh, u,
            [](double x) { return x * x + x + 1.0; },
            [](double x) { return 2.0 * x + 1.0; });
        char rL2[16] = "      --", rH1[16] = "      --";
        if (pL2 > 0) snprintf(rL2, sizeof(rL2), "%8.4f", std::log2(pL2 / eL2));
        if (pH1 > 0) snprintf(rH1, sizeof(rH1), "%8.4f", std::log2(pH1 / eH1));
        printf("%-4d  %.5e   %s   %.5e   %s\n", N, eL2, rL2, eH1, rH1);
        pL2 = eL2; pH1 = eH1;
    }

    printf("\n[N=4 numerical vs exact]\n");
    std::vector<NaturalBC> nbc4 = { {0, -1.0, 0.0}, {4, 6.0, 1.0} };
    const auto u4 = solve(4, prob, {}, nbc4);
    Mesh1D mesh4(4, 0.0, 1.0);
    for (int i = 0; i <= 4; ++i)
        printf("  x=%.4f  u=%.6f  exact=%.6f\n",
            mesh4.x[i], u4[i], mesh4.x[i] * mesh4.x[i] + mesh4.x[i] + 1.0);
}

// ------------------------------------------------------------
// 算例二：对流-扩散
// ------------------------------------------------------------
void runConvection() {
    printf("\n===== 算例二：对流-扩散 =====\n");
    printf("-0.01*u'' + 1*u' = 0\n");
    printf("exact u = (e^(x/0.01) - 1)/(e^(100) - 1)\n");
    printf("BC: Dirichlet u(0)=0, u(1)=1\n\n");

    const double eps = 0.01, b = 1.0;

    Problem prob;
    prob.p = [eps](double) { return eps; };
    prob.q = [](double) { return 0.0; };
    prob.f = [](double) { return 0.0; };
    prob.b = b;
    prob.convection = true;

    auto exactU = [=](double x) {
        return (std::exp(b * x / eps) - 1.0) / (std::exp(b / eps) - 1.0);
        };
    auto exactDU = [=](double x) {
        return b / eps * std::exp(b * x / eps) / (std::exp(b / eps) - 1.0);
        };

    // 中心格式
    printf("[中心格式 center]\n");
    printf("N        E_L2        rateL2      E_H1        rateH1\n");
    prob.scheme = ConvScheme::center;
    double pL2 = 0.0, pH1 = 0.0;
    for (int N : { 4, 8, 16, 32, 64 }) {
        std::vector<DirichletBC> dbc = { {0, 0.0}, {N, 1.0} };
        const auto u = solve(N, prob, dbc, {});
        Mesh1D mesh(N, 0.0, 1.0);
        const auto [eL2, eH1] = computeErrors(mesh, u, exactU, exactDU);
        char rL2[16] = "      --", rH1[16] = "      --";
        if (pL2 > 0) snprintf(rL2, sizeof(rL2), "%8.4f", std::log2(pL2 / eL2));
        if (pH1 > 0) snprintf(rH1, sizeof(rH1), "%8.4f", std::log2(pH1 / eH1));
        printf("%-4d  %.5e   %s   %.5e   %s\n", N, eL2, rL2, eH1, rH1);
        pL2 = eL2; pH1 = eH1;
    }

    // 迎风格式
    printf("\n[迎风格式 upwind]\n");
    printf("N        E_L2        rateL2      E_H1        rateH1\n");
    prob.scheme = ConvScheme::upwind;
    pL2 = pH1 = 0.0;
    for (int N : { 4, 8, 16, 32, 64 }) {
        std::vector<DirichletBC> dbc = { {0, 0.0}, {N, 1.0} };
        const auto u = solve(N, prob, dbc, {});
        Mesh1D mesh(N, 0.0, 1.0);
        const auto [eL2, eH1] = computeErrors(mesh, u, exactU, exactDU);
        char rL2[16] = "      --", rH1[16] = "      --";
        if (pL2 > 0) snprintf(rL2, sizeof(rL2), "%8.4f", std::log2(pL2 / eL2));
        if (pH1 > 0) snprintf(rH1, sizeof(rH1), "%8.4f", std::log2(pH1 / eH1));
        printf("%-4d  %.5e   %s   %.5e   %s\n", N, eL2, rL2, eH1, rH1);
        pL2 = eL2; pH1 = eH1;
    }

    // N=4 中心格式，观察振荡
    printf("\n[N=4 中心格式，Pe=12.5，观察振荡]\n");
    prob.scheme = ConvScheme::center;
    std::vector<DirichletBC> dbc4 = { {0, 0.0}, {4, 1.0} };
    const auto u4c = solve(4, prob, dbc4, {});
    Mesh1D mesh4(4, 0.0, 1.0);
    for (int i = 0; i <= 4; ++i)
        printf("  x=%.4f  u=%.6f  exact=%.6f%s\n",
            mesh4.x[i], u4c[i], exactU(mesh4.x[i]),
            (u4c[i] < 0.0 || u4c[i] > 1.0) ? "   <-- 振荡" : "");

    // N=4 迎风格式
    printf("\n[N=4 迎风格式，单调]\n");
    prob.scheme = ConvScheme::upwind;
    const auto u4u = solve(4, prob, dbc4, {});
    for (int i = 0; i <= 4; ++i)
        printf("  x=%.4f  u=%.6f  exact=%.6f\n",
            mesh4.x[i], u4u[i], exactU(mesh4.x[i]));
}

int main() {
    SetConsoleOutputCP(65001);
    runDiffusion();
    runConvection();
    printf("\n全部完成。\n");
    return 0;
}