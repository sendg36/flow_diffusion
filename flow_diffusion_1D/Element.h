#pragma once
#include <array>
#include <cmath>

enum class ConvScheme { center, upwind };

struct Element
{
    static double g1() { return 0.5 - 0.5 / std::sqrt(3.0); }
    static double g2() { return 0.5 + 0.5 / std::sqrt(3.0); }
    static constexpr double w = 0.5;
    static constexpr int nGauss = 2;

    static constexpr double N(int n, double x) { return n == 0 ? (1 - x) : x; };
    static constexpr double dN(int n, double x) { return n == 0 ? -1 : 1; };

    template <typename P, typename Q, typename F>
    static void compute(double h, double x0, P p, Q q, F f,
        std::array<std::array<double, 2>, 2>& Ke,
        std::array<double, 2>& Fe) {
        Ke = {}; Fe = {};
        const double xiG[nGauss] = { g1(), g2() };
        for (int gp = 0; gp < nGauss; ++gp) {
            const double xi = xiG[gp];
            const double x = x0 + h * xi;
            const double fh = f(x);
            for (int m = 0; m < 2; ++m) {
                Fe[m] += w * h * fh * N(m, xi);
                for (int n = 0; n < 2; ++n)
                    Ke[m][n] += w * (p(x) / h * dN(n, xi) * dN(m, xi)
                        + h * q(x) * N(n, xi) * N(m, xi));
            }
        }
    }

    // 中心格式
    template <typename B>
    static void convectionCentral(B b, std::array<std::array<double, 2>, 2>& Ce) {
        Ce = {};
        const double xiG[nGauss] = { g1(), g2() };
        for (int gp = 0; gp < nGauss; ++gp)
            for (int m = 0; m < 2; ++m)
                for (int n = 0; n < 2; ++n)
                    Ce[m][n] += w * b * dN(n, xiG[gp]) * N(m, xiG[gp]);
    }

    // 迎风格式（b > 0）
    static void convectionUpwind(double b, std::array<std::array<double, 2>, 2>& Ce) {
        Ce = {};
        if (b > 0) { Ce[1][0] = -b; Ce[1][1] = b; }
    }

    // 派发
    static void convection(ConvScheme s, double b,
        std::array<std::array<double, 2>, 2>& Ce) {
        if (s == ConvScheme::center) convectionCentral(b, Ce);
        else                         convectionUpwind(b, Ce);
    }
};