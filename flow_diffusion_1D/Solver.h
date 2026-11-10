#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <utility>

[[nodiscard]] inline std::vector<double> gaussSolve(
    std::vector<std::vector<double>> A,
    std::vector<double> b) {
    const int n = static_cast<int>(b.size());

    for (int k = 0; k < n; ++k) {
        int piv = k;
        for (int i = k + 1; i < n; ++i)
            if (std::fabs(A[i][k]) > std::fabs(A[piv][k])) piv = i;

        if (std::fabs(A[piv][k]) < 1e-14)
            throw std::runtime_error("matrix is singular or nearly singular");

        std::swap(A[piv], A[k]);
        std::swap(b[piv], b[k]);

        for (int i = k + 1; i < n; ++i) {
            const double lik = A[i][k] / A[k][k];
            for (int j = k; j < n; ++j)
                A[i][j] -= lik * A[k][j];
            b[i] -= lik * b[k];
        }
    }

    std::vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < n; ++j)
            s -= A[i][j] * x[j];
        x[i] = s / A[i][i];
    }

    return x;
}
[[nodiscard]] inline std::vector<double> thomasSolve(
    std::vector<std::vector<double>> A, std::vector<double> b)
{
    const int n = static_cast<int>(b.size());

    // 追：顺流消元
    for (int i = 0; i < n - 1; ++i) {
        if (std::fabs(A[i][i]) < 1e-14)
            throw std::runtime_error("zero pivot in thomasSolve");
        const double lik = -A[i + 1][i] / A[i][i];
        A[i + 1][i + 1] += lik * A[i][i + 1];
        b[i + 1] += lik * b[i];
    }

    // 赶：倒回代
    std::vector<double> x(n);
    x[n - 1] = b[n - 1] / A[n - 1][n - 1];
    for (int i = n - 2; i >= 0; --i)
        x[i] = (b[i] - A[i][i + 1] * x[i + 1]) / A[i][i];

    return x;
}