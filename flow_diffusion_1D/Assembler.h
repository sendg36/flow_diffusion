#pragma once
#include <vector>
#include <array>

struct Assembler {
    int n;
    std::vector<std::vector<double>> A;
    std::vector<double> F;

    explicit Assembler(int nNodes)
        : n(nNodes),
        A(nNodes, std::vector<double>(nNodes, 0.0)),
        F(nNodes, 0.0) {
    }

    // 装配：单元阵投放到全局
    void addElement(const std::array<int, 2>& nodes,
        const std::array<std::array<double, 2>, 2>& Ae,
        const std::array<double, 2>& Fe) {
        for (int m = 0; m < 2; ++m) {
            for (int k = 0; k < 2; ++k)
                A[nodes[m]][nodes[k]] += Ae[m][k];
            F[nodes[m]] += Fe[m];
        }
    }

    // 自然边界：Neumann(g=0) 或 Robin
    void addNaturalBC(int k, double g, double beta) {
        A[k][k] += beta;
        F[k] += g;
    }

    // 本质边界：u_k = d
    void addDirichletBC(int k, double d) {
        for (int j = 0; j < n; ++j) {
            if (j != k) {
                F[j] -= A[j][k] * d;   // 先扣右端
                A[j][k] = 0.0;         // 再清第 k 列
                A[k][j] = 0.0;         // 清第 k 行
            }
        }
        A[k][k] = 1.0;
        F[k] = d;
    }
};