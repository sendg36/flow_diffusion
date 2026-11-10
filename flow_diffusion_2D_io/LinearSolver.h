// LinearSolver.h —— 接口 3：求解器插槽
//
// 主流程只依赖 solve(A, b)。手写求解器一份（包装 Solver.h），
// Eigen SparseLU 一份（本机装 Eigen 后定义 FEM2D_WITH_EIGEN 启用）。
// 选型纪律：对流扩散矩阵非对称，不能用 Eigen 的 SparseCG/SimplicialLDLT，
// 直接法对拍用 SparseLU（见 main.cpp T6b 的 CG 失效实验）。

#pragma once

#include <vector>
#include "Assembler.h"
#include "Solver.h"

class LinearSolver
{
public:
    virtual ~LinearSolver() = default;

    // x = A⁻¹b（x 初值由实现决定，一般清 0）
    virtual std::vector<double> solve(const CSRMatrix& A,
                                      const std::vector<double>& b) = 0;

    // 最近一次求解的相对残差历史（调参/对拍用）
    const std::vector<double>& history() const { return hist_; }

protected:
    std::vector<double> hist_;
};

// ---- 手写版（包装 femsolver::cg / bicgstab）----
class CGSolver : public LinearSolver
{
public:
    CGSolver(double tol = 1e-10, int maxIter = 20000, bool jacobi = true)
        : tol_(tol), maxIter_(maxIter), jacobi_(jacobi) {}

    std::vector<double> solve(const CSRMatrix& A,
                              const std::vector<double>& b) override
    {
        std::vector<double> x;
        femsolver::cg(A, b, x, tol_, maxIter_, jacobi_, hist_);
        return x;
    }

private:
    double tol_; int maxIter_; bool jacobi_;
};

class BiCGStabSolver : public LinearSolver
{
public:
    BiCGStabSolver(double tol = 1e-10, int maxIter = 20000, bool jacobi = true)
        : tol_(tol), maxIter_(maxIter), jacobi_(jacobi) {}

    std::vector<double> solve(const CSRMatrix& A,
                              const std::vector<double>& b) override
    {
        std::vector<double> x;
        femsolver::bicgstab(A, b, x, tol_, maxIter_, jacobi_, hist_);
        return x;
    }

private:
    double tol_; int maxIter_; bool jacobi_;
};

// ---- Eigen 版（本机启用：-DFEM2D_WITH_EIGEN -I/path/to/eigen3）----
#ifdef FEM2D_WITH_EIGEN
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

class EigenSparseLU : public LinearSolver
{
public:
    std::vector<double> solve(const CSRMatrix& A,
                              const std::vector<double>& b) override
    {
        // CSR（行内升序）→ Eigen 压缩行存：外层循环一次推入
        typedef Eigen::Triplet<double> T;
        std::vector<T> trips;
        trips.reserve(A.nnz());
        for (int i = 0; i < A.n; ++i)
            for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k)
                trips.emplace_back(i, A.col[k], A.values[k]);

        Eigen::SparseMatrix<double> M(A.n, A.n);
        M.setFromTriplets(trips.begin(), trips.end());

        Eigen::SparseLU<Eigen::SparseMatrix<double>> lu(M);
        Eigen::VectorXd x = lu.solve(Eigen::Map<const Eigen::VectorXd>(b.data(), A.n));

        hist_.assign(1, (M * x - Eigen::Map<const Eigen::VectorXd>(b.data(), A.n)).norm()
                        / Eigen::Map<const Eigen::VectorXd>(b.data(), A.n).norm());
        return std::vector<double>(x.data(), x.data() + A.n);
    }
};
#endif // FEM2D_WITH_EIGEN
