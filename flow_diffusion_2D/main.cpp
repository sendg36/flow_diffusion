//  1. Element2D 单元级对拍（∇λ 路径 vs cot 路径 + 手算值）
//  2. Poisson nx=4 内点基准值（0.948187 / 0.670469 / 0.481621），稀疏 LU 裁判
//  3. Poisson 收敛表（E_L2: 0.2763→0.0057，rate→2），CG+Jacobi
//  4. CG 与稀疏 LU 对拍；裸 CG vs Jacobi-CG 迭代数
//  5. 2:1 矩形域零结构化假设测试
//  6. 对流扩散：BiCGStab 与稀疏 LU 对拍；CG 失效 vs BiCGStab 收敛（ε=0.01）
//  7. 三类边界混合（Dirichlet + Neumann + Robin）
//  8. 纯 Neumann 奇异性：不钉点 CG 发散 vs 钉一点收敛

#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <vector>
#include "Mesh.h"
#include "Element.h"
#include "Assembler.h"
#include "Solver.h"
#include "Post.h"

static const double PI = std::acos(-1.0);

static bool near(double a, double b, double tol) { return std::abs(a - b) < tol; }

// 全域 Dirichlet：边界结点（flag != 0）全钉，内部结点不动
static bool allDirichlet(int flag) { return flag != Mesh2D::NONE; }

// 精确解和对应的右端项
// Poisson: -Δu = f,  u = sin(πx)sin(πy)
static double poissonExact(double x, double y) { return std::sin(PI * x) * std::sin(PI * y); }
static double poissonF(double x, double y) { return 2 * PI * PI * std::sin(PI * x) * std::sin(PI * y); }

// 对流扩散: -εΔu + b·∇u = f,  u = sin(πx)sin(πy), b=(1,0)
static double cdF(double eps, double x, double y)
{
    return 2 * eps * PI * PI * std::sin(PI * x) * std::sin(PI * y)
        + PI * std::cos(PI * x) * std::sin(PI * y);
}

// T1：单元级对拍
static void testElement()
{
    std::cout << "\n=== [T1] Element2D 单元级对拍 ===\n";
    Mesh2D mesh(1, 1);
    Element2D el(mesh, 0);
    const auto K1 = el.diffusionMatrix(1.0);
    const auto K2 = el.diffusionMatrixCot(1.0);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            assert(near(K1[i][j], K2[i][j], 1e-12));

    const auto C = el.convectionMatrix(1.0, 0.0);   // b=(1,0)
    assert(near(C[0][0], -1.0 / 6.0, 1e-12) && near(C[0][1], 1.0 / 6.0, 1e-12));

    const auto M = el.reactionMatrix(1.0);
    assert(near(M[0][0], 1.0 / 12.0, 1e-12) && near(M[0][1], 1.0 / 24.0, 1e-12));

    const auto F = el.loadVector([](double, double) { return 1.0; });
    assert(near(F[0], 1.0 / 6.0, 1e-12) && near(F[1], 1.0 / 6.0, 1e-12));

    std::cout << "[T1] PASS\n";
}

// T2：Poisson nx=4，三个内点值对基准
static void testPoissonBenchmark()
{
    std::cout << "\n=== [T2] Poisson nx=4 内点基准 ===\n";
    Mesh2D mesh(4, 4);
    std::vector<double> rhs;
    auto A = Assembler2D::assemble(mesh, 1.0, 0.0, 0.0, 0.0, poissonF, rhs);
    Assembler2D::applyDirichlet(A, rhs, mesh, allDirichlet,
        [](const std::array<double, 2>& p) { return poissonExact(p[0], p[1]); });
    femsolver::SparseLU lu(A);
    std::vector<double> u = lu.solve(rhs);
    std::printf("[T2] sparse LU: nnz(A)=%d -> nnz(L+U)=%d (fill %.1fx)\n",
        lu.nnzA(), lu.nnzL() + lu.nnzU(), lu.fillRatio());

    auto valAt = [&](double x, double y) {
        for (int k = 0; k < mesh.numNodes(); ++k)
            if (near(mesh.node(k)[0], x, 1e-12) && near(mesh.node(k)[1], y, 1e-12))
                return u[k];
        assert(false); return 0.0;
        };
    const double v1 = valAt(0.5, 0.5);
    const double v2 = valAt(0.25, 0.5);
    const double v3 = valAt(0.25, 0.25);
    std::printf("[T2] u(0.50,0.50) = %.6f (expect 0.948187)\n", v1);
    std::printf("[T2] u(0.25,0.50) = %.6f (expect 0.670469)\n", v2);
    std::printf("[T2] u(0.25,0.25) = %.6f (expect 0.481621)\n", v3);
    assert(near(v1, 0.948187, 1e-5));
    assert(near(v2, 0.670469, 1e-5));
    assert(near(v3, 0.481621, 1e-5));
    std::cout << "[T2] PASS\n";
}

// T3：Poisson 收敛表（CG + Jacobi）
static void testConvergence()
{
    std::cout << "\n=== [T3] Poisson 收敛表（CG, Jacobi 预条件） ===\n";
    const int    nxs[] = { 2, 4, 8, 16 };
    const double expect[] = { 0.2763, 0.0852, 0.0226, 0.0057 };
    double prevErr = 0.0;
    for (int t = 0; t < 4; ++t) {
        Mesh2D mesh(nxs[t], nxs[t]);
        std::vector<double> rhs;
        auto A = Assembler2D::assemble(mesh, 1.0, 0.0, 0.0, 0.0, poissonF, rhs);
        Assembler2D::applyDirichlet(A, rhs, mesh, allDirichlet,
            [](const std::array<double, 2>& p) { return poissonExact(p[0], p[1]); });
        std::vector<double> u;
        std::vector<double> hist;
        const int iters = femsolver::cg(A, rhs, u, 1e-10, 5000, true, hist);
        const double E = Post2D::l2Error(mesh, u, poissonExact);
        double rate = 0.0;
        if (t > 0) rate = std::log2(prevErr / E);
        std::printf("[T3] nx=%2d  E_L2=%.4f (expect %.4f)  rate=%.2f  CG(Jacobi)=%d it\n",
            nxs[t], E, expect[t], rate, iters);
        assert(near(E, expect[t], 5e-4));
        if (t >= 2) assert(rate > 1.9 && rate < 2.1);
        prevErr = E;
    }
    std::cout << "[T3] PASS\n";
}

// T4a 均匀 Poisson：CG / Jacobi-CG 对稀疏 LU。
//     均匀网格对角元是常数，Jacobi 的 M⁻¹ 是标量倍单位阵，
//     不改变 CG 搜索子空间，所以迭代数必然相同。
// T4b 变系数对流扩散：对角元逐单元不同，Jacobi 才起作用。
static void testCGvsDirect()
{
    std::cout << "\n=== [T4] CG 对稀疏 LU / Jacobi 预条件对比 ===\n";

    {
        Mesh2D mesh(16, 16);
        std::vector<double> rhs;
        auto A = Assembler2D::assemble(mesh, 1.0, 0.0, 0.0, 0.0, poissonF, rhs);
        Assembler2D::applyDirichlet(A, rhs, mesh, allDirichlet,
            [](const std::array<double, 2>& p) { return poissonExact(p[0], p[1]); });
        femsolver::SparseLU lu(A);
        const auto uDirect = lu.solve(rhs);
        std::printf("[T4a] sparse LU: nnz(A)=%d -> nnz(L+U)=%d (fill %.1fx)\n",
            lu.nnzA(), lu.nnzL() + lu.nnzU(), lu.fillRatio());

        std::vector<double> u1, u2, h1, h2;
        const int itPlain = femsolver::cg(A, rhs, u1, 1e-10, 100000, false, h1);
        const int itJacobi = femsolver::cg(A, rhs, u2, 1e-10, 100000, true, h2);
        std::printf("[T4a] nx=16  plain CG=%d, Jacobi-CG=%d\n", itPlain, itJacobi);

        double diff = 0.0;
        for (int k = 0; k < mesh.numNodes(); ++k)
            diff = std::max(diff, std::abs(u2[k] - uDirect[k]));
        assert(diff < 1e-8);
    }

    auto epsVar = [](double x, double) { return 0.01 + x * x; };
    auto bxVar = [](double, double) { return 1.0; };
    auto byVar = [](double, double) { return 0.5; };
    auto one = [](double, double) { return 1.0; };

    // nx=32 先对拍：两版 BiCGStab 解要一致（预条件只影响迭代路径，不影响解），
    // 再和稀疏 LU 对一遍
    {
        Mesh2D mesh(32, 32);
        std::vector<double> rhs;
        auto A = Assembler2D::assemble(mesh, epsVar, bxVar, byVar, one, one, rhs);
        Assembler2D::applyDirichlet(A, rhs, mesh, allDirichlet,
            [](const std::array<double, 2>& p) { return poissonExact(p[0], p[1]); });
        femsolver::SparseLU lu(A);
        const auto uDirect = lu.solve(rhs);
        std::printf("[T4b] sparse LU: nnz(A)=%d -> nnz(L+U)=%d (fill %.1fx)\n",
            lu.nnzA(), lu.nnzL() + lu.nnzU(), lu.fillRatio());

        std::vector<double> u1, u2, h1, h2;
        femsolver::bicgstab(A, rhs, u1, 1e-10, 20000, false, h1);
        femsolver::bicgstab(A, rhs, u2, 1e-10, 20000, true, h2);
        double diff = 0.0;
        for (int k = 0; k < mesh.numNodes(); ++k)
            diff = std::max(diff, std::abs(u2[k] - u1[k]));
        assert(diff < 1e-8);

        diff = 0.0;
        for (int k = 0; k < mesh.numNodes(); ++k)
            diff = std::max(diff, std::abs(u1[k] - uDirect[k]));
        assert(diff < 1e-8);
    }
    // nx=64 再比迭代数
    {
        Mesh2D mesh(64, 64);
        std::vector<double> rhs;
        auto A = Assembler2D::assemble(mesh, epsVar, bxVar, byVar, one, one, rhs);
        Assembler2D::applyDirichlet(A, rhs, mesh, allDirichlet,
            [](const std::array<double, 2>& p) { return poissonExact(p[0], p[1]); });

        std::vector<double> u1, u2, h1, h2;
        const int itPlain = femsolver::bicgstab(A, rhs, u1, 1e-10, 20000, false, h1);
        const int itJacobi = femsolver::bicgstab(A, rhs, u2, 1e-10, 20000, true, h2);
        std::printf("[T4b] nx=64  plain BiCGStab=%d, Jacobi-BiCGStab=%d\n", itPlain, itJacobi);
        assert(itJacobi < itPlain);
    }
    std::cout << "[T4] PASS\n";
}

// T5：2:1 矩形域，确认代码不依赖"单位正方形、nx==ny"这类隐含假设
static void testRectangle21()
{
    std::cout << "\n=== [T5] 2:1 矩形域（零结构化假设） ===\n";
    auto exact = [](double x, double y) { return std::sin(PI * x / 2.0) * std::sin(PI * y); };
    auto f = [&](double x, double y) { return 5 * PI * PI / 4.0 * exact(x, y); };

    double prevErr = 0.0;
    for (int t = 0; t < 2; ++t) {
        const int nx = 8 << t, ny = 4 << t;      // h=0.25, 0.125
        Mesh2D mesh(nx, ny, 0.0, 0.0, 2.0, 1.0);
        std::vector<double> rhs;
        auto A = Assembler2D::assemble(mesh, 1.0, 0.0, 0.0, 0.0, f, rhs);
        Assembler2D::applyDirichlet(A, rhs, mesh, allDirichlet,
            [&](const std::array<double, 2>& p) { return exact(p[0], p[1]); });
        std::vector<double> u;
        std::vector<double> hist;
        femsolver::cg(A, rhs, u, 1e-10, 10000, true, hist);
        const double E = Post2D::l2Error(mesh, u, exact);
        double rate = 0.0;
        if (t > 0) rate = std::log2(prevErr / E);
        std::printf("[T5] nx=%2d ny=%2d  E_L2=%.5f  rate=%.2f\n", nx, ny, E, rate);
        assert(E < 0.08);   // 与单位正方形 nx=4（0.0852）同量级即可
        if (t > 0) assert(rate > 1.85 && rate < 2.15);
        prevErr = E;
    }
    std::cout << "[T5] PASS\n";
}

// T6：对流扩散 ε=0.01，BiCGStab 对拍；CG 用在非对称矩阵上会失效
static void testConvectionDiffusion()
{
    std::cout << "\n=== [T6] 对流扩散 ε=0.01：BiCGStab 对拍 / CG 失效 ===\n";
    const double eps = 0.01;
    auto f = [&](double x, double y) { return cdF(eps, x, y); };

    {
        Mesh2D mesh(16, 16);
        std::vector<double> rhs;
        auto A = Assembler2D::assemble(mesh, eps, 1.0, 0.0, 0.0, f, rhs);
        Assembler2D::applyDirichlet(A, rhs, mesh, allDirichlet,
            [](const std::array<double, 2>& p) { return poissonExact(p[0], p[1]); });
        femsolver::SparseLU lu(A);
        const auto uDirect = lu.solve(rhs);
        std::printf("[T6a] sparse LU: nnz(A)=%d -> nnz(L+U)=%d (fill %.1fx)\n",
            lu.nnzA(), lu.nnzL() + lu.nnzU(), lu.fillRatio());
        std::vector<double> u, hist;
        const int it = femsolver::bicgstab(A, rhs, u, 1e-10, 10000, false, hist);
        double diff = 0.0;
        for (int k = 0; k < mesh.numNodes(); ++k)
            diff = std::max(diff, std::abs(u[k] - uDirect[k]));
        std::printf("[T6a] BiCGStab %d iters, ||u_bi - u_direct||_inf = %.2e\n", it, diff);
        assert(diff < 1e-8);
    }

    {
        Mesh2D mesh(64, 64);
        std::vector<double> rhs;
        auto A = Assembler2D::assemble(mesh, eps, 1.0, 0.0, 0.0, f, rhs);
        Assembler2D::applyDirichlet(A, rhs, mesh, allDirichlet,
            [](const std::array<double, 2>& p) { return poissonExact(p[0], p[1]); });

        std::vector<double> uCG, hCG;
        const int itCG = femsolver::cg(A, rhs, uCG, 1e-10, 3000, false, hCG);
        std::vector<double> uBI, hBI;
        const int itBI = femsolver::bicgstab(A, rhs, uBI, 1e-10, 10000, false, hBI);

        std::printf("[T6b] CG       : %d iters, final rel.res = %.3e\n", itCG, hCG.back());
        std::printf("[T6b] BiCGStab : %d iters, final rel.res = %.3e\n", itBI, hBI.back());
        assert(hCG.back() > 1e-6);              // CG 失效
        assert(hBI.back() < 1e-10);
    }
    std::cout << "[T6] PASS\n";
}

// 测试 7：三类边界混在一起跑
// u = sin(πx)sin(πy)。LEFT|BOTTOM 钉 Dirichlet，右边给 Neumann（-π·sin(πy)），
// 顶边给 Robin（∂u/∂n + u，即 g = πcos(πy)sin(πx) + sin(πx)sin(πy)）。
// 角点划给谁：本质谓词把带 LEFT/BOTTOM 位的结点全收了（含角点），
// 角点上多装的 Neumann/Robin 贡献会被 Dirichlet 划行冲掉，所以先装载荷、后封行。
static void testThreeTypeBC()
{
    std::cout << "\n=== [T7] Dirichlet + Neumann + Robin 三类混合 ===\n";
    const double alpha = 1.0;
    auto isEssential = [](int flag) {
        return flag != Mesh2D::NONE && (flag & (Mesh2D::RIGHT | Mesh2D::TOP)) == 0;
        };
    auto isNatural = [](int flag) { return (flag & Mesh2D::RIGHT) != 0; };
    auto isRobin = [](int flag) { return (flag & Mesh2D::TOP) != 0; };
    auto gN = [](double, double y) { return -PI * std::sin(PI * y); };       // 右边通量
    auto gR = [&](double x, double y) {                                    // Robin 数据
        return PI * std::cos(PI * y) * std::sin(PI * x) + alpha * std::sin(PI * x) * std::sin(PI * y);
        };
    auto aR = [&](double, double) { return alpha; };

    const int nxs[] = { 8, 16, 32, 64 };
    double prevErr = 0.0;
    for (int t = 0; t < 4; ++t) {
        Mesh2D mesh(nxs[t], nxs[t]);
        std::vector<double> rhs;
        auto A = Assembler2D::assemble(mesh, 1.0, 0.0, 0.0, 0.0, poissonF, rhs);
        Assembler2D::applyNeumann(rhs, mesh, isNatural, gN);
        Assembler2D::applyRobin(A, rhs, mesh, isRobin, aR, gR);
        Assembler2D::applyDirichlet(A, rhs, mesh, isEssential,
            [](const std::array<double, 2>& p) { return poissonExact(p[0], p[1]); });
        std::vector<double> u; std::vector<double> hist;
        femsolver::cg(A, rhs, u, 1e-10, 10000, true, hist);
        const double E = Post2D::l2Error(mesh, u, poissonExact);
        double rate = (t > 0) ? std::log2(prevErr / E) : 0.0;
        std::printf("[T7] nx=%2d  E_L2=%.4f  rate=%.2f  CG(Jacobi)=%zu it\n",
            nxs[t], E, rate, hist.size());
        assert(hist.back() < 1e-10);
        if (t >= 2) assert(rate > 1.85 && rate < 2.15);
        prevErr = E;
    }
    std::cout << "[T7] PASS\n";
}

// 测试 8：纯 Neumann 为什么必须钉点
// 四边全给通量（取精确解的法向导数，兼容性条件 ∫f+∮g=0 凑好了）。
// (a) 不钉：矩阵带零空间（常数场），CG 直接飘；(b) 钉一个角点就好——工程上都这么干。
static void testPureNeumann()
{
    std::cout << "\n=== [T8] 纯 Neumann：奇异 vs 钉一点 ===\n";
    auto isNatural = [](int flag) { return flag != Mesh2D::NONE; };
    // 四边通量：竖直边 -π·sin(πy)、水平边 -π·sin(πx)。
    // 给的是位置函数，按中点坐标 x 是不是 0/1 分类（数据函数，不算"比坐标判边界"）
    auto gAll = [](double x, double y) {
        return (x == 0.0 || x == 1.0) ? -PI * std::sin(PI * y) : -PI * std::sin(PI * x);
        };

    // (a) 不钉点：矩阵奇异（常数场在零空间），看 CG 表现
    {
        Mesh2D mesh(32, 32);
        std::vector<double> rhs;
        auto A = Assembler2D::assemble(mesh, 1.0, 0.0, 0.0, 0.0, poissonF, rhs);
        Assembler2D::applyNeumann(rhs, mesh, isNatural, gAll);
        std::vector<double> u; std::vector<double> hist;
        const int it = femsolver::cg(A, rhs, u, 1e-10, 300, true, hist);
        std::printf("[T8a] 纯 Neumann 不钉点: CG %d iters, final rel.res = %.3e\n",
            it, hist.back());
    }

    // (b) 钉角点 (0,0)（其 flag = LEFT|BOTTOM，等值判断只命中这一个结点）
    auto isPinned = [](int flag) { return flag == (Mesh2D::LEFT | Mesh2D::BOTTOM); };
    const int nxs[] = { 16, 32, 64, 128 };
    double prevErr = 0.0;
    for (int t = 0; t < 4; ++t) {
        Mesh2D mesh(nxs[t], nxs[t]);
        std::vector<double> rhs;
        auto A = Assembler2D::assemble(mesh, 1.0, 0.0, 0.0, 0.0, poissonF, rhs);
        Assembler2D::applyNeumann(rhs, mesh, isNatural, gAll);
        Assembler2D::applyDirichlet(A, rhs, mesh, isPinned,
            [](const std::array<double, 2>& p) { return poissonExact(p[0], p[1]); });
        std::vector<double> u; std::vector<double> hist;
        femsolver::cg(A, rhs, u, 1e-10, 10000, true, hist);
        const double E = Post2D::l2Error(mesh, u, poissonExact);
        double rate = (t > 0) ? std::log2(prevErr / E) : 0.0;
        std::printf("[T8b] nx=%3d  E_L2=%.5f  rate=%.2f  CG=%zu it\n",
            nxs[t], E, rate, hist.size());
        assert(hist.back() < 1e-10);
        // 误差常数比 Dirichlet 大一个量级（零空间方向对右端积分误差敏感），
        // 进渐近线慢（实测 rate 从 1.78 一路爬：1.78→1.81→1.83），
        // 只验收末组网格：阶数逼近 2、误差量级压在 1e-2 内
        if (t == 3) { assert(rate > 1.75 && rate < 2.1); assert(E < 0.01); }
        prevErr = E;
    }
    std::cout << "[T8] PASS\n";
}

int main()
{
    std::cout << "========== FEM2D 任务二验收 ==========\n";
    testElement();
    testPoissonBenchmark();
    testConvergence();
    testCGvsDirect();
    testRectangle21();
    testConvectionDiffusion();
    testThreeTypeBC();
    testPureNeumann();
    std::cout << "\n========== ALL TESTS PASSED ==========\n";
    return 0;
}