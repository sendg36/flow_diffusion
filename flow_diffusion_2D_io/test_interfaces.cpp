// test_interfaces.cpp —— 阶段三接口验收：T9–T13
//  T9  接口 1：Gmsh 非结构网格 vs 自写结构化网格，同题误差同量级（内核不依赖网格规则性）
//  T10 接口 2：混合边界（三边 Dirichlet + 右边 Neumann），收敛阶 rate≈2
//  T11 接口 2：INI 文件驱动与硬编码跑同一题，结果一致（"不重编译改工况"验证）
//  T12 接口 3：求解器插槽（CGSolver / BiCGStabSolver）与直接法对拍
//  T13 接口 4：VTK 输出文件结构自检（ParaView 打开前的静态验证）

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "Mesh.h"
#include "Element.h"
#include "Assembler.h"
#include "Solver.h"
#include "Post.h"
#include "gmsh_reader.h"
#include "vtk_writer.h"
#include "LinearSolver.h"
#include "BoundaryConditions.h"

static const double PI = std::acos(-1.0);

// 全域 Dirichlet（Gmsh 网格：线单元打过 tag 的结点 flag != 0）
static bool allDirichlet(int flag) { return flag != Mesh2D::NONE; }

static double poissonExact(double x, double y) { return std::sin(PI * x) * std::sin(PI * y); }
static double poissonF(double x, double y) { return 2 * PI * PI * std::sin(PI * x) * std::sin(PI * y); }
static double cdF(double eps, double x, double y)
{
    return 2 * eps * PI * PI * std::sin(PI * x) * std::sin(PI * y)
        + PI * std::cos(PI * x) * std::sin(PI * y);
}

// ---------- T9：Gmsh 非结构网格对拍 ----------
static void testGmshImport()
{
    std::cout << "\n=== [T9] 接口1：Gmsh 非结构网格 vs 结构化网格 ===\n";
    Mesh2D mesh = GmshReader::read("data/square_unstruct.msh");
    std::vector<double> rhs;
    auto A = Assembler2D::assemble(mesh, 1.0, 0.0, 0.0, 0.0, poissonF, rhs);
    Assembler2D::applyDirichlet(A, rhs, mesh, allDirichlet,
        [](const std::array<double, 2>& p) { return poissonExact(p[0], p[1]); });
    CGSolver solver(1e-10, 50000, true);
    const auto u = solver.solve(A, rhs);
    const double E = Post2D::l2Error(mesh, u, poissonExact);

    // 同分辨率结构化网格 nx=8 的误差是 0.0226，非结构网格应同量级
    std::printf("[T9] Gmsh mesh  E_L2 = %.4f (structured nx=8: 0.0226)\n", E);
    assert(E > 0.005 && E < 0.05);
    assert(solver.history().back() < 1e-10);
    std::cout << "[T9] PASS\n";
}

// ---------- T10：混合边界收敛 ----------
static void testMixedBC()
{
    std::cout << "\n=== [T10] 接口2：混合边界（Dirichlet x3 + Neumann 右边） ===\n";
    // 精确解 u = sin(πx)sin(πy)。右边 x=1 上 eps·du/dn = du/dx = -π·sin(πy)
    ProblemSpec spec;
    spec.bc[Mesh2D::RIGHT] = { BCSpec::Neumann, "-pi*sin(pi*y)" };
    auto isEssential = [](int flag) { return flag != Mesh2D::NONE && flag != Mesh2D::RIGHT; };

    const int nxs[] = { 8, 16, 32, 64 };
    double prevErr = 0.0;
    for (int t = 0; t < 4; ++t) {
        Mesh2D mesh(nxs[t], nxs[t]);
        std::vector<double> rhs;
        auto A = Assembler2D::assemble(mesh, 1.0, 0.0, 0.0, 0.0, poissonF, rhs);
        applyNeumann(rhs, mesh, spec);                       // 自然边界：右端加边界通量
        Assembler2D::applyDirichlet(A, rhs, mesh, isEssential,
            [](const std::array<double, 2>& p) { return poissonExact(p[0], p[1]); });
        CGSolver solver;
        const auto u = solver.solve(A, rhs);
        const double E = Post2D::l2Error(mesh, u, poissonExact);
        double rate = (t > 0) ? std::log2(prevErr / E) : 0.0;
        std::printf("[T10] nx=%2d  E_L2=%.4f  rate=%.2f  CG=%zu it\n",
            nxs[t], E, rate, solver.history().size());
        assert(solver.history().back() < 1e-10);
        if (t >= 2) assert(rate > 1.85 && rate < 2.15);
        prevErr = E;
    }
    std::cout << "[T10] PASS\n";
}

// ---------- T11：INI 驱动与硬编码对拍 ----------
static void testIniDriven()
{
    std::cout << "\n=== [T11] 接口2：INI 文件驱动 ===\n";
    const char* iniPath = "data/mixed_bc.ini";
    {
        std::ofstream out(iniPath);
        out << "# 混合边界 Poisson，与 T10 同题\n"
            << "[domain]\n"
            << "eps = 1.0\n"
            << "bx  = 0.0\n"
            << "by  = 0.0\n"
            << "c   = 0.0\n"
            << "f   = 2*pi*pi*sin(pi*x)*sin(pi*y)\n"
            << "[bc.1]   ; LEFT\n"
            << "type  = dirichlet\n"
            << "value = sin(pi*x)*sin(pi*y)\n"
            << "[bc.4]   ; BOTTOM\n"
            << "type  = dirichlet\n"
            << "value = sin(pi*x)*sin(pi*y)\n"
            << "[bc.8]   ; TOP\n"
            << "type  = dirichlet\n"
            << "value = sin(pi*x)*sin(pi*y)\n"
            << "[bc.2]   ; RIGHT\n"
            << "type  = neumann\n"
            << "value = -pi*sin(pi*y)\n";
    }

    const ProblemSpec spec = IniParser::read(iniPath);

    // INI 驱动的装配：系数、右端、边界全部来自文件
    Mesh2D mesh(32, 32);
    std::vector<double> rhs;
    auto A = Assembler2D::assemble(mesh,
        [&](double x, double y) { return expr::eval(spec.epsExpr, x, y); },
        [&](double x, double y) { return expr::eval(spec.bxExpr, x, y); },
        [&](double x, double y) { return expr::eval(spec.byExpr, x, y); },
        [&](double x, double y) { return expr::eval(spec.cExpr, x, y); },
        [&](double x, double y) { return expr::eval(spec.fExpr, x, y); },
        rhs);
    applyNeumann(rhs, mesh, spec);

    // 本质边界谓词：flag 命中任一登记为 dirichlet 的 tag（位测试，角点多边或集也成立）
    std::map<int, BCSpec> bcMap = spec.bc;
    auto isEssential = [&](int flag) {
        for (const auto& kv : bcMap)
            if (kv.second.type == BCSpec::Dirichlet && (flag & kv.first) != 0) return true;
        return false;
        };
    // 取第一个 dirichlet 条目的表达式作边界值（本题三条边同一表达式）
    std::string dirExpr;
    for (const auto& kv : bcMap)
        if (kv.second.type == BCSpec::Dirichlet) { dirExpr = kv.second.valueExpr; break; }
    auto g = [&](const std::array<double, 2>& p) {
        return expr::eval(dirExpr, p[0], p[1]);
        };
    Assembler2D::applyDirichlet(A, rhs, mesh, isEssential, g);

    CGSolver solver;
    const auto uIni = solver.solve(A, rhs);

    // 硬编码同题对拍
    std::vector<double> rhs2;
    auto A2 = Assembler2D::assemble(mesh, 1.0, 0.0, 0.0, 0.0, poissonF, rhs2);
    ProblemSpec hard;
    hard.bc[Mesh2D::RIGHT] = { BCSpec::Neumann, "-pi*sin(pi*y)" };
    applyNeumann(rhs2, mesh, hard);
    auto isEss2 = [](int flag) { return flag != Mesh2D::NONE && flag != Mesh2D::RIGHT; };
    Assembler2D::applyDirichlet(A2, rhs2, mesh, isEss2,
        [](const std::array<double, 2>& p) { return poissonExact(p[0], p[1]); });
    const auto uHard = solver.solve(A2, rhs2);

    double diff = 0.0;
    for (int k = 0; k < mesh.numNodes(); ++k)
        diff = std::max(diff, std::abs(uIni[k] - uHard[k]));
    std::printf("[T11] ||u_ini - u_hard||_inf = %.2e\n", diff);
    assert(diff < 1e-12);
    std::cout << "[T11] PASS\n";
}

// ---------- T12：求解器插槽对拍 ----------
static void testSolverSlot()
{
    std::cout << "\n=== [T12] 接口3：求解器插槽对拍 ===\n";
    {   // 对称正定：CGSolver
        Mesh2D mesh(32, 32);
        std::vector<double> rhs;
        auto A = Assembler2D::assemble(mesh, 1.0, 0.0, 0.0, 0.0, poissonF, rhs);
        Assembler2D::applyDirichlet(A, rhs, mesh, allDirichlet,
            [](const std::array<double, 2>& p) { return poissonExact(p[0], p[1]); });
        femsolver::SparseLU lu(A);
        const auto uDirect = lu.solve(rhs);
        std::printf("[T12a] sparse LU: nnz(A)=%d -> nnz(L+U)=%d (fill %.1fx)\n",
            lu.nnzA(), lu.nnzL() + lu.nnzU(), lu.fillRatio());
        LinearSolver* solver = new CGSolver();
        const auto u = solver->solve(A, rhs);
        double diff = 0.0;
        for (int k = 0; k < mesh.numNodes(); ++k)
            diff = std::max(diff, std::abs(u[k] - uDirect[k]));
        std::printf("[T12a] CGSolver vs dense: %.2e\n", diff);
        assert(diff < 1e-8);
        delete solver;
    }
    {   // 非对称：BiCGStabSolver
        const double eps = 0.01;
        Mesh2D mesh(16, 16);
        std::vector<double> rhs;
        auto f = [&](double x, double y) { return cdF(eps, x, y); };
        auto A = Assembler2D::assemble(mesh, eps, 1.0, 0.0, 0.0, f, rhs);
        Assembler2D::applyDirichlet(A, rhs, mesh, allDirichlet,
            [](const std::array<double, 2>& p) { return poissonExact(p[0], p[1]); });
        femsolver::SparseLU lu(A);
        const auto uDirect = lu.solve(rhs);
        std::printf("[T12b] sparse LU: nnz(A)=%d -> nnz(L+U)=%d (fill %.1fx)\n",
            lu.nnzA(), lu.nnzL() + lu.nnzU(), lu.fillRatio());
        LinearSolver* solver = new BiCGStabSolver();
        const auto u = solver->solve(A, rhs);
        double diff = 0.0;
        for (int k = 0; k < mesh.numNodes(); ++k)
            diff = std::max(diff, std::abs(u[k] - uDirect[k]));
        std::printf("[T12b] BiCGStabSolver vs dense: %.2e\n", diff);
        assert(diff < 1e-8);
        delete solver;
    }
    std::cout << "[T12] PASS\n";
}

// ---------- T13：VTK 输出自检 ----------
static void testVtkOutput()
{
    std::cout << "\n=== [T13] 接口4：VTK 输出 ===\n";
    Mesh2D mesh(8, 8);
    std::vector<double> rhs;
    auto A = Assembler2D::assemble(mesh, 1.0, 0.0, 0.0, 0.0, poissonF, rhs);
    Assembler2D::applyDirichlet(A, rhs, mesh, allDirichlet,
        [](const std::array<double, 2>& p) { return poissonExact(p[0], p[1]); });
    CGSolver solver;
    const auto u = solver.solve(A, rhs);

    const std::string path = "data/out_nx8.vtk";
    VtkWriter::write(path, mesh, u, poissonExact);

    std::ifstream in(path);
    assert(in.good());
    std::stringstream buf;
    buf << in.rdbuf();
    const std::string content = buf.str();

    auto has = [&](const std::string& s) {
        assert(content.find(s) != std::string::npos);
        };
    has("DATASET UNSTRUCTURED_GRID");
    has("POINTS 81 float");            // (8+1)^2
    has("CELLS 128 512");              // 2*8*8 个三角形
    has("CELL_TYPES 128");
    has("POINT_DATA 81");
    has("SCALARS u float");
    has("SCALARS abs_err float");
    std::printf("[T13] %s written, structure ok (%zu bytes)\n",
        path.c_str(), content.size());
    std::cout << "[T13] PASS\n";
}

int main()
{
    std::cout << "========== 阶段三接口验收 ==========\n";
    testGmshImport();
    testMixedBC();
    testIniDriven();
    testSolverSlot();
    testVtkOutput();
    std::cout << "\n========== ALL INTERFACE TESTS PASSED ==========\n";
    return 0;
}