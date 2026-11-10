// benchmark_omp.cpp —— OpenMP SpMV 加速比（1 / 2 / 4 线程）
// 先用百万自由度级网格（本机可把 nx 调大到 316）。
// 用法：g++ -O2 -fopenmp benchmark_omp.cpp -o bench && OMP_NUM_THREADS=4 ./bench

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "Mesh.h"
#include "Assembler.h"
#include "Solver.h"
#include "omp_matvec.h"

int main(int argc, char** argv)
{
    const int nx = (argc > 1) ? std::atoi(argv[1]) : 200;   // 200 → 4 万结点，316 → 10 万
    Mesh2D mesh(nx, nx);
    std::vector<double> rhs;
    const double PI = std::acos(-1.0);
    auto f = [&](double x, double y){ return 2*PI*PI*std::sin(PI*x)*std::sin(PI*y); };
    auto A = Assembler2D::assemble(mesh, 1.0, 0.0, 0.0, 0.0, f, rhs);
    std::printf("matrix: n=%d, nnz=%d\n", A.n, A.nnz());

    std::vector<double> x(A.n, 1.0), y1, y2;
    A.matvec(x, y1);
    matvecOmp(A, x, y2);
    double diff = 0.0;
    for (int i = 0; i < A.n; ++i) diff = std::max(diff, std::abs(y1[i] - y2[i]));
    std::printf("matvec consistency check: %.2e\n", diff);
    if (diff > 1e-12) { std::printf("FAIL\n"); return 1; }

    const int reps = 200;
    auto bench = [&](int threads){
        setOmpThreads(threads);
        matvecOmp(A, x, y2);                                  // 预热
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) matvecOmp(A, x, y2);
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    };

    const int t1 = 1, t2 = std::min(2, maxOmpThreads()), t4 = std::min(4, maxOmpThreads());
    const double ms1 = bench(t1), ms2 = bench(t2), ms4 = bench(t4);
    std::printf("threads  time/ms    speedup\n");
    std::printf("%7d  %8.3f  %8.2f\n", t1, ms1, 1.0);
    if (t2 != t1) std::printf("%7d  %8.3f  %8.2f\n", t2, ms2, ms1/ms2);
    if (t4 != t1 && t4 != t2) std::printf("%7d  %8.3f  %8.2f\n", t4, ms4, ms1/ms4);
    return 0;
}
