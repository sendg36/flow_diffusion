// omp_matvec.h —— OpenMP 版 CSR SpMV（与 CSRMatrix::matvec 结果对拍）
// 内核的 matvec 不动，并行版作为独立函数提供，两版逐分量一致才算对。

#pragma once

#include <vector>
#include "Assembler.h"

#ifdef _OPENMP
#include <omp.h>
#endif

inline void matvecOmp(const CSRMatrix& A, const std::vector<double>& x,
                      std::vector<double>& y)
{
    assert((int)x.size() == A.n);
    y.assign(A.n, 0.0);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < A.n; ++i) {
        double s = 0.0;
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k)
            s += A.values[k] * x[A.col[k]];
        y[i] = s;
    }
}

#ifdef _OPENMP
inline void setOmpThreads(int t) { omp_set_num_threads(t); }
inline int  maxOmpThreads()      { return omp_get_max_threads(); }
#else
inline void setOmpThreads(int) {}
inline int  maxOmpThreads()      { return 1; }
#endif
