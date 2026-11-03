#include "kernels.h"
#include <algorithm>

// Layout: row-major flat arrays. A is n x m, B is m x p, C is n x p.
// Every kernel overwrites C.

void mm_naive(const double* A, const double* B, double* C, int n, int m, int p) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j) {
            double s = 0.0;
            for (int k = 0; k < m; ++k)
                s += A[i * m + k] * B[k * p + j];
            C[i * p + j] = s;
        }
}

void mm_reorder(const double* A, const double* B, double* C, int n, int m, int p) {
    std::fill(C, C + (size_t)n * p, 0.0);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < m; ++k) {
            double a = A[i * m + k];
            for (int j = 0; j < p; ++j)
                C[i * p + j] += a * B[k * p + j];
        }
}
