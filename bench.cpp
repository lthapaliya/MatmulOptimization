// bench.cpp
#include <chrono>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cblas.h>
#include "kernels.h"

static void ref_dgemm(const double* A, const double* B, double* C, int n, int m, int p) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, p, m, 1.0, A, m, B, p, 0.0, C, p);
}

static double best_time(Fn f, const double* A, const double* B, double* C,
                        int n, int m, int p, int reps) {
    std::fill(C, C + (size_t)n * p, 0.0);
    f(A, B, C, n, m, p);                       // warm-up
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        std::fill(C, C + (size_t)n * p, 0.0);
        auto t0 = std::chrono::steady_clock::now();
        f(A, B, C, n, m, p);
        auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

int main() {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1, 1);

    for (int N : {256, 512, 1000, 1024, 1025, 2048}) {
        std::vector<double> A(N*N), B(N*N), C(N*N), R(N*N);
        for (auto& x : A) x = dist(rng);
        for (auto& x : B) x = dist(rng);

        double flops = 2.0 * N * N * N;
        double t_ref = best_time(ref_dgemm, A.data(), B.data(), R.data(), N, N, N, 5);

        struct { const char* name; Fn f; } stages[] = {
            {"naive", mm_naive}, {"reorder", mm_reorder},
        };
        for (auto& s : stages) {
            double t = best_time(s.f, A.data(), B.data(), C.data(), N, N, N, 5);
            double err = 0;
            for (size_t i = 0; i < C.size(); ++i) err = std::max(err, std::fabs(C[i] - R[i]));
            std::printf("N=%4d %-8s %7.2f GFLOPS  %5.1f%% of OpenBLAS  maxerr=%.1e\n",
                        N, s.name, flops / t / 1e9, 100.0 * t_ref / t, err);
        }
    }
}
