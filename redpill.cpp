#include <cstdio>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>
#include "kernels.h"

int main(int argc, char** argv) {
    const char* stage = argc > 1 ? argv[1] : "naive";
    Fn f = nullptr;
    if      (!std::strcmp(stage, "naive"))   f = mm_naive;
    else if (!std::strcmp(stage, "reorder")) f = mm_reorder;
    else { std::fprintf(stderr, "unknown stage: %s\n", stage); return 1; }

    int n, m, p;
    std::cin >> n >> m >> p;
    std::vector<double> A((size_t)n * m), B((size_t)m * p), C((size_t)n * p);
    for (double& x : A) std::cin >> x;
    for (double& x : B) std::cin >> x;

    f(A.data(), B.data(), C.data(), n, m, p);

    std::cout << std::setprecision(17);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) std::cout << C[(size_t)i * p + j] << " ";
        std::cout << "\n";
    }
}
