#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>

using Matrix = std::vector<std::vector<double>>;

Matrix multiply(const Matrix& A, const Matrix& B) {
    size_t n = A.size();
    size_t m = A[0].size();
    size_t p = B[0].size();

    if (B.size() != m)
        throw std::invalid_argument("Dimension mismatch: A cols != B rows");

    Matrix C(n, std::vector<double>(p, 0.0));

    for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < m; ++k)
            for (size_t j = 0; j < p; ++j)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

int main() {
    size_t n, m, p;
    std::cin >> n >> m >> p;

    Matrix A(n, std::vector<double>(m));
    Matrix B(m, std::vector<double>(p));
    for (auto& row : A) for (double& x : row) std::cin >> x;
    for (auto& row : B) for (double& x : row) std::cin >> x;

    std::cout << std::setprecision(17);
    for (const auto& row : multiply(A, B)) {
        for (double x : row) std::cout << x << " ";
        std::cout << "\n";
    }
}
