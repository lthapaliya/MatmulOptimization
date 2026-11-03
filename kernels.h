using Fn = void(*)(const double* A, const double* B, double* C, int n, int m, int p);

void mm_naive  (const double*, const double*, double*, int, int, int);  // i,j,k
void mm_reorder(const double*, const double*, double*, int, int, int);  // i,k,j
// later: mm_blocked, mm_neon, mm_threads

