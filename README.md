# MatmulOptimization

Optimizing dense matrix multiplication (double precision, row-major) in C++ on Apple Silicon, one stage at a time. Each stage is validated against NumPy and benchmarked against OpenBLAS.

Codename: **redpill**.

## Stages

| Stage | Status |
|---|---|
| Naive triple loop (i, j, k) | done |
| Loop reordering (i, k, j) | done |
| Cache blocking | planned |
| NEON SIMD | planned |
| Multithreading | planned |

Goal: reach 60% of OpenBLAS performance.

## Results

Apple M3, single thread, N = 1024:

| Stage | GFLOPS |
|---|---|
| naive | ~2 |
| reorder (i, k, j) | ~15 |

Reordering the loops makes the inner loop read both matrices sequentially, which gives about a 7x speedup over the naive version.

Percent-of-OpenBLAS numbers will be added once the OpenBLAS baseline is pinned to a matching thread count.

## Layout

- `kernels.h`, `kernels.cpp`: the multiplication kernels, one function per stage
- `redpill.cpp`: command-line program that reads matrices from stdin and runs a chosen stage
- `bench.cpp`: benchmark harness (GFLOPS, comparison with OpenBLAS)
- `test_redpill.py`: pytest suite that checks every stage against NumPy

## Build and test

~~~
clang++ -std=c++17 -O3 -mcpu=native redpill.cpp kernels.cpp -o redpill
pip install numpy pytest
pytest -v
~~~

## Benchmark

Requires OpenBLAS (`brew install openblas`).

~~~
clang++ -std=c++17 -O3 -mcpu=native bench.cpp kernels.cpp -o bench \
  -I/opt/homebrew/opt/openblas/include -L/opt/homebrew/opt/openblas/lib \
  -Wl,-rpath,/opt/homebrew/opt/openblas/lib -lopenblas
./bench
~~~
