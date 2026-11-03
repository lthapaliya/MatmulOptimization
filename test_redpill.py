import subprocess
import numpy as np
import pytest

BINARY = "./redpill"
STAGES = ["naive", "reorder"]


def run_redpill(A, B, stage):
    n, m = A.shape
    p = B.shape[1]
    text = f"{n} {m} {p}\n"
    for M in (A, B):
        for row in M:
            text += " ".join(repr(float(x)) for x in row) + "\n"
    result = subprocess.run(
        [BINARY, stage], input=text, capture_output=True, text=True, check=True
    )
    return np.array(result.stdout.split(), dtype=float).reshape(n, p)


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("n,m,p", [
    (1, 1, 1), (2, 3, 2), (5, 5, 5),
    (10, 1, 7), (1, 10, 1), (50, 40, 30),
])
def test_random_matrices(stage, n, m, p):
    rng = np.random.default_rng(seed=n * 100 + m * 10 + p)
    A = rng.uniform(-10, 10, size=(n, m))
    B = rng.uniform(-10, 10, size=(m, p))
    np.testing.assert_allclose(run_redpill(A, B, stage), A @ B, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("stage", STAGES)
def test_identity(stage):
    A = np.random.default_rng(0).uniform(-5, 5, size=(4, 4))
    np.testing.assert_allclose(run_redpill(A, np.eye(4), stage), A, rtol=1e-12)


@pytest.mark.parametrize("stage", STAGES)
def test_zeros(stage):
    A = np.random.default_rng(1).uniform(-5, 5, size=(3, 4))
    np.testing.assert_array_equal(run_redpill(A, np.zeros((4, 2)), stage), np.zeros((3, 2)))
