import numpy as np
from solver import solve, exact, l2_error


def test_crank_nicolson_converges_second_order():
    errs = []
    for N in (21, 41, 81):
        x, u, *_ = solve(N, 0.5, 0.4, 0.05)
        errs.append(l2_error(x, u, 0.05))
    assert np.log2(errs[0] / errs[1]) > 1.5
    assert np.log2(errs[1] / errs[2]) > 1.5


def test_ftcs_unstable_above_half():
    x, u, *_ = solve(41, 0.0, 0.6, 0.05)
    assert l2_error(x, u, 0.05) > 1.0


def test_boundaries_stay_zero():
    x, u, *_ = solve(41, 1.0, 1.0, 0.05)
    assert u[0] == 0 and u[-1] == 0
