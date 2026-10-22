# Transient heat conduction in a slab

A finite-difference solver for the 1D heat equation, validated against the exact Fourier-series solution.

**Live demo:** https://Just-in-ish.github.io/heat-conduction-solver/

## Problem

A slab of thickness L = 1 starts at uniform temperature u0 = 100. At t = 0 both faces are held at 0.

    u_t = alpha * u_xx,   u(0,t) = u(L,t) = 0,   u(x,0) = u0

Exact solution: u(x,t) = sum over odd n of (4 u0 / (n pi)) sin(n pi x / L) exp(-alpha n^2 pi^2 t / L^2).

## Methods

One theta-method covers three schemes: theta = 0 explicit (FTCS), theta = 1 implicit (BTCS), theta = 0.5 Crank-Nicolson. Each step solves a tridiagonal system with the Thomas algorithm, O(N) per step. FTCS is stable only for r = alpha dt / dx^2 <= 0.5.

## Contents

| File | Purpose |
|---|---|
| `index.html` | Interactive demo (JavaScript, no dependencies) |
| `solver.py` | Python implementation (NumPy) |
| `test_solver.py` | Tests: convergence order, FTCS instability, boundary conditions |

## Run

    pip install numpy pytest
    python solver.py      # convergence table
    pytest                # tests

