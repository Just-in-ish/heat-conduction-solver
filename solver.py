"""1D transient heat conduction in a slab: theta-method finite differences
validated against the exact Fourier-series solution.

Problem: u_t = alpha * u_xx on 0 < x < L, u(0,t) = u(L,t) = 0, u(x,0) = u0.
theta = 0 explicit (FTCS), 1 implicit (BTCS), 0.5 Crank-Nicolson.
"""
import numpy as np


def exact(x, t, alpha=1.0, L=1.0, u0=100.0, terms=399):
    """Fourier-series solution (odd modes only)."""
    x = np.asarray(x, dtype=float)
    s = np.zeros_like(x)
    for n in range(1, terms + 1, 2):
        k = n * np.pi / L
        s += 4 * u0 / (n * np.pi) * np.sin(k * x) * np.exp(-alpha * k * k * t)
    return s


def thomas(a, b, c, d):
    """Solve a tridiagonal system (sub a, diag b, super c, rhs d)."""
    n = len(d)
    cp, dp = np.zeros(n), np.zeros(n)
    cp[0], dp[0] = c[0] / b[0], d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def solve(N, theta, r_target, t_end, alpha=1.0, L=1.0, u0=100.0):
    """Advance to t_end. Returns (x, u, dt, r, steps)."""
    dx = L / (N - 1)
    steps = max(1, int(np.ceil(t_end / (r_target * dx**2 / alpha))))
    dt = t_end / steps
    r = alpha * dt / dx**2
    x = np.linspace(0.0, L, N)
    u = np.full(N, u0)
    u[0] = u[-1] = 0.0
    M = N - 2
    a = np.full(M, -theta * r)
    b = np.full(M, 1 + 2 * theta * r)
    c = np.full(M, -theta * r)
    for _ in range(steps):
        rhs = u[1:-1] + (1 - theta) * r * (u[2:] - 2 * u[1:-1] + u[:-2])
        u[1:-1] = thomas(a, b, c, rhs)
    return x, u, dt, r, steps


def l2_error(x, u, t, **kw):
    return float(np.sqrt(np.mean((u - exact(x, t, **kw)) ** 2)))


if __name__ == "__main__":
    t_end, r, theta = 0.05, 0.4, 0.5
    print("Crank-Nicolson, r = 0.4, t = 0.05")
    print(f"{'N':>5} {'L2 error':>12} {'order':>7}")
    prev = None
    for N in (11, 21, 41, 81, 161):
        x, u, *_ = solve(N, theta, r, t_end)
        e = l2_error(x, u, t_end)
        order = f"{np.log2(prev / e):7.2f}" if prev else "      -"
        print(f"{N:5d} {e:12.3e} {order}")
        prev = e
