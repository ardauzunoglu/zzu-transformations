"""
Build a 2D SSE-landscape figure comparing the trajectories of gradient
descent, Gauss-Newton (with self-activating LM damping), and BFGS on the
same problem.

Problem: fit y = a · exp(b · x) on the exponential_multiplicative dataset.
Two parameters means we can plot the SSE surface as a contour and overlay
each optimizer's trajectory, which makes the qualitative differences
between the three methods immediately legible.

Output:  comparison_results/optimizer_trajectories.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import toy_data as td
from reproducibility import reproduce_dir

OUT_PATH = reproduce_dir("comparison_results", PROJECT_ROOT) / "optimizer_trajectories.png"
NAVY = "#1E2761"

# ---------------------------------------------------------------------------
# Set up the problem
# ---------------------------------------------------------------------------

bundle = td.make_exponential_multiplicative()
x = bundle.X.iloc[:, 0].to_numpy()
y = bundle.y.to_numpy()
TRUE_AB = (2.0, 0.7)


def f(theta):
    a, b = theta
    return a * np.exp(b * x)


def sse(theta):
    return float(np.sum((y - f(theta)) ** 2))


def jacobian(theta):
    a, b = theta
    e = np.exp(b * x)
    return np.column_stack([e, a * x * e])


def grad(theta):
    return -2.0 * (jacobian(theta).T @ (y - f(theta)))


# ---------------------------------------------------------------------------
# Mini-implementations that record trajectories
# ---------------------------------------------------------------------------

def gd_path(theta0, lr=2e-7, max_iter=400, tol=1e-8):
    """Plain gradient descent.  Tiny lr because gradients are huge here."""
    path = [np.array(theta0, dtype=float)]
    theta = path[0].copy()
    prev_loss = sse(theta)
    for _ in range(max_iter):
        theta = theta - lr * grad(theta)
        path.append(theta.copy())
        loss = sse(theta)
        if abs(prev_loss - loss) / max(abs(prev_loss), 1e-12) < tol:
            break
        prev_loss = loss
    return np.array(path)


def gn_path(theta0, max_iter=20, tol=1e-8, factor=10.0):
    """Gauss-Newton with self-activating Levenberg-Marquardt damping."""
    path = [np.array(theta0, dtype=float)]
    theta = path[0].copy()
    lam = 0.0
    for _ in range(max_iter):
        r = y - f(theta)
        J = jacobian(theta)
        A = J.T @ J + lam * np.eye(2)
        try:
            delta = np.linalg.solve(A, J.T @ r)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(A, J.T @ r, rcond=None)
        theta_new = theta + delta
        if sse(theta_new) < sse(theta):
            theta = theta_new
            lam = max(lam / factor, 0.0)
            path.append(theta.copy())
            if np.linalg.norm(delta) / (np.linalg.norm(theta) + 1e-12) < tol:
                break
        else:
            lam = max(lam * factor, 1e-4)
            if lam > 1e8:
                break
    return np.array(path)


def bfgs_path(theta0, max_iter=40, tol=1e-8, c1=1e-4):
    path = [np.array(theta0, dtype=float)]
    theta = path[0].copy()
    H = np.eye(2)
    gradient = grad(theta)
    for _ in range(max_iter):
        if np.linalg.norm(gradient) < tol:
            break
        direction = -H @ gradient
        gd_dot = float(gradient @ direction)
        # Backtracking Armijo line search.
        alpha = 1.0
        f0 = sse(theta)
        for _ in range(60):
            if sse(theta + alpha * direction) <= f0 + c1 * alpha * gd_dot:
                break
            alpha *= 0.5
        else:
            break
        s = alpha * direction
        theta_new = theta + s
        gradient_new = grad(theta_new)
        ydiff = gradient_new - gradient
        sy = float(s @ ydiff)
        if sy > 1e-12:
            rho = 1.0 / sy
            I = np.eye(2)
            H = ((I - rho * np.outer(s, ydiff)) @ H @ (I - rho * np.outer(ydiff, s))
                 + rho * np.outer(s, s))
            H = 0.5 * (H + H.T)
        theta = theta_new
        gradient = gradient_new
        path.append(theta.copy())
    return np.array(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    theta0 = np.array([1.0, 0.1])

    gd = gd_path(theta0)
    gn = gn_path(theta0)
    bf = bfgs_path(theta0)

    # Contour grid for the SSE surface (log scale; the basin is narrow in b).
    A = np.linspace(0.5, 3.5, 120)
    B = np.linspace(0.0, 1.05, 120)
    AA, BB = np.meshgrid(A, B)
    Z = np.zeros_like(AA)
    for i in range(AA.shape[0]):
        for j in range(AA.shape[1]):
            Z[i, j] = sse([AA[i, j], BB[i, j]])

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    cs = ax.contourf(AA, BB, np.log10(Z), levels=24, cmap="Greys", alpha=0.6)
    ax.contour(AA, BB, np.log10(Z), levels=12, colors="white", linewidths=0.6)
    fig.colorbar(cs, ax=ax, label=r"$\log_{10}$ SSE", shrink=0.85)

    ax.plot(gd[:, 0], gd[:, 1], "-o", color="#777777", lw=1.4, ms=3,
            label=f"Gradient Descent ({len(gd)-1} steps)")
    ax.plot(gn[:, 0], gn[:, 1], "-s", color="#1F77B4", lw=2.0, ms=6,
            label=f"Gauss-Newton + LM ({len(gn)-1} steps)")
    ax.plot(bf[:, 0], bf[:, 1], "-^", color="#2CA02C", lw=2.0, ms=6,
            label=f"BFGS ({len(bf)-1} steps)")

    ax.scatter([theta0[0]], [theta0[1]], s=140, marker="X",
               color="black", zorder=10, label="start  θ₀ = (1, 0.1)")
    ax.scatter([TRUE_AB[0]], [TRUE_AB[1]], s=240, marker="*",
               color="#E05252", zorder=10,
               label=f"true optimum  (a={TRUE_AB[0]}, b={TRUE_AB[1]})")

    ax.set_xlabel("a", fontsize=12, color=NAVY)
    ax.set_ylabel("b", fontsize=12, color=NAVY)
    ax.set_title(
        r"Optimizer trajectories on the SSE surface for $y = a \cdot e^{b x}$"
        "\n(exponential_multiplicative; same start θ₀ for all three)",
        fontsize=13, color=NAVY, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.2)

    fig.savefig(OUT_PATH, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")
    print(f"  GD:   {len(gd)-1} steps,  final θ = {gd[-1]}")
    print(f"  GN:   {len(gn)-1} steps,  final θ = {gn[-1]}")
    print(f"  BFGS: {len(bf)-1} steps,  final θ = {bf[-1]}")


if __name__ == "__main__":
    main()
