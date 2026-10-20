"""Measurement layer for the psych-gym (PWM §V, §VII.3).

Estimators work on trajectories — simulated here, structured so the same
functions accept EMA-shaped human data later (MAGP 611 S2/S3 design path).

- estimate_loop_gain: ridge-regularized lag-1 MAR(1) on (affect, threat,
  u_model) -> rho(G_hat) plus regime banding with a complex-eigenvalue
  (oscillation) flag.
- hypothesis_entropy: Shannon entropy over the maintained hypothesis set.
- epistemic_velocity: d(log H)/d(log E) with zero-division sentinel.
"""

from __future__ import annotations

import numpy as np

RIDGE_LAMBDA = 1e-3
# PWM §V band edges (documented; tests plant dynamics around them)
DECAY_UPPER = 0.8
AMPLIFY_LOWER = 1.2

# Sentinel returned by epistemic_velocity when its denominator is ~0.
EVIDENCE_FLAT_SENTINEL = float("nan")


def _ridge_mar1(traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit x_t ≈ c + G x_{t-1} on a (T, k) trajectory: centered ridge fit."""
    X = traj[:-1, :]                      # predictors (T-1, k)
    Y = traj[1:, :]                       # targets    (T-1, k)
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - mx, Y - my
    k = X.shape[1]
    G_hat = np.linalg.solve(Xc.T @ Xc + RIDGE_LAMBDA * np.eye(k), Xc.T @ Yc)
    c_hat = my - mx @ G_hat
    return G_hat, c_hat


def lead_pair_is_complex(eigenvalues: np.ndarray, tol: float = 1e-8) -> bool:
    """True when the two largest-magnitude eigenvalues form a complex pair."""
    ev = np.atleast_1d(eigenvalues)
    mag = np.abs(ev)
    order = np.argsort(mag)[::-1]
    top = ev[order[:2]]
    # complex conjugate pair: two entries with nonzero imag, mirrored signs
    return bool(
        np.all(np.abs(top.imag) > tol)
        and np.sign(top[0].imag) != np.sign(top[1].imag)
        and np.isclose(abs(top[0]), abs(top[1]), rtol=1e-6)
    )


def estimate_loop_gain(
    traj: np.ndarray,
    decay_upper: float = DECAY_UPPER,
    amplify_lower: float = AMPLIFY_LOWER,
) -> dict:
    """Estimate loop gain rho(G_hat) from a (T, 3) trajectory and band it.

    Returns dict with rho, eigenvalues, regime (PWM §V band) and oscillation
    flag. The oscillation flag is reported *alongside* the magnitude band —
    a delayed loop can oscillate while |rho| < 1 (bounded cycling).
    """
    traj = np.asarray(traj, dtype=float)
    if traj.ndim != 2 or traj.shape[1] != 3 or traj.shape[0] < 4:
        raise ValueError("traj must be (T, 3) with T >= 4")
    G_hat, _ = _ridge_mar1(traj)
    ev = np.linalg.eigvals(G_hat)
    rho = float(np.max(np.abs(ev)))
    osc = lead_pair_is_complex(ev)
    return {
        "rho": rho,
        "eigenvalues": ev,
        "oscillatory": osc,
        "regime": band_regime(rho, decay_upper, amplify_lower),
    }


def band_regime(rho: float, decay_upper: float = DECAY_UPPER, amplify_lower: float = AMPLIFY_LOWER) -> str:
    if rho < decay_upper:
        return "decay"
    if rho <= amplify_lower:
        return "persist"
    return "amplify"


def hypothesis_entropy(weights: np.ndarray) -> float:
    """Shannon entropy (nats) of a normalized hypothesis-weight vector."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("weights must be a 1-D non-empty vector")
    w = w / w.sum()
    w = np.clip(w, 0.0, None)
    nz = w[w > 0]
    return float(-(nz * np.log(nz)).sum())


def weights_from_logcount(log_count: float) -> np.ndarray:
    """Equal-weight hypothesis vector of length k = round(exp(log_count))."""
    lc = float(np.clip(log_count, 0.0, 12.0))
    k = max(1, int(round(np.exp(lc))))
    return np.full(k, 1.0 / k)


def epistemic_velocity(
    hyp_entropy: np.ndarray,
    evidence: np.ndarray,
    eps: float = 1e-3,
) -> float:
    """Velocity = d(log H)/d(log E) over an episode; NaN sentinel if E is flat.

    PWM §VII.3: > 0 means hypothesis collapse with flat evidence — the
    epistemic-convergence signature. Guarded: flat or negative evidence
    movement returns the sentinel rather than dividing by ~0.
    """
    h = np.asarray(hyp_entropy, dtype=float)
    e = np.asarray(evidence, dtype=float)
    if h.shape != e.shape or h.size < 2:
        raise ValueError("hyp_entropy and evidence must be same-length arrays of length >= 2")
    dE = e[-1] - e[0]
    dH = h[-1] - h[0]
    if abs(dE) < eps:
        return EVIDENCE_FLAT_SENTINEL
    return float(dH / dE)


def oscillation_signature(series: np.ndarray, min_direction_changes: int = 3) -> dict:
    """Detect BFS oscillation (bounded relief→doubt cycling) in a scalar series.

    Oscillatory regime test (PWM §V): the series cycles with bounded amplitude
    rather than diverging. Returns direction-change count, lag-1 increment
    autocorrelation (negative = alternating), and boolean verdict.
    """
    s = np.asarray(series, dtype=float).ravel()
    if s.size < 4:
        raise ValueError("series must have length >= 4")
    d = np.diff(s)
    # direction changes: successive increments of opposite sign
    signs = np.sign(d)
    signs = signs[signs != 0]
    direction_changes = int((np.diff(signs) != 0).sum())
    # lag-1 autocorrelation of increments
    d0 = d - d.mean()
    denom = float(np.sqrt((d0[:-1] ** 2).sum() * (d0[1:] ** 2).sum()))
    lag1 = float((d0[:-1] * d0[1:]).sum() / denom) if denom > 1e-12 else 0.0
    return {
        "direction_changes": direction_changes,
        "lag1_increment_autocorr": lag1,
        "oscillatory": bool(direction_changes >= min_direction_changes and lag1 < 0.0),
        "amplitude": float(s.max() - s.min()),
    }
