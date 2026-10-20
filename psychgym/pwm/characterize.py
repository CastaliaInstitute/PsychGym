"""Characterization of the PWM model and its parameters — without human data.

Everything here runs on `EnvConfig`-generated structure/trajectories and
closed-form algebra. No human data files, no fitted human parameters, and no
probability of contact with them: the outputs are SIMULATED / ANALYTIC
qualifications (theory instruments — see package docstring).

Deliverables
------------
1. regime_map:          spectral radius across a (lam, nu) coupling grid
                        banded per PWM §V, as a figure + summary counts.
2. analytic_boundary:   closed-form amplification boundary of the
                        affect–threat 2×2 block, validated against the grid:
                        amplification  ⟺  c·lam·nu > affect_leak·(1−threat_decay)
                        (c = loop_coupling).
3. real_roots_theorem:  the documented affect–threat block ALWAYS has real
                        eigenvalues (discriminant (a−d)² + 4·c·lam·nu > 0),
                        so oscillation must come from the delayed reassurance
                        loop, not the linear core — verified on the grid.
4. sensitivity:         one-at-a-time ∂ρ/∂parameter (tornado ordering).
5. identifiability:     planted-ρ recovery error of the trajectory estimator
                        vs process noise (σ sweep), fixed seeds.
6. u_model_dominance:   analytic rule — when mu exceeds the loop block's ρ,
                        the slowest mode is unresolved-uncertainty persistence.

CLI:  python -m psychgym.characterize
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "out"

import numpy as np

from .core import EnvConfig

LABEL = "SIMULATED / ANALYTIC characterization of the model — not empirical evidence"


# -- analytic core -------------------------------------------------------------

def loop_block(config: EnvConfig) -> np.ndarray:
    """The 2×2 affect–threat block of the structural G (PWM §V)."""
    cfg = config
    return np.array([
        [1.0 - cfg.affect_leak, -cfg.loop_coupling * cfg.lam],
        [-cfg.nu,               cfg.threat_decay],
    ])


def block_trace_det(config: EnvConfig) -> tuple[float, float]:
    B = loop_block(config)
    return float(np.trace(B)), float(np.linalg.det(B))


def _p_at(config: EnvConfig, r: float) -> float:
    """P(r) = r² − tr·r + det of the 2×2 block."""
    tr, det = block_trace_det(config)
    return r * r - tr * r + det


def amplification_boundary_value(config: EnvConfig) -> float:
    """Worst-case root excursion beyond the unit circle (positive branch).

    Positive amplification  ⟺  P(1) < 0  ⟺  c·lam·nu > affect_leak·(1−threat_decay)
    (hyperbola in (lam, nu)). Use `amplifies()` for the full |λ|>1 test,
    which also excludes the saddle case via P(−1).
    """
    a = 1.0 - config.affect_leak
    d = config.threat_decay
    return (1.0 - a) * (1.0 - d) - config.loop_coupling * config.lam * config.nu


def amplifies(config: EnvConfig) -> bool:
    """True when the 2×2 block has any root with |λ| > 1.

    Full condition for real 2×2 blocks:  P(1) < 0  OR  P(−1) < 0
    (positive-root excursion through +1, or saddle excursion through −1).
    """
    return _p_at(config, 1.0) < 0.0 or _p_at(config, -1.0) < 0.0


def block_real_roots(config: EnvConfig) -> bool:
    """Discriminant (a−d)² + 4·c·lam·nu > 0 for positive couplings → real roots."""
    a = 1.0 - config.affect_leak
    d = config.threat_decay
    return (a - d) ** 2 + 4.0 * config.loop_coupling * config.lam * config.nu > 0.0


def u_model_dominant(config: EnvConfig) -> float:
    """ρ of the loop block (the 3×3 G is block-diagonal with mu on its own axis)."""
    return max(abs(np.linalg.eigvals(loop_block(config))))


def rho_analytic(config: EnvConfig) -> float:
    """Closed-form structural ρ: max(loop-block ρ, mu)."""
    return float(max(config.mu, u_model_dominant(config)))


# -- grid sweep ----------------------------------------------------------------

def regime_map(
    lam_range: tuple[float, float] = (0.05, 1.4),
    nu_range: tuple[float, float] = (0.05, 1.4),
    n: int = 41,
    base: EnvConfig | None = None,
) -> dict:
    cfg = base or EnvConfig()
    lams = np.linspace(*lam_range, n)
    nus = np.linspace(*nu_range, n)
    rho = np.empty((n, n))
    osc_flag = np.zeros((n, n), dtype=bool)
    amp_pred = np.empty((n, n), dtype=bool)
    for i, lam in enumerate(lams):
        for j, nu in enumerate(nus):
            c = EnvConfig(
                lam=lam, nu=nu, threat_decay=cfg.threat_decay,
                affect_leak=cfg.affect_leak, loop_coupling=cfg.loop_coupling, mu=cfg.mu,
                sigma=cfg.sigma, delay=cfg.delay, relief=cfg.relief,
                relief_decay=cfg.relief_decay, act_power=cfg.act_power,
                elaborate_cost=cfg.elaborate_cost,
            )
            rho[i, j] = cfg_loop_rho(c)
            osc_flag[i, j] = _any_complex(c)
            amp_pred[i, j] = amplifies(c)
    bands = np.select(
        [rho < 0.8, rho <= 1.2],
        [0, 1], default=2,
    )
    counts = np.bincount(bands.ravel(), minlength=3)
    return {
        "lams": lams, "nus": nus, "rho": rho, "bands": bands,
        "lam_range": lam_range, "nu_range": nu_range,
        "osc_flag": osc_flag, "amp_pred": amp_pred,
        "band_counts": {"decay": int(counts[0]), "persist": int(counts[1]), "amplify": int(counts[2])},
        "any_oscillatory": bool(osc_flag.any()),
        "analytic_matches_numeric": bool(np.mean(amp_pred == (rho > 1.0)) > 0.999),
    }


def cfg_loop_rho(c: EnvConfig) -> float:
    return max(abs(np.linalg.eigvals(c.loop_matrix())))


def _any_complex(c: EnvConfig) -> bool:
    ev = np.linalg.eigvals(c.loop_matrix())
    return bool(np.any(np.abs(ev.imag) > 1e-9))


# -- sensitivity ---------------------------------------------------------------

def sensitivity(base: EnvConfig | None = None, frac: float = 0.10) -> dict:
    """One-at-a-time ∂ρ/∂parameter around a base configuration."""
    cfg = base or EnvConfig()
    out = {}
    for f in ("lam", "nu", "mu", "threat_decay", "kappa", "xi", "beta", "sigma",
              "relief", "relief_decay", "affect_leak", "loop_coupling"):
        lo = max(0.0, getattr(cfg, f) * (1 - frac))
        hi = min(1.5, getattr(cfg, f) * (1 + frac))
        lo_c = config_with(cfg, **{f: lo})
        hi_c = config_with(cfg, **{f: hi})
        # rho is computed on the structural loop matrix (deterministic)
        out[f] = float((cfg_loop_rho(hi_c) - cfg_loop_rho(lo_c)) / (hi_c.rho() and ((hi - lo) if hi > lo else 1e-9)))
    order = sorted(out.items(), key=lambda kv: -abs(kv[1]))
    return {"gradient": out, "ordering": [k for k, _ in order]}


def config_with(cfg: EnvConfig, **overrides) -> EnvConfig:
    d = {f: getattr(cfg, f) for f in ("lam", "nu", "mu", "threat_decay", "kappa", "xi",
                                       "beta", "sigma", "delay", "relief", "relief_decay",
                                       "affect_leak", "loop_coupling", "act_power", "elaborate_cost")}
    d.update(overrides)
    return EnvConfig(**d)


# -- identifiability probe -----------------------------------------------------

def identifiability(seed: int = 5,
                    sigmas: tuple[float, ...] = (0.002, 0.005, 0.01, 0.02, 0.05),
                    planted: tuple[float, ...] = (0.6, 0.9, 1.1)) -> dict:
    """Planted-ρ recovery error vs process noise (no human data involved).

    Uses linear planted dynamics (the estimator's intended domain), reporting
    mean |error| and max |error| per noise level.
    """
    from .metrics import estimate_loop_gain

    rng = np.random.default_rng(seed)
    results = {}
    for s in sigmas:
        errs = {}
        for p in planted:
            G = np.eye(3) * p
            G[2, 0] = 0.05
            x = np.array([0.5, -0.3, 0.2])
            traj = [x.copy()]
            for _ in range(60):
                x = G @ x + s * rng.standard_normal(3)   # linear, unclipped
                traj.append(x.copy())
            est = estimate_loop_gain(np.asarray(traj))
            errs[round(p, 2)] = abs(est["rho"] - p)
        results[str(s)] = {
            "mean_abs_error": float(np.mean(list(errs.values()))),
            "max_abs_error": float(np.max(list(errs.values()))),
            "per_planted": errs,
        }
    return results


# -- main ----------------------------------------------------------------------

def characterize(save: bool = True, base: EnvConfig | None = None) -> dict:
    """Run the full no-human-data characterization and (optionally) save it."""
    rm = regime_map(base=base)
    # u_model dominance on the default configuration
    default = base or EnvConfig()
    dominant = u_model_dominant(default)
    result = {
        "label": LABEL,
        "analytic": {
            "amplification_boundary": "positive branch: loop_coupling·lam·nu > affect_leak·(1−threat_decay); full |λ|>1 test: P(1) < 0 or P(−1) < 0",
            "real_roots_theorem": "(a−d)² + 4·loop_coupling·lam·nu > 0 for positive couplings ⟹ eigenvalues always real ⟹ oscillation must arise from the delayed reassurance loop, not the linear core",
            "u_model_dominance": "ρ = max(ρ_loop_block, mu); when mu exceeds the loop block, the slowest mode is unresolved-uncertainty persistence",
            "default_config": {
                "rho_numeric": round(default.rho(), 4),
                "rho_analytic": round(rho_analytic(default), 4),
                "amplification_boundary_value": round(amplification_boundary_value(default), 4),
                "block_real_roots": block_real_roots(default),
                "u_model_dominant": bool(default.mu > dominant),
            },
        },
        "regime_map": {
            "grid": "41×41 over (lam, nu) ∈ [0.05, 1.4]², other couplings at defaults",
            "band_counts": rm["band_counts"],
            "any_oscillatory_eigenvalues": rm["any_oscillatory"],
            "analytic_boundary_matches_grid": rm["analytic_matches_numeric"],
        },
        "sensitivity": sensitivity(),
        "identifiability": identifiability(),
    }
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        _save_figures(rm, base)
        (OUT_DIR / "characterization.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _save_figures(rm: dict, base: EnvConfig | None = None) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return
    cfg = base or EnvConfig()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    bands = rm["bands"]
    lo0, hi0 = rm["lam_range"]
    lo1, hi1 = rm["nu_range"]
    ax.imshow(bands, origin="lower", cmap="viridis",
              extent=(lo0, hi0, lo1, hi1), aspect="auto", alpha=0.85)
    # exact analytic boundary in (λ, ν):  c·λ·ν = leak·(1−d)  →  ν = rhs / λ
    rhs = cfg.affect_leak * (1.0 - cfg.threat_decay) / cfg.loop_coupling
    lams = np.linspace(lo0, hi0, 200)
    ax.plot(lams, rhs / lams, color="red", lw=1.5,
            label="analytic amplify boundary (cλν = leak·(1−d))")
    ax.set_xlabel("lam"); ax.set_ylabel("nu")
    ax.set_title("ρ(G) regime map (decay 0 / persist 1 / amplify 2)", fontsize=10)
    ax.set_ylim(lo1, hi1)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "regime_map.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    out = characterize()
    print(json.dumps({k: v for k, v in out.items() if k != "sensitivity"}, indent=2)[:2000])
    print(f"[{out['label']}]")
    raise SystemExit(0)
