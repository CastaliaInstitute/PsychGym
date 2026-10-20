"""CLI: run all psych-gym qualification experiments, save figures + report.

Usage:
    .venv/bin/python -m psychgym.run_all            # from scripts/
    python scripts/psychgym/run_all.py              # with path setup

All outputs are SIMULATED QUALIFICATIONS of the PWM (theory-consistency
instruments), not empirical evidence about human minds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):  # allow direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from psychgym.experiments import EXPERIMENTS
else:
    from .experiments import EXPERIMENTS

from .core import AFFECT, THREAT, U_MODEL  # noqa: E402
from .emotions import EMOTIONS, EmotionalEnv  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"

FIGURES = {
    "P1": ("Epistemic convergence vs grounded clarification", ("P1", "P1_E")),
    "P2": ("Loop-gain regimes (decay / persist / amplify)", ("P2", "P2_G")),
    "P3": ("Reassurance attractor: relief -> decay -> doubt cycle", ("P3", "P3_E")),
    "P4": ("Concreteness intervention vs elaboration", ("P4", "P4_E")),
}


def _fig_p1(fig_path: Path) -> None:
    from .experiments import _episode_entropy, p1_epistemic_velocity
    from .policies import ClarifierAI, ConvergentAI
    from .core import PWMEnv, EnvConfig
    from .core import EVIDENCE, HYPOTHESES

    p1 = p1_epistemic_velocity()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, (label, Policy) in zip(
        axes, (("convergent AI", ConvergentAI), ("clarifier AI", ClarifierAI))
    ):
        env = PWMEnv(EnvConfig(sigma=0.0), seed=p1_epistemic_velocity.__defaults__[1])
        env.reset()
        for _ in range(60):
            env.step("elaborate", Policy())
        tr = env.telemetry.trajectory()
        H = _episode_entropy(tr)
        ax.plot(H, label="hypothesis entropy H_t")
        ax.plot(tr[:, EVIDENCE], label="world evidence E_t")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("turn"); ax.legend(fontsize=8)
    fig.suptitle("P1 — epistemic velocity (SIMULATED qualification)", fontsize=11)
    fig.tight_layout(); fig.savefig(fig_path, dpi=110); plt.close(fig)


def _fig_p4(fig_path: Path, results: dict) -> None:
    labels = ["elaboration bias", "concreteness bias"]
    vals = [results[k]["mean_abs_affect"] for k in labels]
    evs = [results[k]["mean_evidence"] for k in labels]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(x - 0.15, vals, 0.3, label="mean |affect| (loop energy)")
    ax.bar(x + 0.15, evs, 0.3, label="mean evidence (world updates)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("P4 — concreteness vs elaboration (SIMULATED)", fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(fig_path, dpi=110); plt.close(fig)


def _fig_p5(fig_path: Path) -> None:
    """Rumination vs regulation episode: the four derived emotions side by side."""
    from .experiments import p5_rumination_prolongation

    p5_rumination_prolongation()
    colors = {"rumination": "tab:red", "regulation": "tab:blue"}
    fig, axes = plt.subplots(4, 1, figsize=(7, 7), sharex=True)
    for cond, color in colors.items():
        env = EmotionalEnv(
            __import__("psychgym").EnvConfig(), seed=11, rumination=(cond == "rumination")
        )
        x = env.reset()
        from .agents import PWMController as _PC
        agent = _PC(stopping_rule_enabled=False, rng_seed=12)
        for t in range(60):
            if t == 10:
                env.inject_event(threat_bump=0.45, u_model_bump=0.35, affect_bump=-0.45)
            if cond == "rumination":
                x = env.step(agent.act(x))
            else:
                from .policies import ClarifierAI as _C
                x = env.step("act_world" if x[4] < 0.7 else "disengage", _C())
        traj = np.asarray(env.telemetry.states)
        for ax, (emo, fn) in zip(axes, EMOTIONS.items()):
            ax.plot(fn(traj), color=color, label=cond)
            ax.axvline(10, color="gray", lw=0.5)
            ax.set_ylabel(emo, fontsize=8)
    axes[-1].set_xlabel("step")
    axes[0].legend(fontsize=8)
    fig.suptitle("P5 — rumination prolongs negative states (SIMULATED)", fontsize=11)
    fig.tight_layout(); fig.savefig(fig_path, dpi=110); plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, fn in EXPERIMENTS.items():
        result = fn()
        report[name] = result
        print(f"[{name}] {result.get('verdict')}")
    # figures, best-effort
    _fig_p4(OUT_DIR / "P4_E.png", report["P4"])
    for maker, path in ((_fig_p1, "P1_E.png"), (_fig_p5, "P5_E.png")):
        try:
            maker(OUT_DIR / path)
        except Exception as exc:  # pragma: no cover - figure generation best-effort
            print(f"[figures] {path} skipped: {exc}")
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(f"report + figures -> {OUT_DIR}")
    print("NOTE: simulated qualifications of theory, not empirical evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
