"""Emotions layer for the psych-gym: fear, anger, hate, suffering as PWM regimes.

Derived emotion definitions (PWM §V–VI, declared formulas — model constructs,
not claims about humans):

    fear      = threat × U_M                    (§V: fear ≈ threat × uncertainty)
    blame     = latent attribution channel       (elaborate-step attribution accrues
                                                 under unresolved ambiguity; §VI anger)
    anger     = threat × blame                   (attribution introduces agency)
    hate      = compressed adversarial prior     (slow-decay channel; gates threat
                                                 inflow — behavioral confirmation)
    suffering = aversiveness × importance × persistence × (1 − control)
              with control proxied by accumulated world evidence (§VI four-term S)

`EmotionalEnv` extends `PWMEnv` with two state channels (blame, hate) and the
delayed-reassurance machinery inherited unchanged. The rumination flag switches
the attribution/hate dynamics ON (rumination-prone regime) or soft (regulation
regime, where evidence-driven action resets attribution before it consolidates).

Simulation only — theory instrument, not evidence about humans.
"""

from __future__ import annotations

import numpy as np

from .core import (
    AFFECT,
    EVIDENCE,
    HYPOTHESES,
    THREAT,
    U_MODEL,
    Telemetry,
    PWMEnv,
)

BLAME = 6
HATE = 7
EMOTIONAL_STATE_DIM = 8


# -- derived emotions (declared formulas over the 8-dim trajectory) ------------

def fear(traj: np.ndarray) -> np.ndarray:
    """PWM §V: fear ≈ threat × uncertainty."""
    return np.clip(traj[:, THREAT] * traj[:, U_MODEL], 0.0, 1.0)


def anger(traj: np.ndarray) -> np.ndarray:
    """PWM §VI: agency enters via attribution; anger = threat × blame."""
    return np.clip(traj[:, THREAT] * traj[:, BLAME], 0.0, 1.0)


def hate(traj: np.ndarray) -> np.ndarray:
    """PWM §VI: the compressed adversarial model channel."""
    return np.clip(traj[:, HATE], 0.0, 1.0)


def suffering(traj: np.ndarray, persistence_window: int = 10) -> np.ndarray:
    """PWM §VI: S = aversiveness × importance × persistence × (1 − control).

    - aversiveness: negative part of affect
    - importance:   threat (value at risk is carried by the threat model)
    - persistence:  rolling mean of aversiveness (matters when it lasts)
    - (1 − control): 1 − world evidence (no effective action taken)
    """
    traj = np.asarray(traj, dtype=float)
    aversiveness = np.clip(-traj[:, AFFECT], 0.0, 1.0)
    importance = traj[:, THREAT]
    persistence = np.convolve(aversiveness, np.ones(persistence_window) / persistence_window, mode="same")
    return np.clip(aversiveness * importance * persistence * (1.0 - traj[:, EVIDENCE]), 0.0, 1.0)


EMOTIONS = {"fear": fear, "anger": anger, "hate": hate, "suffering": suffering}


def duration_above(series: np.ndarray, threshold: float) -> float:
    """Fraction of steps above a threshold (the prolongation measure)."""
    s = np.asarray(series, dtype=float)
    return float((s > threshold).mean())


def recovery_latency(series: np.ndarray, fall_fraction: float = 0.5) -> int:
    """Steps from the peak until the series falls below `fall_fraction`·peak.
    Returns the remaining length if it never falls (no recovery)."""
    s = np.asarray(series, dtype=float)
    peak_idx = int(np.argmax(s))
    threshold = fall_fraction * s[peak_idx]
    after = s[peak_idx:]
    below = np.where(after < threshold)[0]
    return int(below[0]) if below.size else int(s.size - peak_idx)


# -- environment ---------------------------------------------------------------

class EmotionalEnv(PWMEnv):
    """PWMEnv + attribution (blame) and compressed-threat (hate) channels.

    `rumination=True` models the rumination-prone regime: interpretation-only
    steps accrue blame under unresolved ambiguity, and sustained anger grows
    the hate prior, which gates threat inflow (behavioral confirmation).
    With `rumination=False` the attribution channel resets on evidence-driven
    action — regulation collapses ambiguity before chronic attribution forms.
    """

    def __init__(self, config=None, seed: int = 0, rumination: bool = True):
        self.rumination = bool(rumination)
        super().__init__(config, seed=seed)

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is None:
            base = np.array([0.0, 0.35, 0.6, 0.4, 0.0, np.log(4.0)])
            x = np.zeros(EMOTIONAL_STATE_DIM)
            x[: base.size] = base
        else:
            x = np.asarray(state, dtype=float).copy()
        self.x = x
        assert len(self.x) == EMOTIONAL_STATE_DIM
        self.t = 0
        self._relief_buffer = np.zeros(max(int(self.cfg.delay), 1))
        self.telemetry = Telemetry(states=[], actions=[], policy_updates=[], emotions=[])
        self._record("init", {})
        return self.x.copy()

    def inject_event(self, threat_bump: float = 0.4, u_model_bump: float = 0.35,
                     affect_bump: float = -0.3) -> None:
        """An ambiguous aversive event onset (same stimulation for both conditions)."""
        self.x[THREAT] = min(1.0, self.x[THREAT] + threat_bump)
        self.x[U_MODEL] = min(1.0, self.x[U_MODEL] + u_model_bump)
        self.x[AFFECT] = max(-1.0, self.x[AFFECT] + affect_bump)

    def step(self, action: str, ai=None) -> np.ndarray:
        out = super().step(action, ai)          # inherited loop machinery (6 channels)
        cfg = self.cfg
        # Attribution: interpretation-only effort under unresolved ambiguity
        # converts uncertainty into agency (PWM §VI attribution-as-control).
        if self.rumination and action == "elaborate":
            self.x[BLAME] = min(1.0, self.x[BLAME] + cfg.attribution_rate * self.x[U_MODEL] * (1.0 - self.x[EVIDENCE]))
        if (not self.rumination) and action == "act_world":
            self.x[BLAME] *= 0.6  # evidence-driven attribution is revisable
        self.x[BLAME] *= (1.0 - cfg.blame_decay)
        # Hate: sustained anger compresses the model of the other (§VI).
        anger_now = float(np.clip(self.x[THREAT] * self.x[BLAME], 0.0, 1.0))
        if anger_now > 0.3:
            self.x[HATE] = min(1.0, self.x[HATE] * (1.0 - cfg.hate_plasticity) + cfg.hate_gain * anger_now)
        else:
            self.x[HATE] *= (1.0 - cfg.hate_plasticity)
        # Behavioral confirmation: hate gates threat inflow.
        self.x[THREAT] = min(1.0, self.x[THREAT] + cfg.hate_gate * self.x[HATE])
        self._record_emotions()
        return out

    def _record_emotions(self) -> None:
        x = self.x
        self.telemetry.emotions.append({
            "fear": float(abs(x[THREAT] * x[U_MODEL])),
            "anger": float(np.clip(x[THREAT] * x[BLAME], 0.0, 1.0)),
            "hate": float(np.clip(x[HATE], 0.0, 1.0)),
        })

    # blame/hate are not part of the linear loop matrix; documented as channels.
    def G(self) -> np.ndarray:
        return self.cfg.loop_matrix()
