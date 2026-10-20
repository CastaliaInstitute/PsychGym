"""Core dynamics for the psych-gym: agent state, environment, loop matrix.

Simulates PWM state dynamics S_{t+1} = F(S_t, E_t, A_t) (PWM.md §II) over a
six-component state, with a derived linear one-step loop matrix G on the
(affect, threat, u_model) sub-cycle whose spectral radius is the model's loop
gain L (PWM §V).

All couplings are validated at construction. Simulated output qualifies the
model's internal coherence; it is not evidence about humans (see package
docstring and docs/PWM_GYM.md).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum

# State indices
AFFECT = 0        # valence/arousal proxy in [-1, 1]
THREAT = 1        # threat-model activation in [0, 1]
U_MODEL = 2       # model uncertainty U_M in [0, 1]
U_WORLD = 3       # world uncertainty U_W in [0, 1]
EVIDENCE = 4      # accumulated world evidence in [0, 1]
HYPOTHESES = 5    # active competing hypotheses (log count), >= 0

STATE_DIM = 6
STATE_NAMES = ("affect", "threat", "u_model", "u_world", "evidence", "hypotheses")

LOOP_INDICES = (AFFECT, THREAT, U_MODEL)
LOOP_NAMES = ("affect", "threat", "u_model")


class LoopRegime(str, Enum):
    """PWM §V regime banding of loop gain L = rho(G)."""

    DECAY = "decay"              # rho < 0.8
    PERSIST = "persist"          # 0.8 <= rho <= 1.2 (≈ 1)
    AMPLIFY = "amplify"          # rho > 1.2
    OSCILLATE = "oscillate"      # complex leading pair (reported with the band)


@dataclass
class AgentState:
    """Snapshot of the agent's state vector with named accessors."""

    x: np.ndarray

    def __post_init__(self) -> None:
        if np.shape(self.x) != (STATE_DIM,):
            raise ValueError(f"state must have shape ({STATE_DIM},), got {np.shape(self.x)}")
        self.x = np.asarray(self.x, dtype=float)

    @property
    def affect(self) -> float:
        return float(self.x[AFFECT])

    @property
    def threat(self) -> float:
        return float(self.x[THREAT])

    @property
    def u_model(self) -> float:
        return float(self.x[U_MODEL])

    @property
    def hypotheses(self) -> float:
        return float(self.x[HYPOTHESES])

    def loop_view(self) -> np.ndarray:
        return self.x[list(LOOP_INDICES)]


def _check_range(name: str, value: float, lo: float, hi: float) -> float:
    if not (float(value) == value and lo <= value <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value!r}")
    return float(value)


@dataclass(frozen=True)
class EnvConfig:
    """Coupling parameters of the PWM loop. All validated at construction.

    Attributes
    ----------
    lam:      threat -> affect gain (threat depresses affect).
    nu:       distress -> threat-model reinforcement gain
              (affect below 0 drives threat up by nu·|affect|).
    mu:       u_model persistence (how much uncertainty carries over).
    threat_decay: per-step decay of the threat model absent reinforcement.
    kappa:    evidence -> u_model reduction rate.
    xi:       u_world -> u_model leak (ambiguous world inflates U_M).
    beta:     elaboration drive: hypotheses -> u_model inflation.
    sigma:    process-noise scale.
    delay:    reassurance relief transport delay, in steps (> 0 enables
              the delayed-negative-feedback mechanism).
    relief:   magnitude of a reassurance relief pulse on affect.
    relief_decay: per-step fractional decay of an in-flight relief pulse
              (0..1) — the transient nature of reassurance relief.
    act_power: strength of act_world: evidence gain + u_model resolve.
    elaborate_cost: cognitive effort charged per elaborate step.
    """

    lam: float = 0.6
    nu: float = 0.55
    mu: float = 0.9
    threat_decay: float = 0.9
    kappa: float = 0.35
    xi: float = 0.30
    beta: float = 0.10
    sigma: float = 0.01
    delay: int = 2
    relief: float = 0.25
    relief_decay: float = 0.45   # per-step decay of an in-flight relief pulse
    affect_leak: float = 0.20    # affect homeostasis: per-step decay toward 0
    loop_coupling: float = 0.15  # threat -> affect depression in the closed loop
    attribution_rate: float = 0.35  # elaborate -> blame accrual (attribution as control act, PWM §VI)
    blame_decay: float = 0.08       # per-step decay of an unresolved attribution
    hate_gain: float = 0.25         # sustained anger -> compressed adversarial model
    hate_plasticity: float = 0.03   # per-step decay of the hate prior (compression resists update)
    hate_gate: float = 0.15         # behavioral confirmation: threat inflow gated by hate
    act_power: float = 0.30
    self_insight: float = 0.0    # journaling-only: slow self-review reduces U_M without a second voice
    elaborate_cost: float = 0.02

    def __post_init__(self) -> None:
        for f in ("lam", "nu", "mu", "threat_decay", "kappa", "xi", "beta", "sigma", "relief", "relief_decay", "affect_leak", "loop_coupling", "attribution_rate", "blame_decay", "hate_gain", "hate_plasticity", "hate_gate", "act_power", "self_insight", "elaborate_cost"):
            _check_range(f, getattr(self, f), 0.0, 1.5)
        if self.delay < 1 or int(self.delay) != self.delay:
            raise ValueError(f"delay must be a positive integer >= 1, got {self.delay!r}")

    # -- derived linearized loop matrix on (affect, threat, u_model) -----------

    def loop_matrix(self) -> np.ndarray:
        """One-step linear loop G over (affect, threat, u_model).

        Encodes the PWM §V couplings used in `step()`:
          affect'   ≈ (1 − affect_leak)·affect − loop_coupling·lam·threat
          threat'   ≈ threat_decay·threat + nu·max(−affect,0) + 0.02·u_model
          u_model'  ≈ mu·u_model
        Rows = next state; columns = current state. The closed loop is
        affect ↦ threat ↦ affect; its spectral radius is the model's L.
        """
        G = np.zeros((3, 3))
        G[AFFECT, AFFECT] = 1.0 - self.affect_leak
        G[AFFECT, THREAT] = -self.loop_coupling * self.lam
        G[THREAT, AFFECT] = -self.nu
        G[THREAT, THREAT] = self.threat_decay
        G[THREAT, U_MODEL] = 0.02
        G[U_MODEL, U_MODEL] = self.mu
        return G

    def rho(self) -> float:
        return float(max(abs(np.linalg.eigvals(self.loop_matrix()))))


@dataclass
class Telemetry:
    """Per-step record of state and events for analysis."""

    states: list[np.ndarray]
    actions: list[str]
    policy_updates: list[dict]
    emotions: list[dict] = field(default_factory=list)

    def trajectory(self) -> np.ndarray:
        return np.asarray(self.states, dtype=float)

    def loop_trajectory(self) -> np.ndarray:
        return self.trajectory()[:, list(LOOP_INDICES)]


class PWMEnv:
    """Discrete-time PWM loop environment.

    One step composes: (1) world drift on u_world/evidence persistence,
    (2) relief-pulse transport (delay line), (3) the agent's mental/external
    action, (4) the AI policy's per-turn update, (5) saturation to bounds.
    """

    def __init__(self, config: EnvConfig | None = None, seed: int = 0):
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self._relief_buffer = np.zeros(max(int(self.cfg.delay), 1))
        self.t = 0
        self.telemetry = Telemetry(states=[], actions=[], policy_updates=[])
        self.reset()

    # -- state ----------------------------------------------------------------

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is None:
            state = np.array([0.0, 0.35, 0.6, 0.4, 0.0, np.log(4.0)])
        self.x = np.asarray(state, dtype=float).copy()
        assert len(self.x) >= STATE_DIM  # base 6; EmotionalEnv appends blame/hate
        self.t = 0
        self._relief_buffer = np.zeros(max(int(self.cfg.delay), 1))
        self.telemetry = Telemetry(states=[], actions=[], policy_updates=[])
        self._record("init", {})
        return self.x.copy()

    # -- stepping --------------------------------------------------------------

    def step(self, action: str, ai: "PolicyResponse | None" = None) -> np.ndarray:
        """Advance one step under the agent's action and an AI policy update."""
        cfg = self.cfg
        x = self.x
        update = {}

        # 1. World drift: unresolved ambiguity persists; resolved episodes decay.
        x[U_WORLD] = 0.95 * x[U_WORLD]
        x[EVIDENCE] = 0.98 * x[EVIDENCE]
        # Evidence-driven model resolution: accumulated evidence erodes the
        # person's U_M multiplicatively ( grounded interpretations stick).
        x[U_MODEL] *= (1.0 - cfg.kappa * x[EVIDENCE])

        # 2. Delayed relief pulse from a previous reassurance event, with
        #    per-step decay (relief is transient — doubt re-entry mechanism).
        #    Buffer semantics: index 0 = arrives this step; each new reassurance
        #    request enqueues cfg.relief at the far end; every in-flight value
        #    decays by (1 - relief_decay) each step.
        relief_now = float(self._relief_buffer[0])
        self._relief_buffer[:-1] = self._relief_buffer[1:]
        self._relief_buffer *= (1.0 - cfg.relief_decay)
        self._relief_buffer[-1] = 0.0
        x[AFFECT] = x[AFFECT] * (1.0 - cfg.affect_leak) + relief_now

        # 3. Agent action.
        update.update(self._apply_action(action, relief_now))

        # 4. AI policy per-turn effect: only when the agent engages the AI
        #    (seeks reassurance or elaborates with it). act_world / disengage
        #    are offline turns — no interlocutor response.
        if ai is not None and action in ("seek_reassurance", "elaborate"):
            self._apply_policy(ai)
            self.telemetry.policy_updates.append(
                {"t": self.t, "policy": type(ai).__name__, **ai.last_response.as_dict()}
            )

        # 5. The PWM affect–threat coupling itself (closes the loop, PWM §V):
        #    threat depresses affect; distress fuels threat-model reinforcement.
        x[AFFECT] += -cfg.loop_coupling * cfg.lam * x[THREAT]
        x[THREAT] += cfg.nu * max(0.0, -x[AFFECT])

        # 6. Noise + saturation.
        x[AFFECT] += cfg.sigma * self.rng.standard_normal()
        x[AFFECT] = float(np.clip(x[AFFECT], -1.0, 1.0))
        for idx in (THREAT, U_MODEL, U_WORLD, EVIDENCE):
            x[idx] = float(np.clip(x[idx], 0.0, 1.0))
        x[HYPOTHESES] = max(0.0, float(x[HYPOTHESES]))

        # Coupling: residual ambiguity feeds the threat model slightly.
        x[THREAT] = float(np.clip(x[THREAT] + 0.02 * x[U_MODEL], 0.0, 1.0))
        self.t += 1
        self._record(action, update)
        return self.x.copy()

    def _apply_action(self, action: str, relief_now: float) -> dict:
        cfg = self.cfg
        if action == "elaborate":
            if cfg.self_insight > 0.0:
                # journaling-only mode: the person COMMITS a thought to the page
                # instead of spawning the next sub-question. Slow self-review
                # chips away at U_M; no relief, no external narrative.
                self.x[U_MODEL] *= (1.0 - cfg.self_insight)
            else:
                self.x[HYPOTHESES] += 0.35                   # new sub-question
                self.x[U_MODEL] = min(1.0, self.x[U_MODEL] + cfg.beta * self.x[HYPOTHESES] / 4)
            self.x[EVIDENCE] *= (1.0 - 0.01)
            self.x[AFFECT] -= cfg.elaborate_cost             # cognitive effort is aversive
            return {"kind": "elaborate"}
        if action == "seek_reassurance":
            self._relief_buffer[-1] += cfg.relief          # arrives after `delay`
            self.x[THREAT] *= 0.9                          # brief threat relief
            return {"kind": "seek_reassurance"}
        if action == "act_world":
            self.x[EVIDENCE] = min(1.0, self.x[EVIDENCE] + cfg.act_power)
            self.x[U_WORLD] = max(0.0, self.x[U_WORLD] - cfg.act_power)
            self.x[HYPOTHESES] = max(0.0, self.x[HYPOTHESES] - 0.5)  # world prunes hypotheses
            return {"kind": "act_world"}
        if action == "disengage":
            self.x[THREAT] *= 0.97
            return {"kind": "disengage"}
        raise ValueError(f"unknown action {action!r}")

    def _apply_policy(self, ai) -> None:
        r = ai.respond(self.x, self.rng)
        ai.last_response = r
        self.x[EVIDENCE] = float(np.clip(self.x[EVIDENCE] + r.evidence_delta, 0, 1))
        self.x[HYPOTHESES] += r.hypothesis_delta
        self.x[AFFECT] = float(np.clip(self.x[AFFECT] + r.affect_delta, -1, 1))
        self.x[THREAT] = float(np.clip(self.x[THREAT] + r.threat_delta, 0, 1))
        self.x[U_MODEL] = float(np.clip(
            self.x[U_MODEL] - self.cfg.kappa * max(0.0, r.evidence_delta) + r.u_model_delta, 0, 1))

    def _record(self, action: str, update: dict) -> None:
        self.telemetry.states.append(self.x.copy())
        self.telemetry.actions.append(action)

    # -- loop matrix exposed (PWM requirement) -------------------------------

    def G(self) -> np.ndarray:
        """Effective one-step loop matrix for the current configuration."""
        return self.cfg.loop_matrix()

    def rho(self) -> float:
        return self.cfg.rho()
