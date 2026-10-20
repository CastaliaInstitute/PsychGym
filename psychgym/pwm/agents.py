"""Meta-controller agent for the psych-gym (PWM §VI + §VIII).

The controller selects among mental/external actions by expected epistemic-
pressure reduction. Its metacognitive stopping rule is switchable:

    stopping_rule_enabled=True   healthy regime: evidence past threshold
                                 shifts mass to act_world / disengage.
    stopping_rule_enabled=False  rumination-prone regime: elaboration and
                                 reassurance seeking persist; with
                                 ReassurerAI the loop shows the oscillatory
                                 relief -> doubt signature.

Simulation only — see package docstring (theory instrument, not human evidence).
"""

from __future__ import annotations

import numpy as np

from .core import AFFECT, EVIDENCE, THREAT, U_MODEL

ACTIONS = ("elaborate", "seek_reassurance", "act_world", "disengage")


class PWMController:
    """Softmax meta-controller over PWM actions with a switchable stopping rule.

    ROLE: this controller — together with the PWM loop state it acts on — is
    the MODEL OF THE HUMAN, i.e., the unit under test in this gym. All
    measurement (loop gain, uncertainty velocity, affect statistics)
    describes this system's trajectories. The AIInterlocutor policies in
    `policies.py` are the other side of the loop, not the object of test.
    """

    def __init__(
        self,
        temperature: float = 0.25,
        stopping_rule_enabled: bool = True,
        disengage_threshold: float = 0.15,
        reassurance_bias: float = 0.0,
        rng_seed: int = 0,
    ):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)
        self.stopping_rule_enabled = bool(stopping_rule_enabled)
        self.disengage_threshold = float(disengage_threshold)
        self.reassurance_bias = float(reassurance_bias)
        self.rng = np.random.default_rng(rng_seed)

    # -- perception ------------------------------------------------------------

    def observe(self, x: np.ndarray):
        x = np.asarray(x)
        affect, threat, u_model, u_world, evidence = x[0], x[1], x[2], x[3], x[4]
        # P_E = f(U_M, importance, threat) with importance ≈ threat here (PWM §IV).
        return affect, threat, u_model, threat * u_model

    # -- scoring ---------------------------------------------------------------

    def scores(self, x: np.ndarray) -> dict[str, float]:
        """Expected pressure-reduction scores per action (higher = chosen more)."""
        x = np.asarray(x)
        _, threat, u_model, pressure = self.observe(x)
        evidence = float(x[4])
        s = {}
        # Elaborate: promises resolution of an unresolved, pressurized question
        # (locally reinforcing even when globally unproductive — PWM §VI).
        unresolved = u_model * (1.0 - evidence)
        s["elaborate"] = 1.6 * unresolved + 0.3 * threat
        # Reassurance: attractive when distressed and relief is expected.
        s["seek_reassurance"] = 1.2 * max(0.0, -x[AFFECT]) + 0.8 * threat + self.reassurance_bias
        # Act on world: valuable when evidence deficit is the binding constraint
        # and the stopping rule lets the system see that.
        s["act_world"] = 1.5 * (1.0 - evidence) * (0.5 + 0.5 * u_model)
        s["disengage"] = 0.8 * max(0.0, self.disengage_threshold - threat)
        if self.stopping_rule_enabled:
            # Metacognitive gate: interpretation-only loops lose their pull once
            # evidence is available and pressure persists without resolution.
            if evidence >= self.disengage_threshold:
                s["elaborate"] *= 0.35
                s["seek_reassurance"] *= 0.60
        else:
            # Missing stopping rule: no satiation signal for explanation-seeking.
            s["elaborate"] *= 1.35
            s["seek_reassurance"] *= 1.20
        return s

    def action_probabilities(self, x: np.ndarray) -> dict[str, float]:
        raw = np.array([self.scores(x)[a] for a in ACTIONS]) / self.temperature
        raw = raw - raw.max()
        p = np.exp(raw)
        p /= p.sum()
        return dict(zip(ACTIONS, map(float, p)))

    def act(self, x: np.ndarray) -> str:
        p = np.array([self.action_probabilities(x)[a] for a in ACTIONS])
        return str(ACTIONS[int(self.rng.choice(len(ACTIONS), p=p))])
