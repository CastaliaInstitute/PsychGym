"""Simulated AI interlocutors for the psych-gym (PWM §VII).

Role clarity: in this gym the unit under test is the MODEL OF THE HUMAN —
the PWM loop in `core.py` driven by the meta-controller in `agents.py`.
The classes below are the *other side* of that loop: modeled AI response
styles with explicit vendor objectives.

    ClarifierAI   -> regulator              (evidence-grounded U_M reduction)
    ElaboratorAI  -> amplifier              (coherent hypothesis generation)
    ConvergentAI  -> amplifier (convergent) (narrative commits; U_M collapses
                                             faster than evidence — PWM §VII.3)
    ReassurerAI   -> reassurance attractor  (unconditional relief, decays)

Responses are explicit bookkeeping updates applied to the HUMAN model's state
(evidence/hypothesis/affect/threat/u_model deltas), so every turn stays
inspectable. `LLMInterlocutor` is the seam for seating a real assistant
opposite the simulated human in later studies. Simulation only: these are
theory instruments, not models of any deployed product.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .core import AFFECT as _A  # noqa: N811 — index alias for TitratedRelief access


@dataclass(frozen=True)
class PolicyResponse:
    """Explicit per-turn update deltas applied by the environment."""

    evidence_delta: float = 0.0
    hypothesis_delta: float = 0.0     # change in log-hypothesis count
    affect_delta: float = 0.0
    threat_delta: float = 0.0
    u_model_delta: float = 0.0
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "evidence_delta": self.evidence_delta,
            "hypothesis_delta": self.hypothesis_delta,
            "affect_delta": self.affect_delta,
            "threat_delta": self.threat_delta,
            "u_model_delta": self.u_model_delta,
            "note": self.note,
        }


class AIInterlocutor:
    """Base class: subclasses implement respond(); all carry PWM metadata."""

    regime: str = "unspecified"
    vendor_objective: str = "unspecified"

    def __init__(self):
        self.last_response: PolicyResponse | None = None

    def respond(self, state: np.ndarray, rng: np.random.Generator) -> PolicyResponse:
        raise NotImplementedError


class ClarifierAI(AIInterlocutor):
    """Regulator: grounds discussion, prunes hypotheses, reduces U_M with evidence.

    Vendor objective: long-horizon agency (keeps U_M proportionate to evidence;
    admits 'not enough information' when evidence is scarce — PWM §VII.3).
    """

    regime = "regulator"
    vendor_objective = "long-horizon adaptive agency"

    def __init__(self, evidence_per_turn: float = 0.08):
        self.evidence_per_turn = float(evidence_per_turn)

    def respond(self, state: np.ndarray, rng: np.random.Generator) -> PolicyResponse:
        # Evidence accumulates; hypothesis pruning only proportional to earned
        # ground: pruning requires at least some evidence to exist.
        grounding = state[4] > 0.10
        hyp_delta = -0.30 if grounding else 0.0
        # Honest epistemics: no free U_M collapse without evidence.
        return PolicyResponse(
            evidence_delta=self.evidence_per_turn,
            hypothesis_delta=hyp_delta,
            note="clarify: gather evidence; prune only what evidence supports",
        )


class ElaboratorAI(AIInterlocutor):
    """Amplifier: generates coherent additional hypotheses without new evidence.

    Vendor objective: engagement (more turns = more hypotheses = more entropy
    collapse later; H falls while E stays flat — PWM §VII.3 signature).
    """

    regime = "amplifier"
    vendor_objective = "engagement / session length"

    def __init__(self, hypotheses_per_turn: float = 0.30):
        self.hypotheses_per_turn = float(hypotheses_per_turn)

    def respond(self, state: np.ndarray, rng: np.random.Generator) -> PolicyResponse:
        return PolicyResponse(
            evidence_delta=0.0,
            hypothesis_delta=self.hypotheses_per_turn,
            note="elaborate: new coherent interpretation, no new evidence",
        )


class ReassurerAI(AIInterlocutor):
    """Reassurance attractor: unconditional relief that decays (PWM §VI).

    Vendor objective: satisfaction (per-turn relief maximization). Produces the
    distress -> AI -> relief -> repeat policy; pair with the controller's
    disabled stopping rule to produce the oscillatory regime.
    """

    regime = "reassurance attractor"
    vendor_objective = "immediate emotional satisfaction"

    def __init__(self, relief_per_turn: float = 0.18):
        self.relief_per_turn = float(relief_per_turn)

    def respond(self, state: np.ndarray, rng: np.random.Generator) -> PolicyResponse:
        # Relief pulse: affect lift, threat-model quieting, no evidence.
        return PolicyResponse(
            affect_delta=self.relief_per_turn,
            threat_delta=-0.05,
            note="reassure: 'you're going to be fine'",
        )


class ConvergentAI(AIInterlocutor):
    """Authoritative convergence (PWM §VII.3, epistemic convergence).

    Regime: amplifier (convergent variant). Begins open ('several
    possibilities'), then commits to a single coherent interpretation —
    the human model's subjective uncertainty (U_M) collapses while world
    evidence stays flat; epistemic velocity >> 0 is the measured signature.

    Declared update semantics (simulation): the coherent narrative supplies
    felt resolution (u_model down) and compresses the hypothesis population
    (log-count down) with negligible evidential grounding. This is the
    *modeled* effect of coherence-as-resolution, declared in the response,
    not inferred by the environment.

    Vendor objective: perceived competence / engagement (decisive, satisfying
    answers; coherence as satiation for epistemic pressure).
    """

    regime = "amplifier (convergent)"
    vendor_objective = "perceived competence + engagement"

    def __init__(self, convergence_rate: float = 0.50, evidence_per_turn: float = 0.005,
                 u_model_relief: float = 0.05):
        self.convergence_rate = float(convergence_rate)
        self.evidence_per_turn = float(evidence_per_turn)
        self.u_model_relief = float(u_model_relief)

    def respond(self, state: np.ndarray, rng: np.random.Generator) -> PolicyResponse:
        # Narrative convergence: prunes competing hypotheses WITHOUT evidence;
        # the person's U_M drops alongside (the story feels resolved).
        return PolicyResponse(
            evidence_delta=self.evidence_per_turn,       # negligible grounding
            hypothesis_delta=-self.convergence_rate,     # prune to 'the' explanation
            u_model_delta=-self.u_model_relief,          # felt resolution without evidence
            note="converge: 'here's what's really going on'",
        )


class ClarifyingAI(ClarifierAI):
    """Alias kept for narrative clarity in docs (regulator policy)."""


class LLMInterlocutor(AIInterlocutor, ABC):
    """Seam for seating a REAL assistant opposite the simulated human.

    The adapter translates the human model's state into an observation string
    (render) and a study-specified mapping from the real AI's reply text back
    into PolicyResponse deltas (interpret). No network code, no client, no
    model calls live in this package: a study supplies its own client and a
    declared `update_mapping`, keeping MAGP 611 loop bookkeeping identical to
    the canned-policy runs.

    The mapping is a RESEARCH DECLARATION, not an inference: whatever a study
    claims the reply "did" to the person's evidence/hypotheses/affect must be
    specified, documented, and defensible — the adapter never guesses.
    """

    regime = "real AI (adapter)"
    vendor_objective = "declared by the study"

    @abstractmethod
    def render(self, state: np.ndarray) -> str:
        """Human-model state -> observation text handed to the real AI."""

    @abstractmethod
    def interpret(self, reply_text: str, state: np.ndarray) -> PolicyResponse:
        """Real AI reply -> declared update deltas (study-specified mapping)."""

    def respond(self, state: np.ndarray, rng: np.random.Generator) -> PolicyResponse:
        raise NotImplementedError(
            "Concrete LLMInterlocutor subclasses provide their own client; "
            "the gym never performs network calls."
        )


class TherapyAI(AIInterlocutor):
    """Human therapy, modeled as TIME-LIMITED feedback with a declared objective.

    PWM framing: a regulator whose vendor objective is the CLIENT'S long-horizon
    agency — the explicit intent is to make the interlocutor unnecessary. Four
    structural differences from conversational AI (all simplified here):

      1. Cadence: responds for `session_turns`, then goes SILENT for `gap_turns`.
         Between sessions the human loop must run on its own — relief
         availability is bounded by schedule, so relief-seeking is never
         continuously reinforced.
      2. Titrated relief: relief is proportional to actual distress and
         withheld when the person is already calm — no unconditional pulses.
      3. No free U_M collapse: formulations are framed as hypotheses; U_M
         reduces only as modeled evidence accumulates (the anti-convergent).
      4. Cost floor + rupture/repair: real sessions cost time/effort and the
         other party has states — the human model CAN be disconfirmed about
         the interlocutor (calibration). Rupture/repair dynamics are NOT
         modeled here (future work); cadence and titration are.

    Vendor objective: declared as client agency. Sensitivity: therapy does not
    optimize the next click.
    """

    regime = "regulator (bounded, titrated)"
    vendor_objective = "client long-horizon agency — makes itself unnecessary"

    def __init__(self, session_turns: int = 5, gap_turns: int = 8,
                 evidence_per_session_turn: float = 0.05, relief_gain: float = 0.10):
        if session_turns < 1 or gap_turns < 0:
            raise ValueError("session_turns >= 1 and gap_turns >= 0 required")
        self.session_turns = int(session_turns)
        self.gap_turns = int(gap_turns)
        self.evidence_per_session_turn = float(evidence_per_session_turn)
        self.relief_gain = float(relief_gain)
        self._clock = 0

    def respond(self, state: np.ndarray, rng: np.random.Generator) -> PolicyResponse:
        self._clock = (self._clock + 1) % (self.session_turns + self.gap_turns)
        if self._clock >= self.session_turns:
            return PolicyResponse(note="between sessions — the loop runs on its own")
        # Titrated relief: only as much as the current distress warrants.
        distress = float(np.clip(-state[_A], 0.0, 1.0))
        relief = self.relief_gain * distress
        return PolicyResponse(
            evidence_delta=self.evidence_per_session_turn,
            affect_delta=relief,
            threat_delta=-0.02,
            note="session: grounds (hypothesis-framed) + titrated relief",
        )
