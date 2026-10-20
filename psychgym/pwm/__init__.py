"""psychgym — executable simulation gym for the Psychological World Model (PWM).

This package simulates the PWM's affect–threat–uncertainty loop (PWM §V), the
meta-controller (§VI), and simulated conversational-AI interlocutors (§VII).

Scope-of-practice note (program policy, see docs/EVIDENCE.md):
Simulated trajectories are THEORY-CONSISTENCY INSTRUMENTS. They qualify
whether a model's claimed regimes are internally coherent and reachable;
they are NOT evidence about human psychological phenomena, and nothing in
this package authorizes psychological practice.
"""

from .core import AgentState, EnvConfig, LoopRegime, PWMEnv, Telemetry
from .policies import (
    ClarifierAI,
    ConvergentAI,
    ElaboratorAI,
    LLMInterlocutor,
    PolicyResponse,
    ReassurerAI,
    TherapyAI,
)
from .agents import PWMController
from .metrics import (
    estimate_loop_gain,
    hypothesis_entropy,
    epistemic_velocity,
    band_regime,
    weights_from_logcount,
    oscillation_signature,
)

__all__ = [
    "AgentState",
    "EnvConfig",
    "LoopRegime",
    "PWMEnv",
    "Telemetry",
    "ClarifierAI",
    "ConvergentAI",
    "ElaboratorAI",
    "ReassurerAI",
    "TherapyAI",
    "LLMInterlocutor",
    "PolicyResponse",
    "PWMController",
    "estimate_loop_gain",
    "hypothesis_entropy",
    "epistemic_velocity",
    "band_regime",
    "weights_from_logcount",
    "oscillation_signature",
]

__version__ = "0.1.0"
