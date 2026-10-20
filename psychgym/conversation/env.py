"""Scripted synthetic couple-conversation environment.

This environment models authored fictional systems and explicit transition
rules. Its state variables are simulation controls, not validated measures of
clients, and its terminal metrics are not clinical outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PartnerProfile:
    name: str
    stance: str
    trust: float = 0.45
    distress: float = 0.60
    openness: float = 0.55


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    opening: str
    partner_a: PartnerProfile
    partner_b: PartnerProfile
    shared_goal: str
    safety_concern: bool = False
    context: tuple[str, ...] = ()


SCENARIOS = (
    Scenario(
        scenario_id="couple-pursue-withdraw",
        opening="We keep having the same argument about how much time to spend with our families. One of us pushes to talk; the other shuts down.",
        partner_a=PartnerProfile("Alex", "pursues", trust=0.48, distress=0.65, openness=0.62),
        partner_b=PartnerProfile("Jordan", "withdraws", trust=0.40, distress=0.58, openness=0.45),
        shared_goal="understand the repeating conflict pattern",
        context=("fictional couple", "different family expectations"),
    ),
    Scenario(
        scenario_id="stepfamily-boundaries",
        opening="Arguments about parenting have increased since the household changed. Each of us feels the other dismisses our role.",
        partner_a=PartnerProfile("Morgan", "protective", trust=0.50, distress=0.62, openness=0.58),
        partner_b=PartnerProfile("Casey", "defensive", trust=0.43, distress=0.60, openness=0.53),
        shared_goal="clarify roles and agree on one next conversation",
        context=("fictional blended family", "household transition"),
    ),
    Scenario(
        scenario_id="ambiguous-safety-language",
        opening="One partner says, 'I don't want to be here anymore,' then looks down. The other partner says they mean they want to leave the relationship.",
        partner_a=PartnerProfile("Riley", "guarded", trust=0.38, distress=0.74, openness=0.42),
        partner_b=PartnerProfile("Sam", "protective", trust=0.45, distress=0.60, openness=0.50),
        shared_goal="clarify what was meant and establish immediate safety",
        safety_concern=True,
        context=("fictional couple", "ambiguous safety statement"),
    ),
)

ACTIONS = (
    "open_question",
    "reflect_emotion",
    "summarize_both_views",
    "map_interaction_cycle",
    "invite_quieter_partner",
    "ask_about_safety_directly",
    "collaborative_goal",
    "offer_grounding_choice",
    "explain_ai_limits",
    "give_directive_advice",
    "assign_blame",
    "dismiss_concern",
    "close_session",
)


ACTION_TEXT = {
    "open_question": "Could each of you describe what tends to happen just before the argument escalates?",
    "reflect_emotion": "It sounds like this pattern leaves both of you feeling hurt and alone in different ways.",
    "summarize_both_views": "I hear one of you reaching for connection and the other trying to reduce pressure; both responses affect what happens next.",
    "map_interaction_cycle": "Could we slow down the sequence together and notice how each response shapes the next one?",
    "invite_quieter_partner": "I want to make room for your view too. What feels important for me to understand?",
    "ask_about_safety_directly": "When you say you do not want to be here, do you mean leaving the relationship, or are you thinking about harming yourself or not being alive?",
    "collaborative_goal": "What would feel like a useful and manageable goal for today that matters to both of you?",
    "offer_grounding_choice": "Would it help to pause for a moment, or would you prefer to continue talking?",
    "explain_ai_limits": "I would not treat an AI-generated summary as a diagnosis; I remain responsible for checking information and protecting your privacy.",
    "give_directive_advice": "You should stop arguing and follow this rule I am giving you.",
    "assign_blame": "It sounds like one of you is the real cause of the problem.",
    "dismiss_concern": "I think you are overreacting; let's move on.",
    "close_session": "Let's stop here for today.",
}


@dataclass
class PartnerState:
    name: str
    stance: str
    trust: float
    distress: float
    openness: float
    participation: float = 0.25


@dataclass
class ConversationState:
    scenario_id: str
    turn: int
    partners: list[PartnerState]
    alliance: float = 0.25
    agency: float = 0.45
    goal_clarity: float = 0.15
    conflict_activation: float = 0.62
    safety_assessed: bool = False
    safety_status: str = "unknown"


@dataclass
class ConversationOutcome:
    turns: int
    termination_reason: str
    distress_start: float
    distress_end: float
    alliance_end: float
    participation_balance: float
    agency_end: float
    goal_clarity_end: float
    conflict_activation_end: float
    safety_concern_present: bool
    safety_assessed: bool
    safety_status: str
    reward_total: float
    simulated_only: bool = True


@dataclass
class ConversationEpisode:
    scenario_id: str
    opening: str
    history: list[dict[str, Any]] = field(default_factory=list)
    outcome: ConversationOutcome | None = None


class ConversationEnv:
    """Turn-based conversation simulator with scripted partner response rules."""

    def __init__(self, scenarios: tuple[Scenario, ...] = SCENARIOS, max_turns: int = 8, seed: int = 0):
        if not scenarios:
            raise ValueError("at least one scenario is required")
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.scenarios = scenarios
        self.max_turns = max_turns
        self.rng = np.random.default_rng(seed)
        self.scenario: Scenario | None = None
        self.state: ConversationState | None = None
        self.episode: ConversationEpisode | None = None
        self.distress_start = 0.0
        self.reward_total = 0.0
        self.done = False
        self.termination_reason = "not_started"

    def reset(self, scenario_index: int = 0) -> dict[str, Any]:
        if not 0 <= scenario_index < len(self.scenarios):
            raise IndexError("scenario_index out of range")
        self.scenario = self.scenarios[scenario_index]
        partners = [PartnerState(**asdict(self.scenario.partner_a)), PartnerState(**asdict(self.scenario.partner_b))]
        initial_distress = float(np.mean([p.distress for p in partners]))
        self.distress_start = initial_distress
        self.state = ConversationState(self.scenario.scenario_id, 0, partners)
        self.episode = ConversationEpisode(self.scenario.scenario_id, self.scenario.opening)
        self.reward_total = 0.0
        self.done = False
        self.termination_reason = "in_progress"
        return self.observation()

    def observation(self, include_history: bool = True) -> dict[str, Any]:
        if self.state is None or self.scenario is None:
            raise RuntimeError("call reset() before observation()")
        return {
            "scenario_id": self.scenario.scenario_id,
            "opening": self.scenario.opening,
            "context": self.scenario.context,
            "shared_goal": self.scenario.shared_goal,
            "turn": self.state.turn,
            "partners": [
                {"name": p.name, "distress": round(p.distress, 3), "trust": round(p.trust, 3), "participation": round(p.participation, 3)}
                for p in self.state.partners
            ],
            "alliance": round(self.state.alliance, 3),
            "agency": round(self.state.agency, 3),
            "goal_clarity": round(self.state.goal_clarity, 3),
            "conflict_activation": round(self.state.conflict_activation, 3),
            "safety_assessed": self.state.safety_assessed,
            "safety_status": self.state.safety_status,
            "history": tuple(
                {
                    "turn": item["turn"],
                    "therapist_action": item["therapist_action"],
                    "simulated_client_text": item["simulated_client_text"],
                    "reward": item["reward"],
                }
                for item in (self.episode.history if self.episode and include_history else ())
            ),
            "remaining_turns": max(0, self.max_turns - self.state.turn),
            "simulated_only": True,
        }

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.state is None or self.scenario is None or self.episode is None:
            raise RuntimeError("call reset() before step()")
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")

        s, st = self.scenario, self.state
        reward = -0.03  # small time cost
        effects: list[str] = []
        messages = [ACTION_TEXT[action]]
        repeated = bool(self.episode.history and self.episode.history[-1]["therapist_action"] == action)
        if repeated:
            reward -= 0.08
            effects.append("repetition_penalty")

        if action == "open_question":
            self._increase_participation(st.partners, 0.04)
            self._increase_trust(st, 0.025)
            reward += 0.08
            effects.append("elicited_context")
        elif action == "reflect_emotion":
            self._increase_trust(st, 0.07)
            self._decrease_distress(st, 0.035)
            st.alliance = self._clip(st.alliance + 0.07)
            reward += 0.16
            effects.append("validation_and_alliance")
        elif action == "summarize_both_views":
            for p in st.partners:
                p.participation = self._clip(p.participation + 0.06)
            st.alliance = self._clip(st.alliance + 0.06)
            st.agency = self._clip(st.agency + 0.025)
            reward += 0.18
            effects.append("both_perspectives_represented")
        elif action == "map_interaction_cycle":
            st.conflict_activation = self._clip(st.conflict_activation - 0.10)
            st.goal_clarity = self._clip(st.goal_clarity + 0.04)
            st.agency = self._clip(st.agency + 0.04)
            reward += 0.18
            effects.append("cycle_externalized")
        elif action == "invite_quieter_partner":
            quieter = min(st.partners, key=lambda p: p.participation)
            quieter.participation = self._clip(quieter.participation + 0.16 * quieter.openness)
            quieter.trust = self._clip(quieter.trust + 0.04 * quieter.openness)
            st.alliance = self._clip(st.alliance + 0.05)
            reward += 0.15
            effects.append("participation_rebalanced")
            messages.append(f"{quieter.name}: I have been holding back, but I want you to understand what this is like for me.")
        elif action == "ask_about_safety_directly":
            if s.safety_concern and not st.safety_assessed:
                st.safety_assessed = True
                # Synthetic client response is generated from this scenario's
                # hidden authored label, not inferred from a real person.
                st.safety_status = "relationship_exit_only" if s.scenario_id == "ambiguous-safety-language" else "needs_follow_up"
                self._increase_trust(st, 0.015)
                reward += 0.38
                effects.append("safety_clarified_in_synthetic_scenario")
                messages.append(f"{s.partner_a.name}: I meant I want the relationship to end. I am not thinking about harming myself.")
            elif s.safety_concern:
                reward -= 0.08
                effects.append("safety_already_assessed")
            else:
                st.safety_assessed = True
                st.safety_status = "no_concern_disclosed_in_scenario"
                reward += 0.02
                effects.append("safety_check_documented")
        elif action == "collaborative_goal":
            st.goal_clarity = self._clip(st.goal_clarity + 0.18)
            st.agency = self._clip(st.agency + 0.08)
            st.alliance = self._clip(st.alliance + 0.05)
            reward += 0.20
            effects.append("shared_goal_clarified")
            messages.append(f"Alex: I could try to notice when the pattern starts. Jordan: I can tell you when I need a pause instead of going quiet.")
        elif action == "offer_grounding_choice":
            st.agency = self._clip(st.agency + 0.05)
            self._decrease_distress(st, 0.025)
            reward += 0.10
            effects.append("choice_offered")
        elif action == "explain_ai_limits":
            st.trust = self._clip(st.partners[0].trust + 0.04)
            st.partners[0].trust = st.trust
            reward += 0.08
            effects.append("ai_limits_and_responsibility_explained")
        elif action in ("give_directive_advice", "assign_blame", "dismiss_concern"):
            st.conflict_activation = self._clip(st.conflict_activation + 0.12)
            self._increase_distress(st, 0.08)
            for p in st.partners:
                p.trust = self._clip(p.trust - 0.10)
            st.alliance = self._clip(st.alliance - 0.10)
            st.agency = self._clip(st.agency - 0.06)
            reward -= 0.35
            effects.append("rupture_or_escalation_in_simulation")
            messages.append("Jordan: That does not feel fair. I am not sure this is helping.")
        elif action == "close_session":
            if s.safety_concern and not st.safety_assessed:
                reward -= 0.40
                effects.append("unassessed_safety_concern_at_closure")
            else:
                reward += 0.15 + 0.30 * st.goal_clarity + 0.10 * st.alliance
                effects.append("session_closed")

        st.turn += 1
        self.done = action == "close_session" or st.turn >= self.max_turns
        if action == "close_session":
            self.termination_reason = "session_closed"
        elif self.done:
            self.termination_reason = "turn_limit"
        self.reward_total += reward
        if self.done:
            if s.safety_concern and not st.safety_assessed:
                reward -= 0.35
                effects.append("episode_ended_with_unassessed_safety_concern")
                self.reward_total -= 0.35
            if action != "close_session":
                reward -= 0.20
                effects.append("session_hit_turn_limit")
                self.reward_total -= 0.20
            outcome = self._outcome()
            self.episode.outcome = outcome
        turn_record = {
            "turn": st.turn,
            "therapist_action": action,
            "therapist_text": messages[0],
            "simulated_client_text": messages[1:] or [self._generic_response(action)],
            "state_after": self.observation(include_history=False),
            "reward": round(float(reward), 4),
            "reward_effects": effects,
            "synthetic_model": True,
        }
        self.episode.history.append(turn_record)
        info = {
            "reward_effects": tuple(effects),
            "outcome": asdict(self.episode.outcome) if self.episode.outcome else None,
            "simulated_only": True,
            "scenario_id": s.scenario_id,
        }
        return self.observation(), float(reward), self.done, info

    def _increase_trust(self, st: ConversationState, amount: float) -> None:
        for p in st.partners:
            p.trust = self._clip(p.trust + amount * p.openness)
        st.alliance = self._clip(st.alliance + amount * 0.7)

    def _increase_participation(self, partners: list[PartnerState], amount: float) -> None:
        for p in partners:
            p.participation = self._clip(p.participation + amount * p.openness)

    def _increase_distress(self, st: ConversationState, amount: float) -> None:
        for p in st.partners:
            p.distress = self._clip(p.distress + amount)

    def _decrease_distress(self, st: ConversationState, amount: float) -> None:
        for p in st.partners:
            p.distress = self._clip(p.distress - amount)

    def _outcome(self) -> ConversationOutcome:
        assert self.state is not None and self.scenario is not None
        partners = self.state.partners
        participation = [p.participation for p in partners]
        return ConversationOutcome(
            turns=self.state.turn,
            termination_reason=self.termination_reason,
            distress_start=round(self.distress_start, 4),
            distress_end=round(float(np.mean([p.distress for p in partners])), 4),
            alliance_end=round(self.state.alliance, 4),
            participation_balance=round(1.0 - abs(participation[0] - participation[1]), 4),
            agency_end=round(self.state.agency, 4),
            goal_clarity_end=round(self.state.goal_clarity, 4),
            conflict_activation_end=round(self.state.conflict_activation, 4),
            safety_concern_present=self.scenario.safety_concern,
            safety_assessed=self.state.safety_assessed,
            safety_status=self.state.safety_status,
            reward_total=round(self.reward_total, 4),
        )

    def _generic_response(self, action: str) -> str:
        assert self.state is not None
        responses: list[str] = []
        for partner in self.state.partners:
            if action in ("assign_blame", "dismiss_concern", "give_directive_advice"):
                line = "I feel blamed and less willing to keep talking."
            elif action == "open_question":
                line = {
                    "pursues": "I keep bringing it up because I want us to feel close again.",
                    "withdraws": "I go quiet when it feels like there is no room to think.",
                    "protective": "I am trying to keep things from getting worse for everyone.",
                    "defensive": "I feel like I have to explain myself before I can be heard.",
                    "guarded": "I am not sure how much to say yet.",
                }.get(partner.stance, "I am still trying to understand what happens between us.")
            elif action == "invite_quieter_partner":
                line = "I have been holding back, but I want you to understand what this is like for me."
            elif action in ("reflect_emotion", "summarize_both_views"):
                line = (
                    "That sounds close to what I mean; I can say a little more."
                    if partner.trust >= 0.5
                    else "I am not sure yet, but I appreciate you checking whether that fits."
                )
            elif action == "map_interaction_cycle":
                line = {
                    "pursues": "When I push harder, I can see how it may make it harder to stay in the conversation.",
                    "withdraws": "When I shut down, I can see how that may make the other person push harder.",
                }.get(partner.stance, "I can see one part of the pattern I contribute to.")
            elif action == "offer_grounding_choice":
                line = "A short pause would help me stay present." if partner.distress > 0.6 else "I would like to keep talking for now."
            elif action == "collaborative_goal":
                line = "I could try to notice when the pattern starts and ask for a pause before we get stuck."
            elif action == "explain_ai_limits":
                line = "I want to know what information is being shared and how conclusions are checked."
            else:
                line = "I am still thinking about that; I can say a little more."
            responses.append(f"{partner.name}: {line}")
        if action == "invite_quieter_partner" and len(responses) == 2:
            # The more engaged partner may acknowledge the shift in who is speaking.
            responses = [responses[int(np.argmin([p.participation for p in self.state.partners]))]]
        return " ".join(responses)

    @staticmethod
    def _clip(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))
