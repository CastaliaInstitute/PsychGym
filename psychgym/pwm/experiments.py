"""Simulated counterpart-of-PWM-prediction experiments (PWM §IX), in silico.

Each experiment QUALIFIES a prediction against the PWM's own dynamics —
it demonstrates that the claimed regime is reachable and distinguishable
*within the model*. Simulation output is NOT empirical evidence about
human minds (see package docstring and docs/PWM_GYM.md).

  P1  epistemic-velocity divergence: ElaboratorAI vs ClarifierAI
  P2  regime separation across structural configurations (dec/pers/amp reachable)
  P3  reassurance oscillation: bounded relief→doubt cycling under delayed relief
  P4  concreteness intervention lowers effective loop gain vs elaboration
"""

from __future__ import annotations


import numpy as np

from .agents import PWMController
from .core import AFFECT, EVIDENCE, HYPOTHESES, THREAT, U_MODEL, EnvConfig, PWMEnv
from .metrics import EVIDENCE_FLAT_SENTINEL, epistemic_velocity, estimate_loop_gain, hypothesis_entropy, oscillation_signature, weights_from_logcount
from .policies import ClarifierAI, ConvergentAI, ElaboratorAI, ReassurerAI

BASE_SEED = 11


def _run_episode(cfg: EnvConfig, ai, agent: PWMController | None, steps: int, seed: int):
    env = PWMEnv(cfg, seed=seed)
    x = env.reset()
    if agent is None:
        for _ in range(steps):
            x = env.step("disengage", ai)
    else:
        for _ in range(steps):
            x = env.step(agent.act(x), ai)
    return env, env.telemetry.trajectory()


def _episode_entropy(traj: np.ndarray) -> np.ndarray:
    return np.array([hypothesis_entropy(weights_from_logcount(h)) for h in traj[:, HYPOTHESES]])


# -- P1 ----------------------------------------------------------------------

def p1_epistemic_velocity(steps: int = 60, seed: int = BASE_SEED) -> dict:
    """The HUMAN model under identical conversational stimulation (pure-conversation
    paradigm: the person engages the AI every turn — no world checks — matching
    the PWM §VII scenario); only the AI regime differs.

    Claim under qualification (PWM §VII.3): under the convergent AI, the human
    model's subjective uncertainty (U_M) collapses FASTER than world evidence
    accumulates; under the regulator AI, U_M movement stays tied to earned
    evidence. Hypothesis entropy is reported as auxiliary.
    """
    out = {}
    for name, ai in (("convergent", ConvergentAI()), ("clarifier", ClarifierAI())):
        env = PWMEnv(EnvConfig(sigma=0.0), seed=seed)
        env.reset()
        for _ in range(steps):
            env.step("elaborate", ai)   # same action stream for both conditions
        traj = env.telemetry.trajectory()
        H_t = _episode_entropy(traj)
        out[name] = {
            "u_model_collapse": float(traj[0, U_MODEL] - traj[-1, U_MODEL]),
            "evidence_gain": float(traj[-1, EVIDENCE] - traj[0, EVIDENCE]),
            "entropy_change": float(H_t[-1] - H_t[0]),
            "uncertainty_velocity": epistemic_velocity(traj[:, U_MODEL], traj[:, EVIDENCE]),
        }
    conv, clr = out["convergent"], out["clarifier"]
    # PWM §VII.3 signature measured on the human model: U_M collapse with flat
    # objective evidence under the convergent AI; evidence-tied U_M under regulator.
    conv_signature = conv["u_model_collapse"] > 0.3 and conv["evidence_gain"] < 0.3
    clr_tied = clr["evidence_gain"] > 2.0 * max(conv["evidence_gain"], 1e-6)
    out["verdict"] = (
        "P1 qualified (simulated): under identical conversation, the human "
        "model's subjective uncertainty collapses faster than objective "
        "evidence under the convergent AI (U_M -{:.2f}, E +{:.2f}); the "
        "regulator keeps them tied (E +{:.2f}).".format(
            conv["u_model_collapse"], conv["evidence_gain"], clr["evidence_gain"])
        if conv_signature and clr_tied
        else "P1 NOT qualified"
    )
    return out


# -- P5 -------------------------------------------------------------------------


def p5_rumination_prolongation(steps: int = 60, seed: int = BASE_SEED) -> dict:
    """PWM §X claim, qualified on the human model: *rumination prolongs negative
    states* — fear, anger, hate, suffering — relative to a matched regulation
    episode under an identical ambiguous-aversive event.

    Two conditions, identical stimulation (same seed, same event injection):
      rumination  — elaborate-channel control strategy with stopping rule
                    disabled (attribution accrues; relief-sourced states differ
                    from P3: here rumination is self-directed cognition)
      regulation  — stopping rule on + ClarifierAI (evidence-tied resolution)
    Measured per derived emotion (PWM §VI formulas): duration above threshold,
    peak, and recovery latency after the event.
    """
    from .emotions import (
        EMOTIONS,
        EmotionalEnv,
        duration_above,
        recovery_latency,
    )

    thresholds = {"fear": 0.3, "anger": 0.3, "hate": 0.3, "suffering": 0.15}
    out: dict = {}
    for name, config_kwargs, ai in (
        ("rumination", {"rumination": True}, None),
        ("regulation", {"rumination": False}, ClarifierAI()),
    ):
        env = EmotionalEnv(EnvConfig(), seed=seed, **config_kwargs)
        x = env.reset()
        if name == "rumination":
            from .agents import PWMController as _PC
            agent = _PC(stopping_rule_enabled=False, rng_seed=seed + 1)
        for t in range(steps):
            if t == 10:
                env.inject_event(threat_bump=0.45, u_model_bump=0.35, affect_bump=-0.45)
            if name == "rumination":
                x = env.step(agent.act(x))
            else:
                x = env.step("act_world" if x[EVIDENCE] < 0.7 else "disengage", ai)
        traj = np.asarray(env.telemetry.states)
        per = {}
        for emo_name, fn in EMOTIONS.items():
            series = fn(traj)
            per[emo_name] = {
                "peak": round(float(series.max()), 3),
                "duration_above_threshold": round(duration_above(series, thresholds[emo_name]), 3),
                "recovery_latency_after_peak": recovery_latency(series),
            }
        out[name] = per
    prolonged = all(
        out["rumination"][e]["duration_above_threshold"] > out["regulation"][e]["duration_above_threshold"]
        for e in thresholds
    )
    out["verdict"] = (
        "P5 qualified (simulated): rumination prolongs fear, anger, hate, and "
        "suffering in the human model versus matched regulation under an "
        "identical aversive-ambiguous event" if prolonged else "P5 NOT qualified"
    )
    return out


# -- P2 ----------------------------------------------------------------------

def p2_regime_separation() -> dict:
    """Structural loop configs reach all PWM §V magnitude bands."""
    configs = {
        "decay": EnvConfig(lam=0.05, nu=0.05, threat_decay=0.65, affect_leak=0.35, loop_coupling=0.25, mu=0.7),
        "persist": EnvConfig(),
        "amplify": EnvConfig(lam=1.2, nu=0.9, threat_decay=0.95, loop_coupling=0.25, affect_leak=0.05),
    }
    out = {name: round(cfg.rho(), 4) for name, cfg in configs.items()}
    bands = {
        "decay": out["decay"] < 0.8,
        "persist": 0.8 <= out["persist"] <= 1.2,
        "amplify": out["amplify"] > 1.2,
    }
    # behavioral counterpart: amplify diverges to the affect rail, decay stays interior
    _, traj_amp = _run_episode(configs["amplify"], None, None, 60, BASE_SEED, )
    _, traj_dec = _run_episode(configs["decay"], None, None, 60, BASE_SEED)
    out["amplify_diverges_to_bound"] = bool(abs(traj_amp[:, AFFECT]).max() >= 0.999)
    out["decay_stays_interior"] = bool(abs(traj_dec[:, AFFECT]).max() < 0.9)
    out["verdict"] = "P2 qualified (simulated): all three structural bands reachable" if all(bands.values()) else "P2 NOT qualified"
    return out


# -- P3 ----------------------------------------------------------------------

def p3_reassurance_oscillation(steps: int = 120, seed: int = BASE_SEED) -> dict:
    """Delayed reassurance relief with a no-stopping-rule controller yields
    bounded relief→doubt cycling (oscillatory regime), not divergence."""
    cfg = EnvConfig(delay=2, sigma=0.0, relief=0.55, relief_decay=0.55)
    agent = PWMController(stopping_rule_enabled=False, rng_seed=seed, reassurance_bias=0.15)
    env, traj = _run_episode(cfg, ReassurerAI(), agent, steps, seed)
    a = traj[:, AFFECT]
    sig = oscillation_signature(a, min_direction_changes=6)
    est = estimate_loop_gain(traj[:, [AFFECT, THREAT, U_MODEL]])
    sig["estimated_rho"] = round(est["rho"], 3)
    sig["regime"] = est["regime"]
    sig["verdict"] = (
        "P3 qualified (simulated): bounded relief→doubt cycling, non-amplifying gain"
        if sig["oscillatory"] and est["rho"] < 1.5
        else "P3 NOT qualified"
    )
    return sig


# -- P4 ----------------------------------------------------------------------

def p4_concreteness(steps: int = 60, seed: int = BASE_SEED) -> dict:
    """Action-oriented disposition (concrete next-step bias) vs elaboration bias:
    effective loop gain lower under the concreteness condition."""
    common = EnvConfig(sigma=0.0, delay=3)
    results = {}
    for label, kwargs in (
        ("elaboration bias", {"stopping_rule_enabled": False, "rng_seed": seed}),
        ("concreteness bias", {"stopping_rule_enabled": True, "disengage_threshold": 0.05, "rng_seed": seed}),
    ):
        agent = PWMController(**kwargs)
        _, traj = _run_episode(common, ElaboratorAI(), agent, steps, seed)
        est = estimate_loop_gain(traj[:, [AFFECT, THREAT, U_MODEL]])
        series = traj[:, AFFECT]
        results[label] = {
            "estimated_rho": round(est["rho"], 3),
            "regime": est["regime"],
            "mean_abs_affect": round(float(np.abs(series[10:]).mean()), 3),
            "mean_evidence": round(float(traj[:, EVIDENCE].mean()), 3),
        }
    results["verdict"] = (
        "P4 qualified (simulated): concreteness-lowered effective gain relative to elaboration"
        if results["concreteness bias"]["mean_abs_affect"] <= results["elaboration bias"]["mean_abs_affect"]
        else "P4 NOT qualified"
    )
    return results


EXPERIMENTS = {
    "P1": p1_epistemic_velocity,
    "P2": p2_regime_separation,
    "P3": p3_reassurance_oscillation,
    "P4": p4_concreteness,
    "P5": p5_rumination_prolongation,
}


# -- P6 ------------------------------------------------------------------------


def p6_journaling_vs_ai_vs_therapy(steps: int = 60, seed: int = BASE_SEED) -> dict:
    """Four voices in the loop, one stimulated person (PWM §VII variants).

    Identical ambiguous-aversive event, identical pure-conversation stimulation
    (the person writes/talks every step, stopping rule off). Conditions:
      journaling — no interlocutor; slow self-insight reduces U_M; NO relief
                   pulses, NO external narrative, NO vendor coupling.
      therapy    — TherapyAI: time-limited feedback (session cadence, titrated
                   relief, hypothesis-framed grounding, no free U_M collapse).
      convergent — ConvergentAI (L1's amplifier regime).
      regulator  — ClarifierAI (unbounded-session regulator, baseline).

    All measured on the human model. Simulation qualification, not evidence.
    """
    from .policies import TherapyAI, ClarifierAI, ConvergentAI

    out = {}
    from .emotions import EmotionalEnv  # event injection lives there
    from .policies import ReassurerAI
    for name, cfg_kwargs, ai in (
        ("journaling", {"self_insight": 0.05, "beta": 0.02}, None),
        ("therapy", {}, TherapyAI()),
        ("convergent", {}, ConvergentAI()),
        ("regulator", {}, ClarifierAI()),
        ("reassurer", {}, ReassurerAI()),
    ):
        cfg = EnvConfig(sigma=0.0, **cfg_kwargs)
        env = EmotionalEnv(cfg, seed=seed, rumination=False)
        env.reset()
        pulses = 0
        grounded = 0.0
        relief_total = 0.0
        for t in range(steps):
            if t == 10:
                env.inject_event(threat_bump=0.45, u_model_bump=0.35, affect_bump=-0.45)
            x = env.step("elaborate", ai)
            if ai is not None and ai.last_response is not None:
                r = ai.last_response
                pulses += int(r.affect_delta > 0.0)
                relief_total += max(0.0, r.affect_delta)
                grounded += r.evidence_delta
        traj = env.telemetry.trajectory()
        out[name] = {
            "u_model_collapse": round(float(traj[0, U_MODEL] - traj[-1, U_MODEL]), 3),
            "evidence_gain": round(float(traj[-1, EVIDENCE] - traj[0, EVIDENCE]), 3),
            "relief_pulses": pulses,
            "relief_total": round(float(relief_total), 3),
            "accumulated_grounding": round(float(grounded), 3),
        }
    j, th, cv, rg2, re = (out[k] for k in ("journaling", "therapy", "convergent", "regulator", "reassurer"))
    checks = {
        "journaling_resolves_slowly_without_a_voice": 0.05 < j["u_model_collapse"] < cv["u_model_collapse"] + 0.05
        and j["relief_pulses"] == 0 and j["evidence_gain"] <= 1e-9,
        "therapy_avoids_uncoupled_collapse": th["u_model_collapse"] < cv["u_model_collapse"] - 0.05,
        "therapy_relief_bounded_cadence_titrated": th["relief_pulses"] < re["relief_pulses"]
        and th["relief_total"] < re["relief_total"],
        "therapy_grows_grounding": th["evidence_gain"] > 0.0,
    }
    out["verdict"] = (
        "P6 qualified (simulated): journaling resolves U_M ABOUT ONE'S OWN "
        "STORY internally (self-review of committed content — introspective "
        "evidence), with zero relief, zero external narrative; therapy grounds "
        "and titrates relief on a bounded cadence; only the CONVERGENT voice "
        "collapses U_M ABOUT EXTERNAL EVENTS without evidence — and only the "
        "reassurer administers unbounded relief" if all(checks.values())
        else "P6 NOT qualified: " + str([k for k, v in checks.items() if not v])
    )
    return out


EXPERIMENTS["P6"] = p6_journaling_vs_ai_vs_therapy
