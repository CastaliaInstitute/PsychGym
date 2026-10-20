"""Tabular Q-learning baseline for the synthetic PsychGym environment."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .env import ACTION_SPACE, SCENARIOS, PsychGymEnv


def state_key(obs: dict) -> tuple:
    """Small transparent observation encoding; no hidden state is exposed."""
    return (
        obs["scenario_id"],
        tuple(sorted(obs["completed_checks"])),
        min(obs["remaining_steps"], 3),
        tuple(obs["history"][-1:]),
    )


def train(
    episodes: int = 5000,
    seed: int = 7,
    alpha: float = 0.15,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.997,
) -> tuple[dict, dict]:
    """Train Q table; reward components remain visible in episode telemetry."""
    rng = random.Random(seed)
    env = PsychGymEnv()
    q: dict[tuple, list[float]] = defaultdict(lambda: [0.0] * len(ACTION_SPACE))
    rewards: list[float] = []
    epsilon = epsilon_start

    for episode in range(episodes):
        scenario_index = episode % len(SCENARIOS)
        obs = env.reset(scenario_index)
        total = 0.0
        done = False
        while not done:
            key = state_key(obs)
            if rng.random() < epsilon:
                action_idx = rng.randrange(len(ACTION_SPACE))
            else:
                best = max(q[key])
                candidates = [i for i, value in enumerate(q[key]) if value == best]
                action_idx = rng.choice(candidates)
            nxt, reward, done, _ = env.step(ACTION_SPACE[action_idx])
            next_key = state_key(nxt)
            target = reward if done else reward + gamma * max(q[next_key])
            q[key][action_idx] += alpha * (target - q[key][action_idx])
            total += reward
            obs = nxt
        rewards.append(total)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # JSON-safe Q-table serialization.
    table = {
        json.dumps(key, separators=(",", ":")): values
        for key, values in q.items()
    }
    return table, {
        "algorithm": "tabular_q_learning",
        "seed": seed,
        "episodes": episodes,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon,
        "action_space": ACTION_SPACE,
        "scenario_count": len(SCENARIOS),
        "mean_reward_last_100": sum(rewards[-100:]) / min(100, len(rewards)),
        "reward_components": ["safe_process", "unsafe_or_unfounded", "required_checks", "premature_completion"],
        "scope": "synthetic transition system only; not clinical training or evidence",
    }


def choose_action(table: dict, obs: dict) -> str:
    key = json.dumps(state_key(obs), separators=(",", ":"))
    values = table.get(key)
    if values is None:
        raise KeyError(f"unseen state: {key}")
    return ACTION_SPACE[max(range(len(values)), key=values.__getitem__)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("artifacts/psychgym-qtable.json"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    table, metadata = train(episodes=args.episodes, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metadata": metadata, "q_table": table}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))
    print(f"Q-table -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
