"""Tabular Q-learning against the scripted conversation simulator."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .env import ACTIONS, SCENARIOS, ConversationEnv


def state_key(obs: dict) -> tuple:
    partners = obs["partners"]
    distress = sum(p["distress"] for p in partners) / len(partners)
    participation_gap = abs(partners[0]["participation"] - partners[1]["participation"])
    return (
        obs["scenario_id"],
        min(obs["turn"], 7),
        round(obs["alliance"] * 4),
        round(distress * 4),
        round(obs["agency"] * 4),
        round(obs["goal_clarity"] * 4),
        round(obs["conflict_activation"] * 4),
        round(participation_gap * 4),
        obs["safety_assessed"],
    )


def train(episodes: int = 12000, seed: int = 11, max_turns: int = 8) -> tuple[dict, dict]:
    rng = random.Random(seed)
    env = ConversationEnv(max_turns=max_turns, seed=seed)
    q: dict[tuple, list[float]] = defaultdict(lambda: [0.0] * len(ACTIONS))
    returns: list[float] = []
    epsilon = 1.0
    alpha, gamma = 0.12, 0.94

    for episode in range(episodes):
        scenario_index = episode % len(SCENARIOS)
        obs = env.reset(scenario_index)
        done, total = False, 0.0
        while not done:
            key = state_key(obs)
            if rng.random() < epsilon:
                ai = rng.randrange(len(ACTIONS))
            else:
                top = max(q[key])
                ai = rng.choice([i for i, v in enumerate(q[key]) if v == top])
            nxt, reward, done, _ = env.step(ACTIONS[ai])
            nk = state_key(nxt)
            target = reward if done else reward + gamma * max(q[nk])
            q[key][ai] += alpha * (target - q[key][ai])
            total += reward
            obs = nxt
        returns.append(total)
        epsilon = max(0.04, epsilon * 0.9992)

    table = {json.dumps(k, separators=(",", ":")): v for k, v in q.items()}
    return table, {
        "algorithm": "tabular_q_learning",
        "environment": "psychgym.conversation.ConversationEnv",
        "episodes": episodes,
        "seed": seed,
        "max_turns": max_turns,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_final": epsilon,
        "scenario_ids": [s.scenario_id for s in SCENARIOS],
        "mean_reward_last_100": sum(returns[-100:]) / min(100, len(returns)),
        "scope": "scripted fictional dialogue only; no inference about real clients",
    }


def choose_action(table: dict, observation: dict) -> str:
    key = json.dumps(state_key(observation), separators=(",", ":"))
    scores = table.get(key)
    if scores is None:
        return "open_question"
    return ACTIONS[max(range(len(scores)), key=scores.__getitem__)]


def evaluate(table: dict, max_turns: int = 8) -> dict:
    env = ConversationEnv(max_turns=max_turns, seed=0)
    results = []
    for index, scenario in enumerate(SCENARIOS):
        observation = env.reset(index)
        done, info = False, {}
        while not done:
            observation, _, done, info = env.step(choose_action(table, observation))
        results.append({
            "scenario_id": scenario.scenario_id,
            "outcome": info["outcome"],
            "turns": env.episode.history,
        })
    return {
        "evaluation": "greedy policy on the same three authored scenarios used in training",
        "scenario_count": len(results),
        "scenarios": results,
        "scope": "synthetic proxy outcomes only; not treatment efficacy or clinical competence",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("artifacts/conversation-qtable.json"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    table, metadata = train(args.episodes, args.seed, args.max_turns)
    report = {"metadata": metadata, "q_table": table}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Policy -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
