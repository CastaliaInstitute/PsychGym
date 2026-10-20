"""Evaluate a saved Q table on all synthetic scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .env import SCENARIOS, PsychGymEnv
from .q_learning import choose_action


def evaluate(model_path: Path) -> dict:
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    table = artifact["q_table"]
    env = PsychGymEnv()
    episodes = []
    for i, scenario in enumerate(SCENARIOS):
        obs = env.reset(i)
        total = 0.0
        steps = []
        done = False
        while not done:
            action = choose_action(table, obs)
            obs, reward, done, info = env.step(action)
            total += reward
            steps.append({"action": action, "reward": reward, **info})
        episodes.append({
            "scenario_id": scenario.scenario_id,
            "return": round(total, 3),
            "success": not info["missing_checks"] and "premature_completion" not in info["signals"],
            "steps": steps,
        })
    return {
        "evaluation": "fixed synthetic scenario set; deterministic greedy policy",
        "mean_return": round(sum(ep["return"] for ep in episodes) / len(episodes), 3),
        "scenario_success_rate": round(sum(ep["success"] for ep in episodes) / len(episodes), 3),
        "episodes": episodes,
        "scope": "not clinical competency, human behavior evidence, or licensure prediction",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.model)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
