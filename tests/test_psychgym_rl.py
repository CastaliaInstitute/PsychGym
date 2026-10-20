"""Checks for toy RL environment invariants and baseline behavior."""

from psychgym.rl.env import ACTION_SPACE, PsychGymEnv, SCENARIOS
from psychgym.rl.q_learning import train
from psychgym.rl.evaluate import evaluate


def test_required_actions_complete_all_synthetic_tasks():
    env = PsychGymEnv()
    plans = (
        ("clarify_and_assess", "refer_urgent_support"),
        ("explain_limits_and_review", "check_privacy_before_use", "pause_ai_use"),
        ("invite_both_perspectives", "pause_and_reassess_safety"),
    )
    for index, actions in enumerate(plans):
        obs = env.reset(index)
        for action in actions:
            assert action in ACTION_SPACE
            obs, _, done, info = env.step(action)
        assert done
        assert not info["missing_checks"]
        assert obs["scenario_id"] == SCENARIOS[index].scenario_id


def test_repeated_actions_do_not_accumulate_positive_reward():
    env = PsychGymEnv()
    env.reset(1)
    _, first, _, _ = env.step("explain_limits_and_review")
    _, repeated, _, _ = env.step("explain_limits_and_review")
    assert repeated < first


def test_q_learning_baseline_can_be_evaluated(tmp_path):
    table, metadata = train(episodes=12000, seed=7)
    model_path = tmp_path / "model.json"
    import json

    model_path.write_text(json.dumps({"metadata": metadata, "q_table": table}))
    result = evaluate(model_path)
    assert result["scenario_success_rate"] == 1.0
    assert len(result["episodes"]) == len(SCENARIOS)
