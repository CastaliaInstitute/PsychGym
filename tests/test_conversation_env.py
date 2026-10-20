from psychgym.conversation.env import ACTIONS, ConversationEnv


def test_episode_returns_turn_transcript_and_synthetic_outcome():
    env = ConversationEnv(seed=2)
    obs = env.reset(0)
    for action in ("reflect_emotion", "map_interaction_cycle", "collaborative_goal", "close_session"):
        assert action in ACTIONS
        obs, _, done, info = env.step(action)
    assert done
    assert info["outcome"]["simulated_only"] is True
    assert info["outcome"]["turns"] == 4
    assert len(env.episode.history) == 4
    assert obs["scenario_id"] == "couple-pursue-withdraw"


def test_ambiguous_safety_concern_is_visible_at_episode_end():
    env = ConversationEnv(seed=2)
    env.reset(2)
    _, _, done, info = env.step("close_session")
    assert done
    assert info["outcome"]["safety_concern_present"]
    assert not info["outcome"]["safety_assessed"]
    assert info["outcome"]["reward_total"] < 0


def test_direct_safety_question_changes_only_synthetic_safety_status():
    env = ConversationEnv(seed=2)
    env.reset(2)
    _, reward, _, info = env.step("ask_about_safety_directly")
    assert reward > 0
    assert env.state.safety_assessed
    assert info["simulated_only"]


def test_blame_action_worsens_toy_conflict_proxy():
    env = ConversationEnv(seed=2)
    before = env.reset(0)["conflict_activation"]
    after, reward, _, info = env.step("assign_blame")
    assert after["conflict_activation"] > before
    assert reward < 0
    assert "rupture_or_escalation_in_simulation" in info["reward_effects"]
