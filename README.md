# PsychGym

PsychGym is a research sandbox for training and evaluating AI behavior with reinforcement learning in authored synthetic scenarios. It contains three separate components:

- `psychgym.rl`: a small tabular Q-learning scaffold and synthetic MFT-related process scenarios.
- `psychgym.conversation`: a multi-turn synthetic couple-conversation simulator with scripted partner responses and explicit proxy outcomes.
- `psychgym.pwm`: the existing Psychological World Model (PWM) simulator for abstract state-dynamics experiments.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m psychgym.rl.q_learning --episodes 12000 --seed 7 --output artifacts/psychgym-qtable.json
python -m psychgym.rl.evaluate artifacts/psychgym-qtable.json
```

The first RL baseline learns a finite action policy over a small state representation. The conversation simulator produces a turn transcript, but both therapist and partner messages are selected from authored templates. It does not fine-tune an LLM or generate open-ended dialogue. See [`docs/RL_DESIGN.md`](docs/RL_DESIGN.md).

Train a policy against the conversation simulator:

```bash
python -m psychgym.conversation.rl --episodes 12000 --seed 11 --output artifacts/conversation-qtable.json
python -m psychgym.conversation.evaluate artifacts/conversation-qtable.json --output artifacts/conversation-eval.json
```

Run the existing PWM simulator:

```bash
python -m psychgym.pwm.run_all
```

## Scope

The conversation component is explicitly a scripted **synthetic conversation simulator**. Its state variables (distress, trust, alliance, agency, and conflict activation) are invented simulation controls, not validated measures, and its outcomes do not model treatment efficacy. This project is not a clinical training policy, clinical decision aid, or measure of human psychology. Toy task rewards cannot establish clinical competence, licensure readiness, or an AMFTRB pass. Do not connect it to clinical systems, real patient data, live clinical decisions, or secure exam materials.

Scenarios are synthetic project drafts and require qualified MFT review before any meaningful evaluation claims. Future items should include provenance, citations, authorship, review metadata, versioning, and rights. Keep held-out evaluation items separate from training data.
