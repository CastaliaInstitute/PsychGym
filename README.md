# PsychGym

PsychGym is a research sandbox for training and evaluating AI behavior with reinforcement learning in authored synthetic scenarios. It contains two separate components:

- `psychgym.rl`: a small tabular Q-learning scaffold and synthetic MFT-related process scenarios.
- `psychgym.pwm`: the existing Psychological World Model (PWM) simulator for abstract state-dynamics experiments.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m psychgym.rl.q_learning --episodes 12000 --seed 7 --output artifacts/psychgym-qtable.json
python -m psychgym.rl.evaluate artifacts/psychgym-qtable.json
```

The RL baseline learns a finite action policy over a tiny state representation. It is intended to demonstrate reward design and RL mechanics; it does not fine-tune an LLM or produce free-text therapy responses. See [`docs/RL_DESIGN.md`](docs/RL_DESIGN.md).

Run the existing PWM simulator:

```bash
python -m psychgym.pwm.run_all
```

## Scope

This project is not a patient simulator, therapy policy, clinical decision aid, or measure of human psychology. Toy task rewards cannot establish clinical competence, licensure readiness, or an AMFTRB pass. Do not connect it to clinical systems, real patient data, live clinical decisions, or secure exam materials.

Scenarios are synthetic project drafts and require qualified MFT review before any meaningful evaluation claims. Future items should include provenance, citations, authorship, review metadata, versioning, and rights. Keep held-out evaluation items separate from training data.
