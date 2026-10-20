# PsychGym RL and conversation research scaffold

## What the conversation environment simulates

`psychgym.conversation.ConversationEnv` provides multi-turn dialogues between an RL-controlled therapist-side agent and a scripted synthetic couple. Each scenario contains fictional partner profiles, an opening statement, context tags, a shared goal, and (in one case) a synthetic safety concern. The therapist chooses among bounded process actions. The simulator returns authored text templates, updates explicit state variables, calculates a reward, and records a turn-level trace.

Available proxy state includes each partner's modeled distress, trust, openness, and participation plus couple-level alliance, agency, goal clarity, conflict activation, and safety-assessment status. Terminal outcomes record those proxies and reward totals. Rules are inspectable in `psychgym/conversation/env.py`.

These values are invented transition variables, not validated psychological measures. The partner response rules do not model the range or causal complexity of people, relationships, therapy, or treatment outcomes. Lower simulated distress or higher alliance is not evidence of treatment benefit.

## Train and inspect a conversation policy

```bash
python -m psychgym.conversation.rl --episodes 12000 --seed 11 --output artifacts/conversation-qtable.json
```

In Python, load the serialized table and run the greedy policy through the fixed scenario set:

```python
import json
from psychgym.conversation.rl import evaluate

artifact = json.load(open("artifacts/conversation-qtable.json"))
report = evaluate(artifact["q_table"])
```

The result includes every simulated turn and terminal proxy values. The tabular baseline selects from a finite action set; therapist utterances are fixed templates. It is not LLM fine-tuning and does not generate natural clinical dialogue.

## Reward and transition assumptions

The reward favors selected process actions (reflection, balanced participation, collaborative goals, interaction-cycle mapping, and direct safety assessment in the authored safety scenario). It penalizes directive advice, blame assignment, and dismissing a concern. State transitions are hand-authored and deterministic. This intentionally exposes the reward model for inspection, but creates a substantial risk of overfitting and reward hacking.

Before using RL results to support research claims, test alternative reward formulations, randomize profile parameters, create independently authored held-out scenarios, stress-test missing and conflicting cues, compare against simple baselines, and obtain review from qualified MFT experts. Human or clinical outcome claims require separate validated measures and appropriately governed studies.

## Separate components

- `psychgym.conversation`: turn-based scripted dialogue and proxy outcomes.
- `psychgym.rl`: a simpler single-step/short-horizon process-choice toy.
- `psychgym.pwm`: abstract Psychological World Model state dynamics.

Do not conflate their simulations or claims.

## Scope and data rules

PsychGym is not a patient simulator, clinical decision aid, therapy policy, clinical training substitute, measure of human psychology, or evidence of treatment efficacy. It cannot establish competence, LMFT eligibility, licensure, or AMFTRB exam passage. Do not use real patient/client data or connect the simulator to live clinical systems.

Scenarios are synthetic drafts and require qualified MFT review before meaningful evaluation. Do not copy AMFTRB sample, retired, recalled, or live items. Public availability does not establish reuse rights. Keep held-out scenarios separate from training and document authorship, source, reviewer, version, and rights metadata.
