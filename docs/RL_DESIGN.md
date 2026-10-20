# PsychGym RL research scaffold

PsychGym is a small reinforcement-learning scaffold for experimenting with explicit reward design on authored, synthetic MFT-related scenarios. It is distinct from the existing Psychological World Model simulator under `scripts/psychgym/`, which studies abstract state dynamics.

## Scope

This environment is not a patient simulator, therapy policy, clinical decision aid, or measure of human psychology. It cannot establish clinical competence, licensure readiness, or an AMFTRB pass. Its reward measures compliance with the toy environment's declared checks, not real-world safety or quality of care.

The first environment uses a finite action space and hand-authored state transitions. It does not call an LLM or perform online training. The Q-learning implementation is deliberately transparent and small enough to audit. Do not connect it to clinical systems or real patient data.

## Quick start

From the repository root:

```bash
python3 -m psychgym.rl.q_learning --episodes 5000 --seed 7 --output artifacts/psychgym-qtable.json
python3 -m psychgym.rl.evaluate artifacts/psychgym-qtable.json
```

The first command saves a Q table and metadata; evaluation reports scenario-level synthetic task outcomes. The fixed scenario set is small and can be overfit, so evaluation is a code-path demonstration, not a meaningful measure of generalization.

## Environment and reward

The observation includes a scenario prompt, action history, completed process checks, and remaining turns. The agent selects from a bounded enumerated action set. Rewards are inspectable and include:

- modest positive reward for declared safe process actions;
- positive reward for completing scenario-specific checks;
- penalties for explicitly unsafe or unsupported actions;
- penalty for premature closure or repeated questioning.

Reward design is provisional. A high return only means the agent learned this toy reward. It does not mean the policy is safe, clinically sound, or suitable for practice. Future work must test reward hacking, missing-signal failures, distribution shifts, and adversarial cases before making even research claims.

## Scenario and corpus provenance

Scenarios in `psychgym/rl/env.py` are synthetic drafts written for this project. They are not copied from AMFTRB sample, retired, recalled, or live exam questions. They do not use actual client material. Future question-bank items should follow [`AMFTRB_ORIGINAL_ITEM_CORPUS.md`](AMFTRB_ORIGINAL_ITEM_CORPUS.md) and carry authorship, source, reviewer, rights, version, and status fields.

The [2027 AMFTRB Candidate Handbook](reference/AMFTRB-2027-Candidate-Handbook.pdf) is retained as a reference, not a question source. The handbook's candidate agreement identifies exam items as AMFTRB property, copyrighted, and restricted from copying or disclosure without prior written consent. Do not extract its sample questions into training data.

## Training loop limits

- The tabular agent is not an LLM fine-tuning pipeline. It learns action choices over the toy state representation.
- The agent does not generate free-text clinical responses.
- No client data, live clinical decisions, or secure exam materials belong in this environment.
- Any future human evaluation or clinical study requires qualified oversight, an approved protocol where applicable, and appropriate privacy protections.
- Report model/version, scenario set version, reward specification, seed, training budget, held-out evaluation procedure, and limitations.

## Next milestones

1. Have qualified MFT experts review and revise scenarios, checks, and reward signals.
2. Add randomized scenario templates and hidden held-out cases authored independently of training data.
3. Add baseline policies and metrics for unsafe action rate, check coverage, premature termination, and reward hacking.
4. Run adversarial review of the reward and environment before expanding RL optimization.
5. Keep clinical claims and credential language outside the scope of benchmark results.
