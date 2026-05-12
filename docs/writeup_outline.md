# Write-up Outline

## 1. Problem Framing

- Build a stem agent for one bounded domain: deep research over agent-system failure modes.
- The system adapts configuration, not code. This keeps the experiment reproducible and avoids vague autonomy claims.
- The domain was chosen because common agent failures are observable in small examples: context drift, noisy retrieval, broken tools, weak memory, and misleading evaluations.

## 2. Architecture

- State machine: Bootstrap -> WorkflowMining -> ConfigProposal -> Experiment -> Evaluation -> Freeze -> Execute.
- Typed contracts describe domain specs, tasks, tools, memory, retrieval, planning, configs, eval results, and experiment runs.
- Candidate configs are validated before execution. Invalid candidates are rolled back and logged.
- JSONL traces record tool calls, retries, config proposals, rollbacks, and scores.

## 3. Experiments

- Baseline: shallow retrieval, direct synthesis, no durable memory, no citation verifier, no retries.
- Candidates are derived from workflow examples.
- One invalid candidate is included to test rollback behavior.
- The best validation config is frozen and measured on holdout tasks.

## 4. Results

Use the table from `artifacts_api_final/evaluations/latest_baseline_holdout.json` and `artifacts_api_final/evaluations/latest_frozen_holdout.json`.

Focus on evidence recall, unsupported claim rate, contradiction handling, and tool failure recovery.

## 5. Failures and Surprises

- Fluent answers were worse than terse evidence-first answers under citation metrics.
- Increasing top-k alone pulled repeated evidence; source diversity mattered.
- Memory helped only after notes were source-grounded and bounded.
- API-mined workflow summaries initially added noisy retrieval terms, so they were normalized into compact lesson tags.
- Strict evidence-ID metrics can underrate useful synthesis with different wording.

## 6. Next Steps

- Larger corpus with hidden holdout tasks.
- Advisory LLM judge with disagreement analysis against deterministic metrics.
- Claim-to-source entailment checks for stronger citation verification.
- Repeated API runs to measure variance and cost.
