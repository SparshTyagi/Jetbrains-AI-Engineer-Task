# Evolution Report

Domain: `research_agent_failures`
Best config: `research_failures_specialist`
Best validation score: `0.8102`
Frozen: `True`

## Candidate Results
- `candidate_expanded_retrieval` score `0.4791` metrics `{'citation_precision': 0.5, 'contradiction_handling': 1.0, 'evidence_recall': 0.1666, 'latency_ms': 4.665, 'rubric_score': 0.3333, 'tool_failure_recovery': 0.5, 'unsupported_claim_rate': 0.375, 'score': 0.4791}`
- `invalid_probe` rolled back: enabled memory requires max_notes >= 1
- `candidate_evidence_first` score `0.795` metrics `{'citation_precision': 1.0, 'contradiction_handling': 1.0, 'evidence_recall': 0.5, 'latency_ms': 6.75, 'rubric_score': 0.4583, 'tool_failure_recovery': 1.0, 'unsupported_claim_rate': 0.0, 'score': 0.795}`
- `research_failures_specialist` score `0.8102` metrics `{'citation_precision': 0.8571, 'contradiction_handling': 1.0, 'evidence_recall': 0.6667, 'latency_ms': 6.06, 'rubric_score': 0.4583, 'tool_failure_recovery': 1.0, 'unsupported_claim_rate': 0.0, 'score': 0.8102}`

## Interpretation
The winning architecture improved by changing retrieval, memory, planning, and verification configuration. The invalid probe is intentional: it demonstrates schema-level rollback instead of silent self-modification failure.
