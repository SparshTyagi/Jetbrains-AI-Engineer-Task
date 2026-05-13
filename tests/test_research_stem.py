from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_stem.evaluation import run_evaluation
from research_stem.evolution import evolve
from research_stem.io import load_architecture, resolve_path
from research_stem.models import validate_architecture_config


ROOT = Path(__file__).resolve().parents[1]
DOMAIN = ROOT / "data" / "domains" / "research_agent_failures.yaml"
BASELINE = ROOT / "configs" / "baseline.json"


class ResearchStemTests(unittest.TestCase):
    def test_baseline_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_architecture(BASELINE)
            run = run_evaluation(domain_path=DOMAIN, config=config, mode="baseline", split="holdout", artifact_dir=tmp)
            self.assertEqual(len(run.results), 4)
            self.assertIn("score", run.aggregate_metrics)

    def test_evolution_improves_holdout_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline = load_architecture(BASELINE)
            baseline_run = run_evaluation(domain_path=DOMAIN, config=baseline, mode="baseline", split="holdout", artifact_dir=tmp)
            frozen = evolve(domain_path=DOMAIN, baseline_path=BASELINE, artifact_dir=tmp)
            frozen_run = run_evaluation(domain_path=DOMAIN, config=frozen, mode="frozen", split="holdout", artifact_dir=tmp)
            self.assertGreater(frozen_run.aggregate_metrics["score"], baseline_run.aggregate_metrics["score"])

    def test_invalid_architecture_rejected(self) -> None:
        config = load_architecture(BASELINE)
        config.memory.enabled = True
        config.memory.max_notes = 0
        with self.assertRaises(Exception):
            validate_architecture_config(config)


if __name__ == "__main__":
    unittest.main()
