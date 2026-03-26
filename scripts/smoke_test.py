"""CRISP end-to-end smoke test.

Runs the full pipeline for N iterations and generates a Markdown report
showing coach problems, player rollouts, discussions, rewards, and losses.

Usage:
    python scripts/smoke_test.py --config configs/smoke_test.yaml --output smoke_report.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, List

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CRISP smoke test")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output", type=str, default="smoke_report.md",
                        help="Output path for the Markdown report")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Config overrides as key=value",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        level=logging.INFO,
    )

    args = parse_args(argv)

    from crisp.config_loader import load_config
    from crisp.train import init_infra, parse_overrides
    from crisp.workflow.collector import StepCollector
    from crisp.workflow.main_loop import step
    from scripts.write_smoke_report import generate_report

    overrides = parse_overrides(args.override)
    config = load_config(args.config, overrides=overrides)
    n_iters = config.training.num_iterations

    logger.info("Smoke test: %d iterations, model=%s", n_iters, config.training.model_name)

    ctx = init_infra(config)
    collector = StepCollector()

    for i in range(n_iters):
        result = step(ctx, collector=collector)
        logger.info(
            "iter=%d problems=%d discussions=%d accuracy=%.3f "
            "alice_loss=%.4f bob_loss=%.4f coach_loss=%s",
            i, result.num_problems, result.num_discussions,
            result.player_accuracy, result.alice_loss, result.bob_loss,
            f"{result.coach_loss:.4f}" if result.coach_loss is not None else "N/A",
        )

    report = generate_report(collector, config)

    with open(args.output, "w") as f:
        f.write(report)

    logger.info("Report written to %s", args.output)


if __name__ == "__main__":
    main()
