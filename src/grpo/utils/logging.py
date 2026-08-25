"""Metric logging to console, JSONL, and optionally Weights & Biases.

JSONL is the primary sink and the console is the convenience view. An RL run's
interesting quantities are ratios between series that were logged at different
steps -- reward against KL, completion length against clip fraction -- and
scraping those back out of a text log is miserable. One JSON object per step
keeps the run analyzable after the fact with no extra service required.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

__all__ = ["MetricLogger", "format_metrics"]

# Ordered so the console line reads as a story: is reward moving, is the policy
# drifting, are completions getting longer, is anything being clipped.
_PRIORITY = (
    "reward/total",
    "reward/math_correctness",
    "reward/format",
    "loss",
    "kl",
    "clip_frac",
    "completion/mean_length",
    "completion/frac_truncated",
    "reward/frac_zero_variance_groups",
    "grad_norm",
    "lr",
)


def format_metrics(metrics: dict[str, float], step: int, total: int | None = None) -> str:
    """Render a compact one-line console summary."""
    head = f"step {step}" + (f"/{total}" if total else "")
    ordered = [k for k in _PRIORITY if k in metrics]
    ordered += sorted(k for k in metrics if k not in _PRIORITY)
    parts = []
    for key in ordered:
        value = metrics[key]
        if not isinstance(value, (int, float)):
            continue
        short = key.split("/")[-1] if key.count("/") else key
        parts.append(f"{short}={value:.4g}" if abs(value) < 1e4 else f"{short}={value:.3e}")
    return f"{head} | " + " ".join(parts)


class MetricLogger:
    """Fan-out logger. W&B is optional and never a hard dependency."""

    def __init__(
        self,
        output_dir: str | Path,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
        config: dict[str, Any] | None = None,
        total_steps: int | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.completions_path = self.output_dir / "completions.jsonl"
        self.total_steps = total_steps
        self._start = time.time()
        self._wandb = None

        if config is not None:
            (self.output_dir / "config.json").write_text(
                json.dumps(config, indent=2, default=str), encoding="utf-8"
            )

        if wandb_project:
            try:
                import wandb

                self._wandb = wandb
                wandb.init(project=wandb_project, name=wandb_run_name, config=config)
            except ImportError:
                print("[grpo] wandb not installed; logging to JSONL only.")

    def log(self, metrics: dict[str, float], step: int, to_console: bool = True) -> None:
        record = {"step": step, "elapsed_s": round(time.time() - self._start, 2), **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=float) + "\n")
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        if to_console:
            print(format_metrics(metrics, step, self.total_steps), flush=True)

    def log_completions(self, step: int, records: list[dict[str, Any]]) -> None:
        """Persist a few sampled completions per step.

        The cheapest possible defence against reward hacking: scalar rewards
        cannot tell you that the model started emitting empty ``<think>`` blocks
        or answering in the wrong units, and reading ten completions usually can.
        """
        if not records:
            return
        with self.completions_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps({"step": step, **record}, default=str) + "\n")

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
