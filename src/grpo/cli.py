"""Command line entry points: ``grpo train`` and ``grpo eval``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config

__all__ = ["main"]


def _load_config(args: argparse.Namespace) -> Config:
    config = Config.from_yaml(args.config) if args.config else Config().validate()
    if args.set:
        config = config.apply_overrides(args.set)
    return config


def _load_samples(config: Config, split: str, limit: int | None, tokenizer):
    from .data import load_gsm8k, load_jsonl

    name = config.train.dataset
    if name == "gsm8k":
        return load_gsm8k(split=split, limit=limit, tokenizer=tokenizer)
    return load_jsonl(name, limit=limit, tokenizer=tokenizer)


def cmd_train(args: argparse.Namespace) -> int:
    config = _load_config(args)
    from .rewards import build_ensemble
    from .rollout import build_engine
    from .trainer import GRPOTrainer, build_model_and_tokenizer, build_reference_model
    from .utils.seed import set_seed

    set_seed(config.train.seed)
    print(f"[grpo] loading {config.model.name_or_path}", flush=True)
    model, tokenizer = build_model_and_tokenizer(config)
    ref_model = build_reference_model(config)

    samples = _load_samples(config, config.train.dataset_split, config.train.dataset_limit, tokenizer)
    print(f"[grpo] {len(samples)} training prompts", flush=True)

    engine = build_engine(config, model, tokenizer)
    rewards = build_ensemble(config.rewards)
    print(f"[grpo] rewards: {rewards}", flush=True)
    print(
        f"[grpo] {config.rollout.prompts_per_step} prompts x {config.rollout.group_size} samples "
        f"= {config.sequences_per_step} sequences/step, "
        f"{config.grad_accum_steps} micro-batches of {config.optim.micro_batch_size}",
        flush=True,
    )

    trainer = GRPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        engine=engine,
        rewards=rewards,
        samples=samples,
        ref_model=ref_model,
    )
    if config.train.resume_from:
        trainer.load_checkpoint(config.train.resume_from)
        print(f"[grpo] resumed at step {trainer.step}", flush=True)

    try:
        trainer.train()
    finally:
        engine.close()
    print(f"[grpo] done -> {config.train.output_dir}", flush=True)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    config = _load_config(args)
    from .evaluate import evaluate_model
    from .trainer import build_model_and_tokenizer

    if args.model:
        config.model.name_or_path = args.model
    model, tokenizer = build_model_and_tokenizer(config)
    samples = _load_samples(config, args.split, args.limit, tokenizer)

    result = evaluate_model(
        model,
        tokenizer,
        samples,
        max_new_tokens=config.rollout.max_completion_length,
        batch_size=args.batch_size,
        k=args.k,
        max_prompt_length=config.rollout.max_prompt_length,
    )
    print(f"[grpo] {config.model.name_or_path} on {args.split}: {result}", flush=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result.as_metrics(), indent=2), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grpo", description="GRPO post-training for LLMs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--config", type=str, help="path to a YAML config")
        sub.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="SECTION.FIELD=VALUE",
            help="override a config field, e.g. --set algo.beta=0.0 (repeatable)",
        )

    train = subparsers.add_parser("train", help="run GRPO post-training")
    add_common(train)
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("eval", help="score a checkpoint on a held-out split")
    add_common(evaluate)
    evaluate.add_argument("--model", type=str, help="checkpoint path (overrides config)")
    evaluate.add_argument("--split", type=str, default="test")
    evaluate.add_argument("--limit", type=int, default=200)
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--k", type=int, default=1, help="samples per prompt; k=1 is greedy")
    evaluate.add_argument("--output", type=str, help="write metrics JSON here")
    evaluate.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
