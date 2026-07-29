from __future__ import annotations

import argparse
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from model_clients import GeminiModel, OpenAIModel
from prompting import PROMPT_VERSION, build_prompt
from validate_dataset import validate


def load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def completed_keys(existing: pd.DataFrame) -> set[tuple[str, str, str]]:
    required = {"scenario_id", "language", "provider", "status"}
    if existing.empty or not required.issubset(existing.columns):
        return set()

    successful = existing[existing["status"] == "ok"]
    return {
        (str(row.scenario_id), str(row.language), str(row.provider))
        for row in successful.itertuples()
    }


def append_row(path: Path, row: dict) -> None:
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def main(input_path: Path, output_path: Path, limit: int | None, shuffle: bool) -> None:
    validate(input_path)
    scenarios = pd.read_csv(input_path, dtype={"scenario_id": str})

    # A fixed seed makes the optional randomized order reproducible.
    if shuffle:
        scenarios = scenarios.sample(frac=1, random_state=42).reset_index(drop=True)

    if limit is not None:
        scenarios = scenarios.head(limit)

    models = {
        "openai": OpenAIModel(),
        "gemini": GeminiModel(),
    }

    existing = load_existing(output_path)
    done = completed_keys(existing)

    total = len(scenarios) * len(models)
    current = 0

    for row in scenarios.itertuples(index=False):
        prompt = build_prompt(row.language, row.option_a, row.option_b)

        for provider, model_client in models.items():
            current += 1
            key = (str(row.scenario_id), str(row.language), provider)
            if key in done:
                print(f"[{current}/{total}] Übersprungen: {key}")
                continue

            print(f"[{current}/{total}] {provider}: Szenario {row.scenario_id} ({row.language})")
            result = model_client.run(prompt)

            record = row._asdict()
            record.update(
                {
                    "provider": provider,
                    "model": model_client.model,
                    "decision": result.decision,
                    "raw_response": result.raw_response,
                    "status": result.status,
                    "error": result.error,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "latency_seconds": round(result.latency_seconds, 4),
                    "prompt_version": PROMPT_VERSION,
                    "python_version": sys.version.split()[0],
                    "operating_system": platform.platform(),
                }
            )
            append_row(output_path, record)

            # Kurze Pause, um unnötige Rate-Limit-Probleme zu reduzieren.
            time.sleep(0.5)

    print(f"Fertig. Ergebnisse: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Reihenfolge reproduzierbar mit Seed 42 randomisieren.",
    )
    args = parser.parse_args()
    main(args.input, args.output, args.limit, args.shuffle)
