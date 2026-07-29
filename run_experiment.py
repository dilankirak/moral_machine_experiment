import argparse
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from model_clients import GeminiModel, OpenAIModel
from prompting import PROMPT_VERSION, build_prompt
from validate_dataset import validate


def load_existing(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def completed_keys(existing):
    required = {"scenario_id", "language", "provider", "status"}

    if existing.empty or not required.issubset(existing.columns):
        return set()

    return {
        (str(row.scenario_id), str(row.language), str(row.provider))
        for row in existing[existing["status"] == "ok"].itertuples()
    }


def append_row(path, row):
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def main(input_path, output_path, limit=None, shuffle=False):
    validate(input_path)

    scenarios = pd.read_csv(
        input_path,
        dtype={"scenario_id": str},
    )

    if shuffle:
        scenarios = (
            scenarios.sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

    if limit:
        scenarios = scenarios.head(limit)

    models = {
        "openai": OpenAIModel(),
        "gemini": GeminiModel(),
    }

    done = completed_keys(load_existing(output_path))

    total = len(scenarios) * len(models)
    current = 0

    for row in scenarios.itertuples(index=False):
        prompt = build_prompt(
            row.language,
            row.option_a,
            row.option_b,
        )

        for provider, model in models.items():
            current += 1

            key = (
                str(row.scenario_id),
                str(row.language),
                provider,
            )

            if key in done:
                print(f"[{current}/{total}] Skip {key}")
                continue

            print(
                f"[{current}/{total}] "
                f"{provider} | "
                f"{row.scenario_id} ({row.language})"
            )

            result = model.run(prompt)

            record = row._asdict()

            record.update(
                {
                    "provider": provider,
                    "model": model.model,
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

            time.sleep(0.5)

    print(f"\nDone. Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--limit",
        type=int,
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
    )

    args = parser.parse_args()

    main(
        args.input,
        args.output,
        args.limit,
        args.shuffle,
    )