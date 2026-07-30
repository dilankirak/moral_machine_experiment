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


KEY_COLUMNS = ["scenario_id", "language", "provider"]


def normalize_scenario_id(value):
    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value.lstrip("0") or "0"


def make_key(scenario_id, language, provider):
    return (
        normalize_scenario_id(scenario_id),
        str(language).strip(),
        str(provider).strip(),
    )


def load_existing(path):
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(
        path,
        dtype={"scenario_id": str},
    )


def load_records(path):
    existing = load_existing(path)

    if existing.empty:
        return {}

    required = set(KEY_COLUMNS + ["status"])

    if not required.issubset(existing.columns):
        missing = required.difference(existing.columns)
        raise ValueError(
            f"Fehlende Spalten in {path}: {sorted(missing)}"
        )

    records = {}

    for row in existing.to_dict("records"):
        key = make_key(
            row["scenario_id"],
            row["language"],
            row["provider"],
        )

        current = records.get(key)

        if current is None:
            records[key] = row
            continue

        current_ok = current.get("status") == "ok"
        new_ok = row.get("status") == "ok"

        if new_ok or not current_ok:
            records[key] = row

    return records


def completed_keys(records):
    return {
        key
        for key, record in records.items()
        if record.get("status") == "ok"
    }


def save_records(path, records):
    rows = list(records.values())

    if not rows:
        return

    dataframe = pd.DataFrame(rows)

    dataframe["_scenario_number"] = pd.to_numeric(
        dataframe["scenario_id"],
        errors="coerce",
    )

    dataframe = (
        dataframe.sort_values(
            ["_scenario_number", "language", "provider"],
            na_position="last",
        )
        .drop(columns="_scenario_number")
    )

    dataframe.to_csv(
        path,
        index=False,
    )


def main(input_path, output_path, limit=None, shuffle=False):
    validate(input_path)

    scenarios = pd.read_csv(
        input_path,
        dtype={"scenario_id": str},
    )

    if shuffle:
        scenarios = scenarios.sample(
            frac=1,
            random_state=42,
        ).reset_index(drop=True)

    if limit is not None:
        scenarios = scenarios.head(limit)

    models = {
        "openai": OpenAIModel(),
        "gemini": GeminiModel(),
    }

    records = load_records(output_path)
    done = completed_keys(records)

    save_records(output_path, records)

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

            key = make_key(
                row.scenario_id,
                row.language,
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
                    "scenario_id": normalize_scenario_id(
                        row.scenario_id
                    ),
                    "provider": provider,
                    "model": model.model,
                    "decision": result.decision,
                    "raw_response": result.raw_response,
                    "status": result.status,
                    "error": result.error,
                    "timestamp_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "latency_seconds": round(
                        result.latency_seconds,
                        4,
                    ),
                    "prompt_version": PROMPT_VERSION,
                    "python_version": sys.version.split()[0],
                    "operating_system": platform.platform(),
                }
            )

            records[key] = record
            save_records(output_path, records)

            if result.status == "ok":
                done.add(key)

            time.sleep(0.5)

    results = pd.DataFrame(records.values())

    print(f"\nDone. Results saved to {output_path}")

    if not results.empty:
        print(results[["provider", "status"]].value_counts())
        print("Gesamt:", len(results))


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