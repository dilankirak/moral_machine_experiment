from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"scenario_id", "language", "option_a", "option_b"}
EXPECTED_LANGUAGES = {"en", "de", "tr"}


def validate(path: Path) -> None:
    # scenario_id wird als Text eingelesen, damit z. B. "001" erhalten bleibt.
    df = pd.read_csv(path, dtype={"scenario_id": str})

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Pflichtspalten: {sorted(missing)}")

    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Mindestens eine Pflichtspalte enthält leere Werte.")

    unknown = set(df["language"].astype(str)) - EXPECTED_LANGUAGES
    if unknown:
        raise ValueError(f"Unbekannte Sprachcodes: {sorted(unknown)}")

    duplicates = df.duplicated(subset=["scenario_id", "language"])
    if duplicates.any():
        rows = df.loc[duplicates, ["scenario_id", "language"]]
        raise ValueError(
            "Doppelte Szenario-Sprach-Kombinationen:\n"
            f"{rows.to_string(index=False)}"
        )

    counts = df.groupby("scenario_id")["language"].nunique()
    incomplete = counts[counts != 3]

    if not incomplete.empty:
        raise ValueError(
            "Folgende Szenarien liegen nicht in genau drei Sprachen vor: "
            f"{incomplete.index.tolist()}"
        )

    for scenario_id, group in df.groupby("scenario_id"):
        languages = set(group["language"].astype(str))

        if languages != EXPECTED_LANGUAGES:
            raise ValueError(
                f"Szenario {scenario_id} enthält nicht exakt en, de und tr."
            )

    print(
        f"Datensatz gültig: {df['scenario_id'].nunique()} Szenarien, "
        f"{len(df)} Sprachversionen."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validiert den mehrsprachigen Moral-Machine-Datensatz."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Pfad zur CSV-Datei mit den Szenarien.",
    )

    args = parser.parse_args()
    validate(args.input)