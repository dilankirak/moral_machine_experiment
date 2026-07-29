import csv
import sys
from collections import defaultdict
from pathlib import Path


REQUIRED_COLUMNS = [
    "scenario_id",
    "language",
    "dimension",
    "option_a",
    "option_b",
    "target_preference",
    "target_option",
]

VALID_LANGUAGES = {"en", "de", "tr"}

VALID_PREFERENCES = {
    "number": {"fewer", "more"},
    "age": {"younger", "older"},
    "species": {"humans", "animals"},
    "role": {"pedestrians", "passengers"},
    "law": {"lawful", "unlawful"},
}


def validate(file_path="scenarios.csv"):
    path = Path(file_path)

    if not path.exists():
        print(f"Datei nicht gefunden: {path}")
        return False

    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    errors = []

    if reader.fieldnames != REQUIRED_COLUMNS:
        errors.append(
            "Die Spalten entsprechen nicht der erwarteten Struktur."
        )

    scenarios = defaultdict(list)
    seen = set()

    for line_number, row in enumerate(rows, start=2):
        scenario_id = row["scenario_id"].strip()
        language = row["language"].strip().lower()
        dimension = row["dimension"].strip().lower()
        preference = row["target_preference"].strip().lower()
        target_option = row["target_option"].strip().upper()

        if not all(row[column].strip() for column in REQUIRED_COLUMNS):
            errors.append(f"Zeile {line_number}: Ein Wert fehlt.")

        if language not in VALID_LANGUAGES:
            errors.append(
                f"Zeile {line_number}: Ungültige Sprache '{language}'."
            )

        if dimension not in VALID_PREFERENCES:
            errors.append(
                f"Zeile {line_number}: Ungültige Dimension '{dimension}'."
            )
        elif preference not in VALID_PREFERENCES[dimension]:
            errors.append(
                f"Zeile {line_number}: Präferenz passt nicht zur Dimension."
            )

        if target_option not in {"A", "B"}:
            errors.append(
                f"Zeile {line_number}: target_option muss A oder B sein."
            )

        if row["option_a"].strip() == row["option_b"].strip():
            errors.append(
                f"Zeile {line_number}: Option A und B sind identisch."
            )

        key = (scenario_id, language)

        if key in seen:
            errors.append(
                f"Zeile {line_number}: Szenario und Sprache sind doppelt."
            )

        seen.add(key)
        scenarios[scenario_id].append(row)

    for scenario_id, scenario_rows in scenarios.items():
        languages = {
            row["language"].strip().lower()
            for row in scenario_rows
        }

        if languages != VALID_LANGUAGES:
            errors.append(
                f"Szenario {scenario_id}: Nicht alle Sprachen vorhanden."
            )

        for column in [
            "dimension",
            "target_preference",
            "target_option",
        ]:
            values = {
                row[column].strip().lower()
                for row in scenario_rows
            }

            if len(values) > 1:
                errors.append(
                    f"Szenario {scenario_id}: '{column}' ist uneinheitlich."
                )

    if errors:
        print("Datensatz enthält Fehler:\n")

        for error in errors:
            print(f"- {error}")

        return False

    print("Datensatz ist gültig.")
    print(f"Szenarien: {len(scenarios)}")
    print(f"Zeilen: {len(rows)}")

    return True


if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "scenarios.csv"

    if not validate(file_path):
        sys.exit(1)