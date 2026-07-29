import argparse
from pathlib import Path

import pandas as pd


RESULT_COLUMNS = {
    "scenario_id",
    "language",
    "provider",
    "model",
    "decision",
    "status",
}

SCENARIO_COLUMNS = {
    "scenario_id",
    "language",
    "dimension",
    "target_preference",
    "target_option",
}


def load_csv(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    return pd.read_csv(path, dtype={"scenario_id": str})


def check_columns(df, required, file_name):
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"In {file_name} fehlen folgende Spalten: "
            + ", ".join(sorted(missing))
        )


def normalize_ids(df):
    df = df.copy()
    df["scenario_id"] = df["scenario_id"].astype(str).str.strip().str.zfill(3)
    df["language"] = df["language"].astype(str).str.strip().str.lower()
    return df


def prepare_results(df):
    check_columns(df, RESULT_COLUMNS, "results.csv")
    df = normalize_ids(df)

    df["provider"] = df["provider"].astype(str).str.strip().str.lower()
    df["status"] = df["status"].astype(str).str.strip().str.lower()
    df["decision"] = df["decision"].astype("string").str.strip().str.upper()

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(
            df["timestamp_utc"],
            errors="coerce",
            utc=True,
        )
        df = df.sort_values("timestamp_utc", na_position="first")

    return df.drop_duplicates(
        subset=["scenario_id", "language", "provider"],
        keep="last",
    )


def prepare_scenarios(df):
    check_columns(df, SCENARIO_COLUMNS, "scenarios.csv")
    df = normalize_ids(df)

    df["dimension"] = df["dimension"].astype(str).str.strip()
    df["target_preference"] = (
        df["target_preference"].astype(str).str.strip()
    )
    df["target_option"] = (
        df["target_option"].astype("string").str.strip().str.upper()
    )

    if not df["target_option"].isin(["A", "B"]).all():
        raise ValueError("target_option darf nur A oder B enthalten.")

    columns = [
        "scenario_id",
        "language",
        "dimension",
        "target_preference",
        "target_option",
    ]

    return df[columns].drop_duplicates(
        subset=["scenario_id", "language"]
    )


def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Gespeichert: {path}")


def create_overview(df, valid):
    values = {
        "total_rows": len(df),
        "successful_rows": len(valid),
        "invalid_responses": (df["status"] == "invalid_response").sum(),
        "api_errors": (df["status"] == "error").sum(),
        "unique_scenarios": df["scenario_id"].nunique(),
        "languages": df["language"].nunique(),
        "providers": df["provider"].nunique(),
        "models": df["model"].nunique(),
    }

    return pd.DataFrame(
        {
            "metric": values.keys(),
            "value": values.values(),
        }
    )


def create_decision_summary(df, group):
    columns = [group, "decision", "count", "percentage"]

    if df.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        df.groupby([group, "decision"])
        .size()
        .reset_index(name="count")
    )

    totals = summary.groupby(group)["count"].transform("sum")
    summary["percentage"] = (summary["count"] / totals * 100).round(2)

    return summary


def create_provider_agreement(df):
    columns = [
        "language",
        "compared_cases",
        "agreements",
        "agreement_rate",
    ]

    comparison = df.pivot_table(
        index=["scenario_id", "language"],
        columns="provider",
        values="decision",
        aggfunc="first",
    ).reset_index()

    if not {"openai", "gemini"}.issubset(comparison.columns):
        return pd.DataFrame(columns=columns)

    comparison = comparison.dropna(subset=["openai", "gemini"])

    if comparison.empty:
        return pd.DataFrame(columns=columns)

    comparison["agreement"] = comparison["openai"] == comparison["gemini"]

    summary = (
        comparison.groupby("language")["agreement"]
        .agg(compared_cases="size", agreements="sum")
        .reset_index()
    )

    summary["agreement_rate"] = (
        summary["agreements"] / summary["compared_cases"] * 100
    ).round(2)

    overall = pd.DataFrame(
        [
            {
                "language": "overall",
                "compared_cases": len(comparison),
                "agreements": int(comparison["agreement"].sum()),
                "agreement_rate": round(
                    comparison["agreement"].mean() * 100,
                    2,
                ),
            }
        ]
    )

    return pd.concat([summary, overall], ignore_index=True)


def create_language_consistency(df):
    columns = [
        "provider",
        "complete_scenarios",
        "consistent_scenarios",
        "consistency_rate",
    ]
    languages = ["en", "de", "tr"]

    comparison = df.pivot_table(
        index=["scenario_id", "provider"],
        columns="language",
        values="decision",
        aggfunc="first",
    ).reset_index()

    if not set(languages).issubset(comparison.columns):
        return pd.DataFrame(columns=columns)

    comparison = comparison.dropna(subset=languages)

    if comparison.empty:
        return pd.DataFrame(columns=columns)

    comparison["consistent"] = comparison[languages].nunique(axis=1) == 1

    summary = (
        comparison.groupby("provider")["consistent"]
        .agg(
            complete_scenarios="size",
            consistent_scenarios="sum",
        )
        .reset_index()
    )

    summary["consistency_rate"] = (
        summary["consistent_scenarios"]
        / summary["complete_scenarios"]
        * 100
    ).round(2)

    return summary


def summarize_preferences(df, groups):
    summary = (
        df.groupby(groups)["preference_followed"]
        .agg(evaluated_cases="size", preference_followed="sum")
        .reset_index()
    )

    summary["preference_rate"] = (
        summary["preference_followed"]
        / summary["evaluated_cases"]
        * 100
    ).round(2)

    return summary


def create_dimension_summary(df):
    columns = [
        "dimension",
        "target_preference",
        "provider",
        "language",
        "evaluated_cases",
        "preference_followed",
        "preference_rate",
    ]

    data = df.dropna(
        subset=["dimension", "target_preference", "target_option"]
    ).copy()

    if data.empty:
        return pd.DataFrame(columns=columns)

    data["preference_followed"] = (
        data["decision"] == data["target_option"]
    )

    summary = summarize_preferences(
        data,
        ["dimension", "target_preference", "provider", "language"],
    )

    overall = summarize_preferences(
        data,
        ["dimension", "target_preference", "provider"],
    )
    overall["language"] = "overall"
    overall = overall[columns]

    return (
        pd.concat([summary, overall], ignore_index=True)
        .sort_values(["dimension", "provider", "language"])
        .reset_index(drop=True)
    )


def print_table(title, df, empty_message):
    print(f"\n=== {title} ===")

    if df.empty:
        print(empty_message)
    else:
        print(df.to_markdown(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Auswertung des mehrsprachigen Moral-Machine-Experiments"
    )
    parser.add_argument("--input", default="results.csv")
    parser.add_argument("--scenarios", default="scenarios.csv")
    parser.add_argument("--output-dir", default="analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = prepare_results(load_csv(args.input))
    scenarios = prepare_scenarios(load_csv(args.scenarios))

    metadata = ["dimension", "target_preference", "target_option"]
    results = results.drop(
        columns=[column for column in metadata if column in results.columns]
    )

    df = results.merge(
        scenarios,
        on=["scenario_id", "language"],
        how="left",
        validate="many_to_one",
    )

    valid = df[
        (df["status"] == "ok")
        & df["decision"].isin(["A", "B"])
    ].copy()

    outputs = {
        "overview.csv": create_overview(df, valid),
        "decisions_by_provider.csv": create_decision_summary(
            valid,
            "provider",
        ),
        "decisions_by_language.csv": create_decision_summary(
            valid,
            "language",
        ),
        "provider_agreement_summary.csv": create_provider_agreement(valid),
        "language_consistency_summary.csv": create_language_consistency(valid),
        "moral_machine_dimensions.csv": create_dimension_summary(valid),
    }

    for file_name, data in outputs.items():
        save_csv(data, output_dir / file_name)

    print_table(
        "Überblick",
        outputs["overview.csv"],
        "Keine Ergebnisse vorhanden.",
    )

    print_table(
        "Übereinstimmung zwischen den Modellen",
        outputs["provider_agreement_summary.csv"],
        "Keine vergleichbaren Modellentscheidungen vorhanden.",
    )

    print_table(
        "Sprachliche Konsistenz",
        outputs["language_consistency_summary.csv"],
        "Keine vollständigen Sprachversionen vorhanden.",
    )

    print_table(
        "Moral-Machine-Dimensionen",
        outputs["moral_machine_dimensions.csv"],
        "Keine auswertbaren Dimensionen vorhanden.",
    )

    print(f"\nAuswertung abgeschlossen. Ergebnisse: {output_dir}")


if __name__ == "__main__":
    main()