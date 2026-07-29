from pathlib import Path
import argparse

import pandas as pd


DIMENSIONS = {
    "number_preference": "Anzahl",
    "age_preference": "Alter",
    "species_preference": "Mensch oder Tier",
    "role_preference": "Passagier oder Fußgänger",
    "law_preference": "Regelkonformität",
}


def load_csv(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Die Datei wurde nicht gefunden: {file_path}"
        )

    return pd.read_csv(
        path,
        dtype={"scenario_id": str},
    )


def prepare_results(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "scenario_id",
        "language",
        "provider",
        "model",
        "decision",
        "status",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "In der Ergebnisdatei fehlen folgende Spalten: "
            + ", ".join(missing_columns)
        )

    df["scenario_id"] = (
        df["scenario_id"]
        .astype(str)
        .str.strip()
        .str.zfill(3)
    )

    df["language"] = (
        df["language"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["provider"] = (
        df["provider"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["status"] = (
        df["status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["decision"] = (
        df["decision"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    return df


def prepare_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "scenario_id",
        "language",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "In der Szenariodatei fehlen folgende Spalten: "
            + ", ".join(missing_columns)
        )

    df["scenario_id"] = (
        df["scenario_id"]
        .astype(str)
        .str.strip()
        .str.zfill(3)
    )

    df["language"] = (
        df["language"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


def save_csv(dataframe: pd.DataFrame, file_path: Path) -> None:
    dataframe.to_csv(
        file_path,
        index=False,
        encoding="utf-8",
    )

    print(f"Gespeichert: {file_path}")


def create_decision_summary(
    df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    summary = (
        df.groupby([group_column, "decision"])
        .size()
        .reset_index(name="count")
    )

    totals = (
        summary.groupby(group_column)["count"]
        .transform("sum")
    )

    summary["percentage"] = (
        summary["count"] / totals * 100
    ).round(2)

    return summary


def create_provider_agreement(
    df: pd.DataFrame,
) -> pd.DataFrame:
    agreement = df.pivot_table(
        index=["scenario_id", "language"],
        columns="provider",
        values="decision",
        aggfunc="first",
    ).reset_index()

    if (
        "openai" not in agreement.columns
        or "gemini" not in agreement.columns
    ):
        return pd.DataFrame()

    agreement = agreement.dropna(
        subset=["openai", "gemini"]
    ).copy()

    agreement["agreement"] = (
        agreement["openai"] == agreement["gemini"]
    )

    summary = (
        agreement.groupby("language")
        .agg(
            compared_cases=("agreement", "size"),
            agreements=("agreement", "sum"),
        )
        .reset_index()
    )

    summary["agreement_rate"] = (
        summary["agreements"]
        / summary["compared_cases"]
        * 100
    ).round(2)

    overall = pd.DataFrame(
        {
            "language": ["overall"],
            "compared_cases": [len(agreement)],
            "agreements": [
                int(agreement["agreement"].sum())
            ],
            "agreement_rate": [
                round(
                    agreement["agreement"].mean() * 100,
                    2,
                )
                if len(agreement) > 0
                else 0.0
            ],
        }
    )

    return pd.concat(
        [summary, overall],
        ignore_index=True,
    )


def create_language_consistency(
    df: pd.DataFrame,
) -> pd.DataFrame:
    consistency = df.pivot_table(
        index=["scenario_id", "provider"],
        columns="language",
        values="decision",
        aggfunc="first",
    ).reset_index()

    required_languages = ["en", "de", "tr"]

    if not all(
        language in consistency.columns
        for language in required_languages
    ):
        return pd.DataFrame()

    consistency = consistency.dropna(
        subset=required_languages
    ).copy()

    consistency["consistent"] = (
        consistency[required_languages]
        .nunique(axis=1)
        == 1
    )

    summary = (
        consistency.groupby("provider")
        .agg(
            complete_scenarios=("consistent", "size"),
            consistent_scenarios=("consistent", "sum"),
        )
        .reset_index()
    )

    summary["consistency_rate"] = (
        summary["consistent_scenarios"]
        / summary["complete_scenarios"]
        * 100
    ).round(2)

    return summary


def create_dimension_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    results = []

    for column, dimension_name in DIMENSIONS.items():
        if column not in df.columns:
            print(
                f"Hinweis: Die Spalte '{column}' "
                "wurde nicht gefunden."
            )
            continue

        dimension_data = df[
            df[column].notna()
        ].copy()

        dimension_data[column] = (
            dimension_data[column]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        dimension_data = dimension_data[
            dimension_data[column].isin(["A", "B"])
        ].copy()

        if dimension_data.empty:
            print(
                f"Hinweis: Für '{column}' wurden "
                "keine A/B-Werte gefunden."
            )
            continue

        dimension_data["preference_followed"] = (
            dimension_data["decision"]
            == dimension_data[column]
        )

        summary = (
            dimension_data.groupby(
                ["provider", "language"]
            )
            .agg(
                evaluated_cases=(
                    "preference_followed",
                    "size",
                ),
                preference_followed=(
                    "preference_followed",
                    "sum",
                ),
            )
            .reset_index()
        )

        summary["preference_rate"] = (
            summary["preference_followed"]
            / summary["evaluated_cases"]
            * 100
        ).round(2)

        summary.insert(
            0,
            "dimension",
            dimension_name,
        )

        results.append(summary)

        overall = (
            dimension_data.groupby("provider")
            .agg(
                evaluated_cases=(
                    "preference_followed",
                    "size",
                ),
                preference_followed=(
                    "preference_followed",
                    "sum",
                ),
            )
            .reset_index()
        )

        overall["language"] = "overall"

        overall["preference_rate"] = (
            overall["preference_followed"]
            / overall["evaluated_cases"]
            * 100
        ).round(2)

        overall.insert(
            0,
            "dimension",
            dimension_name,
        )

        overall = overall[
            [
                "dimension",
                "provider",
                "language",
                "evaluated_cases",
                "preference_followed",
                "preference_rate",
            ]
        ]

        results.append(overall)

    if not results:
        return pd.DataFrame(
            columns=[
                "dimension",
                "provider",
                "language",
                "evaluated_cases",
                "preference_followed",
                "preference_rate",
            ]
        )

    return pd.concat(
        results,
        ignore_index=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Auswertung des mehrsprachigen "
            "Moral-Machine-Experiments."
        )
    )

    parser.add_argument(
        "--input",
        default="results.csv",
        help="Pfad zur Ergebnisdatei",
    )

    parser.add_argument(
        "--scenarios",
        default="scenarios.csv",
        help="Pfad zur Szenariodatei",
    )

    parser.add_argument(
        "--output-dir",
        default="analysis",
        help="Ordner für die Auswertungsdateien",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = load_csv(args.input)
    scenarios = load_csv(args.scenarios)

    results = prepare_results(results)
    scenarios = prepare_scenarios(scenarios)

    dimension_columns = [
        column
        for column in DIMENSIONS
        if column in scenarios.columns
    ]

    scenario_columns = [
        "scenario_id",
        "language",
    ] + dimension_columns

    scenarios = scenarios[
        scenario_columns
    ].drop_duplicates(
        subset=["scenario_id", "language"]
    )

    df = results.merge(
        scenarios,
        on=["scenario_id", "language"],
        how="left",
    )

    df_ok = df[
        (df["status"] == "ok")
        & (df["decision"].isin(["A", "B"]))
    ].copy()

    overview = pd.DataFrame(
        {
            "metric": [
                "total_rows",
                "successful_rows",
                "invalid_responses",
                "api_errors",
                "unique_scenarios",
                "languages",
                "providers",
                "models",
            ],
            "value": [
                len(df),
                len(df_ok),
                int(
                    (
                        df["status"]
                        == "invalid_response"
                    ).sum()
                ),
                int(
                    (
                        df["status"]
                        == "error"
                    ).sum()
                ),
                df["scenario_id"].nunique(),
                df["language"].nunique(),
                df["provider"].nunique(),
                df["model"].nunique(),
            ],
        }
    )

    decisions_by_provider = create_decision_summary(
        df_ok,
        "provider",
    )

    decisions_by_language = create_decision_summary(
        df_ok,
        "language",
    )

    agreement_summary = create_provider_agreement(
        df_ok
    )

    consistency_summary = create_language_consistency(
        df_ok
    )

    dimension_summary = create_dimension_summary(
        df_ok
    )

    save_csv(
        overview,
        output_dir / "overview.csv",
    )

    save_csv(
        decisions_by_provider,
        output_dir / "decisions_by_provider.csv",
    )

    save_csv(
        decisions_by_language,
        output_dir / "decisions_by_language.csv",
    )

    if not agreement_summary.empty:
        save_csv(
            agreement_summary,
            output_dir
            / "provider_agreement_summary.csv",
        )

    if not consistency_summary.empty:
        save_csv(
            consistency_summary,
            output_dir
            / "language_consistency_summary.csv",
        )

    save_csv(
        dimension_summary,
        output_dir
        / "moral_machine_dimensions.csv",
    )

    print("\n=== Überblick ===")
    print(overview.to_markdown(index=False))

    if not agreement_summary.empty:
        print(
            "\n=== Übereinstimmung "
            "zwischen den Modellen ==="
        )
        print(
            agreement_summary.to_markdown(
                index=False
            )
        )

    if not consistency_summary.empty:
        print(
            "\n=== Sprachliche Konsistenz ==="
        )
        print(
            consistency_summary.to_markdown(
                index=False
            )
        )

    if not dimension_summary.empty:
        print(
            "\n=== Moral-Machine-Dimensionen ==="
        )
        print(
            dimension_summary.to_markdown(
                index=False
            )
        )
    else:
        print(
            "\nKeine Moral-Machine-Dimensionen "
            "konnten ausgewertet werden."
        )

    print(
        "\nAuswertung abgeschlossen. "
        f"Ergebnisse befinden sich im Ordner: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()