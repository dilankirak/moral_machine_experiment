
# Bibliotheken für Kommandozeilenargumente, Systeminformationen,
# Zeitmessung, Zeitstempel und Dateipfade
import argparse
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Pandas zur Verarbeitung und Speicherung der CSV-Daten
import pandas as pd

# Eigene Module für die Modellanbindung, Prompterstellung
# und Validierung des Datensatzes
from model_clients import GeminiModel, OpenAIModel
from prompting import PROMPT_VERSION, build_prompt
from validate_dataset import validate


# Diese drei Spalten identifizieren jede Modellanfrage eindeutig.
# So kann überprüft werden, ob ein Szenario für ein bestimmtes
# Modell und eine bestimmte Sprache bereits bearbeitet wurde.
KEY_COLUMNS = ["scenario_id", "language", "provider"]


def normalize_scenario_id(value):
    """
    Vereinheitlicht die Szenario-ID.

    Beispielsweise werden '001', '1.0' und '1'
    auf denselben Wert '1' gebracht.
    Dadurch werden doppelte Einträge vermieden.
    """
    value = str(value).strip()

    # Entfernt die Endung '.0', falls vorhanden
    if value.endswith(".0"):
        value = value[:-2]

    # Entfernt führende Nullen
    return value.lstrip("0") or "0"


def make_key(scenario_id, language, provider):
    """
    Erstellt einen eindeutigen Schlüssel für eine Modellanfrage.

    Der Schlüssel besteht aus:
    - Szenario-ID
    - Sprache
    - Modellanbieter

    Beispiel: ('1', 'de', 'openai')
    """
    return (
        normalize_scenario_id(scenario_id),
        str(language).strip(),
        str(provider).strip(),
    )


def load_existing(path):
    """
    Lädt eine bereits vorhandene Ergebnisdatei.

    Falls die Datei noch nicht existiert, wird ein
    leerer DataFrame zurückgegeben.
    """

    if not path.exists():
        return pd.DataFrame()

    # Szenario-IDs werden als Text eingelesen,
    # damit ihre ursprüngliche Darstellung erhalten bleibt.
    return pd.read_csv(
        path,
        dtype={"scenario_id": str},
    )


def load_records(path):
    """
    Lädt vorhandene Modellantworten und ordnet sie
    ihren eindeutigen Schlüsseln zu.

    Bei mehrfach vorhandenen Einträgen wird nach
    Möglichkeit eine erfolgreiche Antwort bevorzugt.
    """

    existing = load_existing(path)

    if existing.empty:
        return {}

    # Prüft, ob die notwendigen Spalten vorhanden sind
    required = set(KEY_COLUMNS + ["status"])

    if not required.issubset(existing.columns):
        missing = required.difference(existing.columns)
        raise ValueError(
            f"Fehlende Spalten in {path}: {sorted(missing)}"
        )

    records = {}

    # Durchläuft alle bereits gespeicherten Ergebnisse
    for row in existing.to_dict("records"):
        key = make_key(
            row["scenario_id"],
            row["language"],
            row["provider"],
        )

        current = records.get(key)

        # Falls noch kein Ergebnis existiert, wird es gespeichert
        if current is None:
            records[key] = row
            continue

        # Überprüft, ob das bisherige und das neue
        # Ergebnis erfolgreich waren
        current_ok = current.get("status") == "ok"
        new_ok = row.get("status") == "ok"

        # Erfolgreiche Ergebnisse werden bevorzugt.
        # Ist das bisherige Ergebnis fehlerhaft,
        # kann es durch einen neueren Eintrag ersetzt werden.
        if new_ok or not current_ok:
            records[key] = row

    return records


def completed_keys(records):
    """
    Ermittelt alle bereits erfolgreich abgeschlossenen Anfragen.

    Diese Anfragen müssen bei einer erneuten Ausführung
    des Experiments nicht noch einmal durchgeführt werden.
    """

    return {
        key
        for key, record in records.items()
        if record.get("status") == "ok"
    }


def save_records(path, records):
    """
    Speichert die bisherigen Modellergebnisse in einer CSV-Datei.

    Vor dem Speichern werden die Ergebnisse nach
    Szenario-ID, Sprache und Modellanbieter sortiert.
    """

    rows = list(records.values())

    if not rows:
        return

    dataframe = pd.DataFrame(rows)

    # Wandelt die Szenario-ID für die Sortierung in Zahlen um
    dataframe["_scenario_number"] = pd.to_numeric(
        dataframe["scenario_id"],
        errors="coerce",
    )

    # Sortiert die Ergebnisse und entfernt anschließend
    # die nur für die Sortierung benötigte Hilfsspalte
    dataframe = (
        dataframe.sort_values(
            ["_scenario_number", "language", "provider"],
            na_position="last",
        )
        .drop(columns="_scenario_number")
    )

    # Speichert die Ergebnisse ohne zusätzlichen Zeilenindex
    dataframe.to_csv(
        path,
        index=False,
    )


def main(input_path, output_path, limit=None, shuffle=False):
    """
    Hauptfunktion zur automatisierten Durchführung des Experiments.

    Ablauf:
    1. Datensatz validieren
    2. Szenarien einlesen
    3. Sprachmodelle initialisieren
    4. Bereits vorhandene Ergebnisse laden
    5. Prompts erstellen und an die Modelle senden
    6. Modellantworten dokumentieren und speichern
    7. Zusammenfassung der Ergebnisse ausgeben
    """

    # SCHRITT 1: DATENSATZ VALIDIEREN

    # Prüft den Eingabedatensatz vor der Durchführung
    # auf die im Validierungsskript definierten Anforderungen
    validate(input_path)

    # SCHRITT 2: SZENARIEN EINLESEN

    # Liest die Moral-Machine-Szenarien aus der CSV-Datei
    scenarios = pd.read_csv(
        input_path,
        dtype={"scenario_id": str},
    )

    # Optional: Mischt die Reihenfolge der Szenarien.
    # random_state=42 ermöglicht eine reproduzierbare Reihenfolge.
    if shuffle:
        scenarios = scenarios.sample(
            frac=1,
            random_state=42,
        ).reset_index(drop=True)

    # Optional: Beschränkt die Anzahl der Szenarien,
    # beispielsweise für einen Testlauf
    if limit is not None:
        scenarios = scenarios.head(limit)

    # SCHRITT 3: SPRACHMODELLE INITIALISIEREN

    # Erstellt die Modell-Clients für GPT-4o und Gemini.
    # Die eigentliche API-Kommunikation ist in
    # model_clients.py implementiert.
    models = {
        "openai": OpenAIModel(),
        "gemini": GeminiModel(),
    }

    # SCHRITT 4: VORHANDENE ERGEBNISSE LADEN

    # Bereits gespeicherte Ergebnisse werden eingelesen.
    records = load_records(output_path)

    # Ermittelt alle erfolgreich abgeschlossenen Anfragen
    done = completed_keys(records)

    # Speichert vorhandene Ergebnisse in sortierter Form
    save_records(output_path, records)

    # Berechnet die Anzahl der vorgesehenen Modellanfragen
    total = len(scenarios) * len(models)
    current = 0

    # SCHRITT 5: EXPERIMENT DURCHFÜHREN

    # Durchläuft sämtliche Szenarien des Datensatzes
    for row in scenarios.itertuples(index=False):

        # Erstellt einen Prompt in der jeweiligen Sprache.
        # Der Prompt enthält die beiden Entscheidungsoptionen A und B.
        prompt = build_prompt(
            row.language,
            row.option_a,
            row.option_b,
        )

        # Jedes Szenario wird an beide Modelle übermittelt
        for provider, model in models.items():
            current += 1

            # Erstellt einen eindeutigen Schlüssel für die Anfrage
            key = make_key(
                row.scenario_id,
                row.language,
                provider,
            )

            # Überspringt bereits erfolgreich bearbeitete Anfragen.
            # Dadurch wird die unnötige Wiederholung von
            # API-Anfragen vermieden.
            if key in done:
                print(f"[{current}/{total}] Skip {key}")
                continue

            # Zeigt den aktuellen Fortschritt in der Konsole
            print(
                f"[{current}/{total}] "
                f"{provider} | "
                f"{row.scenario_id} ({row.language})"
            )

            # Sendet den Prompt an das jeweilige Sprachmodell.
            # Das Ergebnis enthält unter anderem die Entscheidung,
            # den Antwortstatus und die Antwortzeit.
            result = model.run(prompt)

            # Übernimmt die ursprünglichen Szenariodaten
            record = row._asdict()

            # SCHRITT 6: ERGEBNISSE DOKUMENTIEREN

            # Ergänzt die Szenariodaten um das Modellergebnis
            # und weitere Informationen zur Nachvollziehbarkeit
            record.update(
                {
                    # Einheitliche Szenario-ID
                    "scenario_id": normalize_scenario_id(
                        row.scenario_id
                    ),

                    # Verwendeter Modellanbieter
                    "provider": provider,

                    # Konkrete Modellbezeichnung
                    "model": model.model,

                    # Getroffene Entscheidung (A oder B,
                    # sofern eine gültige Antwort vorliegt)
                    "decision": result.decision,

                    # Unveränderte ursprüngliche Modellantwort
                    "raw_response": result.raw_response,

                    # Status der Modellanfrage
                    "status": result.status,

                    # Fehlermeldung, falls ein Fehler aufgetreten ist
                    "error": result.error,

                    # Zeitpunkt der Anfrage in UTC
                    "timestamp_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),

                    # Dauer der Modellanfrage in Sekunden
                    "latency_seconds": round(
                        result.latency_seconds,
                        4,
                    ),

                    # Version der verwendeten Promptvorlage
                    "prompt_version": PROMPT_VERSION,

                    # Verwendete Python-Version
                    "python_version": sys.version.split()[0],

                    # Informationen zum Betriebssystem
                    "operating_system": platform.platform(),
                }
            )

            # Speichert das aktuelle Ergebnis im Dictionary
            records[key] = record

            # Speichert die Ergebnisse unmittelbar nach jeder Anfrage.
            # Dadurch bleiben bereits erhobene Daten auch dann
            # erhalten, wenn das Experiment unterbrochen wird.
            save_records(output_path, records)

            # Erfolgreiche Anfragen werden als abgeschlossen markiert
            if result.status == "ok":
                done.add(key)

            # Kurze Pause zwischen den Modellanfragen
            time.sleep(0.5)

    # SCHRITT 7: ERGEBNISSE ZUSAMMENFASSEN

    # Wandelt die gesammelten Ergebnisse in einen DataFrame um
    results = pd.DataFrame(records.values())

    print(f"\nDone. Results saved to {output_path}")

    # Gibt die Anzahl der Ergebnisse nach Anbieter
    # und Bearbeitungsstatus aus
    if not results.empty:
        print(results[["provider", "status"]].value_counts())
        print("Gesamt:", len(results))


# Einstiegspunkt des Python-Programms.
# Dieser Abschnitt wird nur ausgeführt, wenn die Datei
# direkt gestartet wird.
if __name__ == "__main__":

    # Erstellt den Parser für Kommandozeilenargumente
    parser = argparse.ArgumentParser()

    # Pfad zur CSV-Datei mit den Moral-Machine-Szenarien
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    # Pfad zur CSV-Datei für die Modellergebnisse
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    # Optional: Anzahl der zu bearbeitenden Szenariozeilen
    parser.add_argument(
        "--limit",
        type=int,
    )

    # Optional: Aktiviert eine zufällige,
    # aber reproduzierbare Szenarioreihenfolge
    parser.add_argument(
        "--shuffle",
        action="store_true",
    )

    # Liest die übergebenen Kommandozeilenargumente aus
    args = parser.parse_args()

    # Startet das Experiment mit den gewählten Einstellungen
    main(
        args.input,
        args.output,
        args.limit,
        args.shuffle,
    )
