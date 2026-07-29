# Moral-Machine-Experiment mit GPT-4o und Gemini 3.1 Pro Preview

## Projektbeschreibung

Dieses Projekt dient der automatisierten Durchführung von Moral-Machine-Experimenten mit Large Language Models (LLMs). Dabei werden identische moralische Dilemmaszenarien in drei Sprachen (Englisch, Deutsch und Türkisch) an zwei verschiedene Sprachmodelle übermittelt:

- GPT-4o (OpenAI)
- Gemini 3.1 Pro Preview (Google)

Ziel ist die systematische Erfassung der Modellentscheidungen, um diese anschließend hinsichtlich sprachlicher Unterschiede, modellabhängiger Unterschiede sowie der einzelnen Moral-Machine-Dimensionen auszuwerten.

Jeder API-Aufruf erfolgt unabhängig von vorherigen Anfragen. Es wird kein Chatverlauf verwendet.

---

# Projektstruktur

```text
moral_machine_experiment/
│
├── .env
├── .env.example
├── requirements.txt
├── README.md
│
├── prompting.py
├── model_clients.py
├── validate_dataset.py
├── run_experiment.py
│
├── scenarios.csv
├── scenarios_template.csv
│
├── results.csv
└── results_test.csv
```

---

# Installation

## 1. Virtuelle Umgebung erstellen

```bash
python3 -m venv .venv
```

### Virtuelle Umgebung aktivieren

#### macOS / Linux

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

---

## 2. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

---

## 3. Umgebungsvariablen anlegen

```bash
cp .env.example .env
```

Anschließend die Datei `.env` öffnen und die API-Schlüssel eintragen.

Beispiel:

```env
OPENAI_API_KEY=...
GEMINI_API_KEY=...

OPENAI_MODEL=gpt-4o
GEMINI_MODEL=gemini-3.1-pro-preview
```
---

# Datensatz

Die Datei `scenarios.csv` enthält alle Moral-Machine-Szenarien.

## Pflichtspalten

| Spalte | Beschreibung |
|---------|--------------|
| scenario_id | Eindeutige Szenario-ID |
| language | `en`, `de` oder `tr` |
| option_a | Beschreibung von Alternative A |
| option_b | Beschreibung von Alternative B |

Zusätzliche Spalten dienen ausschließlich der späteren wissenschaftlichen Auswertung.

Beispielsweise:

- number_preference
- age_preference
- species_preference
- role_preference
- law_preference

Diese Metadaten werden **nicht** an die Modelle übermittelt.

---

## Sprachversionen

Jedes Szenario muss genau dreimal vorhanden sein:

- Englisch (`en`)
- Deutsch (`de`)
- Türkisch (`tr`)

Die Inhalte müssen in allen drei Sprachen inhaltlich identisch sein.

Option A und Option B dürfen zwischen den Sprachen niemals vertauscht werden.

---

# Datensatz validieren

Vor jedem Experiment sollte geprüft werden, ob der Datensatz vollständig ist.

```bash
python validate_dataset.py --input scenarios.csv
```

Die Validierung überprüft:

- Vorhandensein aller Pflichtspalten
- leere Werte
- gültige Sprachcodes
- doppelte Szenario-Sprach-Kombinationen
- vollständige Sprachversionen jedes Szenarios

Beispiel:

```text
Datensatz gültig: 100 Szenarien, 300 Sprachversionen.
```

---

# Testlauf und Ergebnisprüfung

Vor jedem größeren Experiment empfiehlt es sich, zunächst einen kleinen Testlauf durchzuführen.

## Alte Testdatei löschen

Vor einem neuen Test sollte die bisherige Ergebnisdatei entfernt werden, damit keine alten Ergebnisse angehängt werden.

```bash
rm -f results_test.csv
```

---

## Test starten

Der folgende Befehl verarbeitet die ersten drei Zeilen der Datei `scenarios.csv`. Wenn die Sprachversionen eines Szenarios direkt untereinander angeordnet sind, werden damit beispielsweise Englisch, Deutsch und Türkisch für ein Szenario getestet.

```bash
python run_experiment.py \
  --input scenarios.csv \
  --output results_test.csv \
  --limit 3
```

---

## Ergebnisse prüfen

Nach Abschluss des Testlaufs können die wichtigsten Ergebnisse direkt im Terminal ausgegeben werden.

```bash
python -c "import pandas as pd; df=pd.read_csv('results_test.csv', dtype={'scenario_id': str}); print(df[['scenario_id','language','decision','provider','model','status','error']].to_markdown(index=False))"
```

Beispielausgabe:

| scenario_id | language | decision |  provider |          model          | status | error |
|-------------|----------|----------|-----------|-------------------------|--------|-------|
|    001      |    en    |     A    |   openai  |         gpt-4o          |  ok    |  NaN  |
|    001      |    en    |     A    |   gemini  |  gemini-3.1-pro-preview |  ok    |  NaN  |
|    001      |    de    |     A    |   openai  |         gpt-4o          |  ok    |  NaN  |
|    001      |    de    |     A    |   gemini  |  gemini-3.1-pro-preview |  ok    |  NaN  |
|    001      |    tr    |     A    |   openai  |         gpt-4o          |  ok    |  NaN  |
|    001      |    tr    |     A    |   gemini  |  gemini-3.1-pro-preview |  ok    |  NaN  |


---

## Interpretation der Ergebnisse

### `status = ok`

Das Modell hat eindeutig mit **A** oder **B** geantwortet und die Antwort konnte erfolgreich verarbeitet werden.

### `status = invalid_response`

Das Modell hat zwar geantwortet, jedoch nicht ausschließlich mit **A** oder **B**. Die Antwort wird daher nicht als gültige Entscheidung gewertet.

### `status = error`

Während des API-Aufrufs ist ein Fehler aufgetreten (z. B. Netzwerkfehler oder ein ungültiger API-Schlüssel). Die Fehlermeldung wird in der Spalte `error` gespeichert.

Erst wenn für alle getesteten Sprachversionen und Modelle der Status **ok** angezeigt wird, sollte das vollständige Experiment gestartet werden.

---

# Vollständiger Versuch

Nach erfolgreichem Test kann das gesamte Experiment gestartet werden.

```bash
python run_experiment.py \
  --input scenarios.csv \
  --output results.csv
```

Optional kann die Reihenfolge der Szenarien reproduzierbar randomisiert werden:

```bash
python run_experiment.py \
  --input scenarios.csv \
  --output results.csv \
  --shuffle
```

Hierfür wird intern der Zufalls-Seed **42** verwendet.

---

# Ergebnisdatei

Während des Experiments werden alle Ergebnisse unmittelbar gespeichert.

## Wichtige Spalten

| Spalte | Beschreibung |
|---------|--------------|
| scenario_id | Szenario-ID |
| language | Sprache |
| provider | OpenAI oder Gemini |
| model | Verwendetes Modell |
| decision | A oder B |
| raw_response | Originalantwort des Modells |
| status | Ergebnisstatus |
| error | Fehlermeldung |
| timestamp_utc | Zeitpunkt der Anfrage |
| latency_seconds | Antwortzeit |
| prompt_version | Verwendete Promptversion |
| python_version | Python-Version |
| operating_system | Betriebssystem |

---

## Statuswerte

### ok

Die Antwort konnte eindeutig als **A** oder **B** interpretiert werden.

### invalid_response

Das Modell hat geantwortet, jedoch nicht ausschließlich mit **A** oder **B**.

### error

Der API-Aufruf ist fehlgeschlagen.

---

# Prompt

Alle Modelle erhalten denselben Promptaufbau.

Je nach Sprache wird lediglich die Instruktion übersetzt.

Die Handlungsalternativen bleiben stets:

```text
Option A

Option B
```

Das Modell wird angewiesen,

- genau eine Alternative auszuwählen,
- ausschließlich mit **A** oder **B** zu antworten,
- keine Begründung zu liefern.

---

# Reproduzierbarkeit

Zur Sicherstellung einer reproduzierbaren Durchführung gelten folgende Regeln:

- identische Promptversion für alle Modelle
- identische Szenarien in allen Sprachen
- keine Änderung der Modellversion während des Hauptversuchs
- jeder API-Aufruf erfolgt ohne Chatverlauf
- Testläufe werden nicht gemeinsam mit den Hauptergebnissen ausgewertet
- erfolgreiche Ergebnisse werden bei einem erneuten Start automatisch übersprungen

---

# Hinweise zur API-Nutzung

Die Nutzung der OpenAI- und Gemini-APIs kann Kosten verursachen.

Es wird empfohlen,

- zunächst kleine Testläufe durchzuführen,
- API-Schlüssel vertraulich zu behandeln,
- die Ergebnisdateien regelmäßig zu sichern.

---

# Verwendete Modelle

| Anbieter | Modell |
|----------|--------|
| OpenAI | GPT-4o |
| Google | Gemini 3.1 Pro Preview |

Die tatsächlich verwendeten Modellnamen werden zusätzlich in der Ergebnisdatei gespeichert.

---

# Autor

Dieses Projekt wurde im Rahmen des Moduls **Forschungsprojekt Teil B** entwickelt.

Es dient der Durchführung des Forschungsprojekts **„Mehrsprachige Reproduktion und Erweiterung des Moral-Machine-Experiments zur Evaluation moralischer Entscheidungen in Large Language Models“**.

Ziel des Projekts ist die automatisierte Durchführung und Auswertung mehrsprachiger Moral-Machine-Szenarien mit verschiedenen Large Language Models, um sprachliche Einflüsse sowie modellabhängige Unterschiede in moralischen Entscheidungen systematisch zu untersuchen.

