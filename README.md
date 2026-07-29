# Mehrsprachiges Moral-Machine-Experiment mit Large Language Models

## Projektbeschreibung

Dieses Projekt dient der automatisierten Durchführung und Auswertung mehrsprachiger Moral-Machine-Experimente mit Large Language Models (LLMs). Identische moralische Dilemmaszenarien werden in drei Sprachen (Englisch, Deutsch und Türkisch) an verschiedene Sprachmodelle übermittelt.

Aktuell werden folgende Modelle unterstützt:

- OpenAI GPT-4o
- Google Gemini

Ziel des Projekts ist die Untersuchung,

- sprachlicher Einflüsse auf moralische Entscheidungen,
- modellabhängiger Unterschiede,
- der Konsistenz zwischen Sprachversionen sowie
- der Präferenzraten einzelner Moral-Machine-Dimensionen.

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
├── analyze_results.py
│
├── scenarios.csv
├── results.csv
│
└── analysis/
    ├── overview.csv
    ├── decisions_by_provider.csv
    ├── decisions_by_language.csv
    ├── provider_agreement_summary.csv
    ├── language_consistency_summary.csv
    └── moral_machine_dimensions.csv
```

---

# Installation

## Virtuelle Umgebung erstellen

```bash
python3 -m venv .venv
```

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

---

## Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

---

## Umgebungsvariablen konfigurieren

```bash
cp .env.example .env
```

Anschließend die API-Schlüssel eintragen.

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
|----------|-------------|
| scenario_id | eindeutige Szenario-ID |
| language | Sprache (`en`, `de`, `tr`) |
| dimension | Moral-Machine-Dimension |
| option_a | Beschreibung von Alternative A |
| option_b | Beschreibung von Alternative B |
| target_preference | Erwartete Präferenz |
| target_option | Entsprechende Option (`A` oder `B`) |

Die Spalten `dimension`, `target_preference` und `target_option` werden ausschließlich für die spätere wissenschaftliche Auswertung verwendet und werden **nicht** an die Sprachmodelle übermittelt.

---

## Sprachversionen

Jedes Szenario muss in drei Sprachversionen vorliegen:

- Englisch (`en`)
- Deutsch (`de`)
- Türkisch (`tr`)

Die Bedeutung der beiden Alternativen muss in allen Sprachen identisch sein.

Option A und Option B dürfen zwischen den Sprachversionen niemals vertauscht werden.

---

# Datensatz validieren

Vor jedem Experiment sollte der Datensatz überprüft werden.

```bash
python validate_dataset.py --input scenarios.csv
```

Geprüft werden unter anderem:

- Pflichtspalten
- leere Werte
- gültige Sprachcodes
- doppelte Einträge
- vollständige Sprachversionen

Beispiel:

```text
Datensatz gültig: 100 Szenarien, 300 Sprachversionen.
```

---

# Experiment durchführen

## Testlauf

Vor einem vollständigen Experiment empfiehlt sich ein kurzer Testlauf.

```bash
python run_experiment.py \
    --input scenarios.csv \
    --output results.csv \
    --limit 3
```

---

## Vollständiges Experiment

```bash
python run_experiment.py \
    --input scenarios.csv \
    --output results.csv
```

Optional können die Szenarien reproduzierbar zufällig sortiert werden.

```bash
python run_experiment.py \
    --input scenarios.csv \
    --output results.csv \
    --shuffle
```

Hierfür wird intern der Zufalls-Seed **42** verwendet.

---

# Ergebnisdatei

Während des Experiments werden alle Ergebnisse direkt in `results.csv` gespeichert.

## Wichtige Spalten

| Spalte | Beschreibung |
|----------|-------------|
| scenario_id | Szenario-ID |
| language | Sprache |
| provider | Anbieter |
| model | Modellname |
| decision | Entscheidung (`A` oder `B`) |
| raw_response | Originalantwort |
| status | Status der Anfrage |
| error | Fehlermeldung |
| timestamp_utc | Zeitpunkt |
| latency_seconds | Antwortzeit |
| prompt_version | Promptversion |
| python_version | Python-Version |
| operating_system | Betriebssystem |

---

## Statuswerte

### ok

Das Modell hat eindeutig mit **A** oder **B** geantwortet.

### invalid_response

Das Modell hat geantwortet, jedoch nicht ausschließlich mit **A** oder **B**.

### error

Beim API-Aufruf ist ein Fehler aufgetreten.

---

# Ergebnisse auswerten

Nach Abschluss des Experiments können die Ergebnisse automatisch ausgewertet werden.

```bash
python analyze_results.py \
    --input results.csv \
    --scenarios scenarios.csv \
    --output-dir analysis
```

---

## Erzeugte Auswertungsdateien

| Datei | Inhalt |
|---------|--------|
| overview.csv | Allgemeine Statistiken |
| decisions_by_provider.csv | Entscheidungen je Modell |
| decisions_by_language.csv | Entscheidungen je Sprache |
| provider_agreement_summary.csv | Übereinstimmung zwischen den Modellen |
| language_consistency_summary.csv | Konsistenz zwischen den Sprachversionen |
| moral_machine_dimensions.csv | Präferenzraten je Moral-Machine-Dimension |

---

# Prompt

Alle Modelle erhalten denselben Promptaufbau.

Die Instruktion wird lediglich in die jeweilige Sprache übersetzt.

Die Antwort darf ausschließlich aus

```
A
```

oder

```
B
```

bestehen.

Zusätzliche Erklärungen oder Begründungen sind nicht erlaubt.

---

# Reproduzierbarkeit

Zur Sicherstellung reproduzierbarer Experimente gelten folgende Regeln:

- identischer Prompt für alle Modelle
- identische Szenarien in allen Sprachen
- kein Chatverlauf
- reproduzierbare Zufallsreihenfolge (`Seed = 42`)
- erfolgreiche Antworten werden bei erneutem Start übersprungen
- Metadaten werden nicht an die Modelle übermittelt

---

# Hinweise zur API-Nutzung

Die Nutzung der OpenAI- und Google-APIs kann Kosten verursachen.

Es wird empfohlen,

- zunächst Testläufe durchzuführen,
- API-Schlüssel vertraulich zu behandeln,
- Ergebnisse regelmäßig zu sichern.

---

# Verwendete Modelle

Die tatsächlich verwendeten Modellnamen werden über die Datei `.env` festgelegt und zusätzlich in der Ergebnisdatei gespeichert.

Beispiel:

| Anbieter | Modell |
|-----------|---------|
| OpenAI | GPT-4o |
| Google | Gemini 3.1 Pro Preview |

---

# Autor

Dieses Projekt wurde im Rahmen des Moduls **Forschungsprojekt Teil B** an der **HTW Berlin** entwickelt.

Es unterstützt die Durchführung des Forschungsprojekts:

**„Mehrsprachige Reproduktion und Erweiterung des Moral-Machine-Experiments zur Evaluation moralischer Entscheidungen in Large Language Models“**

Ziel ist die automatisierte Durchführung und Auswertung mehrsprachiger Moral-Machine-Experimente, um sprachliche Einflüsse sowie Unterschiede zwischen verschiedenen Large Language Models systematisch zu untersuchen.