from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Ausgabeordner
analysis = Path("analysis")
figures = analysis / "figures"
figures.mkdir(parents=True, exist_ok=True)

plt.style.use("default")

# -----------------------------
# Abbildung 1
# Entscheidungen A/B
# -----------------------------
df = pd.read_csv(analysis / "decisions_by_provider.csv")

pivot = df.pivot(
    index="provider",
    columns="decision",
    values="percentage"
)

ax = pivot.plot(kind="bar", figsize=(6,4))

ax.set_ylabel("Anteil (%)")
ax.set_xlabel("")
ax.set_title("Entscheidungsverteilung nach Modell")

plt.tight_layout()
plt.savefig(figures / "figure_1_decisions.png", dpi=300)
plt.close()

# -----------------------------
# Abbildung 2
# Modellübereinstimmung
# -----------------------------
df = pd.read_csv(
    analysis / "provider_agreement_summary.csv"
)

df = df[df.language != "overall"]

ax = df.plot(
    x="language",
    y="agreement_rate",
    kind="bar",
    legend=False,
    figsize=(6,4)
)

ax.set_ylim(0,100)
ax.set_ylabel("Agreement (%)")
ax.set_xlabel("")
ax.set_title("Übereinstimmung der Modelle")

plt.tight_layout()
plt.savefig(figures / "figure_2_agreement.png", dpi=300)
plt.close()

# -----------------------------
# Abbildung 3
# Sprachkonsistenz
# -----------------------------
df = pd.read_csv(
    analysis / "language_consistency_summary.csv"
)

ax = df.plot(
    x="provider",
    y="consistency_rate",
    kind="bar",
    legend=False,
    figsize=(6,4)
)

ax.set_ylim(0,100)
ax.set_ylabel("Konsistenz (%)")
ax.set_xlabel("")
ax.set_title("Sprachkonsistenz")

plt.tight_layout()
plt.savefig(figures / "figure_3_consistency.png", dpi=300)
plt.close()

# -----------------------------
# Abbildung 4
# Moral-Machine-Dimensionen
# -----------------------------
df = pd.read_csv(
    analysis / "moral_machine_dimensions.csv"
)

df = df[df.language == "overall"]

pivot = df.pivot(
    index="dimension",
    columns="provider",
    values="preference_rate"
)

ax = pivot.plot(
    kind="bar",
    figsize=(10,5)
)

ax.set_ylim(0,100)
ax.set_ylabel("Präferenzrate (%)")
ax.set_xlabel("")
ax.set_title("Präferenzraten nach Moral-Machine-Dimension")

plt.tight_layout()
plt.savefig(figures / "figure_4_dimensions.png", dpi=300)
plt.close()

print("Alle Abbildungen wurden erstellt.")