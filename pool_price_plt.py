from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '7.5'

DATA_FILE = Path(__file__).with_name("P&A Table.csv")


def clean_price(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.str.replace(r"[$,]", "", regex=True))


df = pd.read_csv(DATA_FILE, sep="\t", encoding="utf-16")
df["Date - MST"] = pd.to_datetime(df["Date - MST"])
df["Avg. Price"] = clean_price(df["Avg. Price"])
df["Avg. Gas Price"] = clean_price(df["Avg. Gas Price"])

fig, ax = plt.subplots(figsize=(5, 3.5))
ax2 = ax.twinx()

ax.plot(df["Date - MST"], df["Avg. Price"], color="blue", label="Avg. Price", linewidth=1)
ax2.plot(df["Date - MST"], df["Avg. Gas Price"], color="black", label="Avg. Gas Price", linewidth=1)

ax.set_xlabel("Year")
ax.set_ylabel("Avg. Electricity Price")
ax2.set_ylabel("Avg. Natural Gas Price")

lines_1, labels_1 = ax.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
fig.autofmt_xdate()
fig.tight_layout()

plt.savefig("pool_price_plot.png", dpi=1200, bbox_inches="tight")
plt.show()
