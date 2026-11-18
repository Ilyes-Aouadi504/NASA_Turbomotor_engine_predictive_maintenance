import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from Data import SENSOR_COLS, SETTING_COLS, add_train_rul, constant_cols, load_split
FD = sys.argv[1] if len(sys.argv) > 1 else "FD001"
REPORTS = Path("predictive maintenance end to end").resolve().parent / "reports"
REPORTS.mkdir(exist_ok=True)
train, test, rul_test = load_split(FD)
train = add_train_rul(train)

life = train.groupby("unit")["cycle"].max()
dropped = constant_cols(train)
informative = [c for c in SENSOR_COLS if c not in dropped]

print("engines:", train["unit"].nunique())
print("missing values:", int(train.isna().sum().sum()))
print("lifetimes:\n", life.describe().round(1))
print("test trajectory lengths:\n", test.groupby("unit")["cycle"].max().describe().round(1))
print("constant columns:", dropped)
print("informative sensors:", informative)
rng = np.random.default_rng(0)
units = rng.choice(train["unit"].unique(), size=6, replace=False)
cols = informative[:12]
nrows = int(np.ceil(len(cols) / 3))
fig, axes = plt.subplots(nrows, 3, figsize=(14, 3 * nrows), sharex=True)
for ax, c in zip(np.ravel(axes), cols):
    for u in units:
        d = train[train["unit"] == u]
        ax.plot(-d["rul_raw"], d[c], lw=0.9, alpha=0.8)
    ax.set_title(c, fontsize=9)
fig.suptitle(f"{FD}: sensor drift as engines approach failure (x = -RUL)")
fig.tight_layout()
fig.savefig(REPORTS / "degradation.png", dpi=130)
plt.close(fig)
corr = train[informative + ["rul"]].corr()["rul"].drop("rul").sort_values()
fig, ax = plt.subplots(figsize=(6, 5))
ax.barh(corr.index, corr.values)
ax.set(title="Correlation with capped RUL")
fig.tight_layout()
fig.savefig(REPORTS / "corr_target.png", dpi=130)
plt.close(fig)
cm = train[informative].corr()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(cm)))
ax.set_xticklabels(cm.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(cm)))
ax.set_yticklabels(cm.columns, fontsize=7)
fig.colorbar(im)
ax.set_title("Sensor correlation")
fig.tight_layout()
fig.savefig(REPORTS / "corr_heatmap.png", dpi=130)
plt.close(fig)
d = train[train["unit"] == units[0]]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(d["cycle"], d["rul_raw"], label="true RUL")
ax.plot(d["cycle"], d["rul"], label="capped RUL (target)")
ax.set(xlabel="cycle", ylabel="RUL", title="Piece-wise linear target")
ax.legend()
fig.tight_layout()
fig.savefig(REPORTS / "rul_cap.png", dpi=130)
plt.close(fig)
print("Saved figures to", REPORTS)