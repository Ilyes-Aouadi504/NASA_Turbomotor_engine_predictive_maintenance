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