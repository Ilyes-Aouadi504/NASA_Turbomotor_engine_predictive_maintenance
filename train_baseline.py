import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from data import RUL_CAP, SENSOR_COLS, SETTING_COLS, add_train_rul, constant_cols, load_split
from features import add_rolling
from metrics import mae, nasa_score, rmse
SEED = 42
ML_DIR = Path("predictive maintenance end to end").resolve().parent
MODELS = {
    "linear_regression": (lambda **p: make_pipeline(StandardScaler(), LinearRegression()), {}),
    "random_forest": (RandomForestRegressor, dict(
        n_estimators=200, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
    "xgboost": (XGBRegressor, dict(
        n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, n_jobs=-1, random_state=SEED)),
}
def build_dataset(fd, rolling):
    train, test, rul_test = load_split(fd)
    train = add_train_rul(train)
    dropped = constant_cols(train)
    base = [c for c in SETTING_COLS + SENSOR_COLS if c not in dropped]
    if rolling:
        train, feats = add_rolling(train, base)
        test, _ = add_rolling(test, base)
    else:
        feats = base
    test_last = test.sort_values(["unit", "cycle"]).groupby("unit").tail(1)
    y_test = np.minimum(rul_test["rul"].to_numpy(), RUL_CAP)
    return train, test_last, feats, y_test, dropped
