#import librairies ili nist7a9hom 
from pathlib import Path
import numpy as np
import pandas as pd
#importi data ta3ik ou fixi les colonnes
DataPath = Path(__file__).parent / "Raw Data"
INDEX_COLS=["unit","cycle"]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS
RUL_CAP = 125
#charger les données et les séparer en train et test
def load_split(fd="FD001"):
    kw = dict(sep=r"\s+", header=None)
    train = pd.read_csv(DataPath / f"train_{fd}.txt", names=ALL_COLS, **kw)
    test = pd.read_csv(DataPath / f"test_{fd}.txt", names=ALL_COLS, **kw)
    rul_test = pd.read_csv(DataPath / f"RUL_{fd}.txt", names=["rul"], **kw)
    return train, test, rul_test
#ajouter la colonne RUL aux données d'entraînement
def add_train_rul(df, cap=RUL_CAP):
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul_raw"] = max_cycle - df["cycle"]
    df["rul"] = df["rul_raw"].clip(upper=cap)
    return df
#identifier les colonnes constantes dans le dataframe
def constant_cols(df, threshold=0.01):
    cols = SETTING_COLS + SENSOR_COLS
    std = df[cols].std()
    return std[std < threshold].index.tolist()
if __name__ == "__main__":
    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        train, test, rul_test = load_split(fd)
        train = add_train_rul(train)
        print(f"\n=== {fd} ===")
        print("shapes:", train.shape, test.shape, rul_test.shape)
        print("engines:", train["unit"].nunique())
        print("missing values:", int(train.isna().sum().sum()))
        print("constant columns:", constant_cols(train))


