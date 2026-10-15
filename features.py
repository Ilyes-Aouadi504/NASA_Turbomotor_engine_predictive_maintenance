import pandas as pd
def add_rolling(df, base_cols, windows=(5, 10)):
    df = df.sort_values(["unit", "cycle"]).copy()
    grouped = df.groupby("unit")[base_cols]
    new = {}
    for w in windows:
        roll = grouped.rolling(w, min_periods=1)
        mean = roll.mean().reset_index(level=0, drop=True)
        std = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        for c in base_cols:
            new[f"{c}_mean{w}"] = mean[c]
            new[f"{c}_std{w}"] = std[c]
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return df, base_cols + list(new)
if __name__ == "__main__":
    from Data import SENSOR_COLS, SETTING_COLS, add_train_rul, constant_cols, load_split

    train, _, _ = load_split("FD001")
    train = add_train_rul(train)
    base = [c for c in SETTING_COLS + SENSOR_COLS if c not in constant_cols(train)]

    out, feats = add_rolling(train, base)
    print("base features:", len(base), "| total features:", len(feats), "| table shape:", out.shape)

    # leakage check: first row of every engine must equal its own raw value
    first = out.groupby("unit").head(1)
    print("no leakage between engines:", (first["s_2_mean5"] == first["s_2"]).all())