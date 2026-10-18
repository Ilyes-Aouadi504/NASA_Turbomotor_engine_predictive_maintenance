import numpy as np
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))
def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))
def nasa_score(y_true, y_pred):
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
if __name__ == "__main__":
    y_true = [100, 50]
    y_pred = [90, 60]      # one early by 10, one late by 10
    print(rmse(y_true, y_pred), mae(y_true, y_pred), nasa_score(y_true, y_pred))