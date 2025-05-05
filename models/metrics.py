import numpy as np
import pandas as pd
from typing import Any
from statsmodels.tsa.stattools import acf


def correlation(stations: pd.DataFrame) -> Any:
    corr = stations.corr()
    print("Initial correlation matrix:\n", corr)  # Print initial correlation matrix
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= 0.9:
                print(f"Correlation between {stations.columns[i]} and {stations.columns[j]} is {corr.iloc[i, j]}, marking {stations.columns[j]} for removal")
                if columns[j]:
                    columns[j] = False
    print("Columns mask after filtering:", columns)
    selected_columns = stations.columns[columns]
    print("Selected columns after filtering:", selected_columns)
    stations = stations[selected_columns]
    return np.abs(stations.corr())

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast-actual)/np.abs(actual))
    me = np.mean(forecast-actual)
    mae = np.mean(np.abs(forecast-actual))
    mpe = np.mean((forecast-actual)/actual)
    rmse = np.mean((forecast-actual)**2)**0.5
    corr = np.corrcoef(forecast, actual)[0,1]
    nse = 1 - (np.sum((actual - forecast) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
    mins = np.amin(np.hstack([forecast[:,None], actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)
    acfe = acf(forecast - actual)[1]
    return ({'mape':mape, 'me':me, 'mae':mae, 'mpe':mpe, 'rmse':rmse, 'corr':corr, 'nse':nse, 'minmax':minmax, 'acfe':acfe})