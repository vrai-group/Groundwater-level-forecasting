import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler

sample = 'box1'  # Can be 'box1' or 'box2', please ensure depth is zero for box1
depth = 0  # Depth is in meters can be 0, 0.6, 0.9, or 1.7
net_rainfall = False  # Can be True or False

model_name = 'NEURALPROPHET'
norm = True
cross_valid = False
print('Model:', model_name)
print('Sample:', sample)
print('Normalization:', norm)
print('Cross Validation:', cross_valid)
print('Depth:', depth)
print('Net Rainfall:', net_rainfall)

np.random.seed(123)
scaler = MinMaxScaler()

# Set file path and sheet name based on the sample
if sample == 'box1':
    if net_rainfall:
        excel_file_path = "/data/box1_net_rainfall.xlsx"
        target_sheet_name = 'Sheet1'
        regressors = ['Temperature', 'Rainfall']
    else:
        excel_file_path = "/data/box1.xlsx"
        target_sheet_name = 'Sheet1'
        regressors = ['Temperature', 'Rainfall']
elif sample == 'box2':
    if depth == 0.6:
        excel_file_path = "/data/box2_depth6.xlsx"
        target_sheet_name = 'Sheet1'
        regressors = ['Soil Temperature', 'Water Content']
    elif depth == 0.9:
        excel_file_path = "/data/box2_depth9.xlsx"
        target_sheet_name = 'Sheet1'
        regressors = ['Soil Temperature', 'Water Content']
    elif depth == 1.7:
        excel_file_path = "/data/box2_depth17.xlsx"
        target_sheet_name = 'Sheet1'
        regressors = ['Soil Temperature', 'Water Content']
    else:
        raise ValueError("Invalid depth for box2. Must be one of [0.6, 0.9, 1.7]")
else:
    raise ValueError("Invalid sample. Must be 'box1' or 'box2'.")

# Read data from the specified sheet
df = pd.read_excel(os.path.dirname(__file__) + excel_file_path, sheet_name=target_sheet_name, engine='openpyxl', nrows=620)
df['ds'] = pd.to_datetime(df['ds'])
df = df.ffill()

# Select the appropriate columns for training based on the model
if model_name in ['PROPHET', 'NEURALPROPHET']:
    dftraining = df[['ds', 'y'] + regressors].copy()
else:
    dftraining = df[['y'] + regressors].copy()

sz = round(len(df) / 100 * 10)
dftrain = dftraining[:-sz].copy()
dftest = dftraining[-sz:].copy()

forecast_base_path = os.path.join(os.path.dirname(__file__), 'forecast_data')

models = ['ARIMA', 'PROPHET', 'SARIMAX', 'NEURALPROPHET']
depths = ['0', '06', '09', '17']

for depth in depths:
    plt.figure(figsize=(14, 7))
    plt.plot(dftrain['ds'], dftrain['y'], 'k-', label='Training')

    for model in models:
        forecast_data_path = os.path.join(forecast_base_path, model, depth, 'selected_forecast_data.xlsx')
        forecast_data = pd.read_excel(forecast_data_path, engine='openpyxl')
        forecast_data['ds'] = pd.to_datetime(forecast_data['ds'])
        yhat_column = 'yhat1' if model == 'NEURALPROPHET' else 'yhat'
        plt.plot(forecast_data['ds'], forecast_data[yhat_column], label=f'{model.capitalize()}')

    plt.plot(dftest['ds'], dftest['y'], 'r-', linewidth=3, label='Validation')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(False)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=5))  # Set xticks every 3 months
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    save_folder = '/Users/gagan/Desktop/ITALY RESEARCH OFFLINE/MetroLivEnv/IAH_Water/merge_plot/plot'
    plt.savefig(os.path.join(save_folder, f'performance_comparison_depth_{depth}.pdf'))
