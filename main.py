import pandas as pd
import numpy as np
from datetime import datetime
import models
import os
import utilis
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import plotly.io as pio

def invTransform(scaler, data, colName, colNames):
    dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
    dummy[colName] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
    return dummy[colName].values

# command for vscode to view all the data at terminal
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)


sample = 'box2'  # Can be 'box1' or 'box2', please ensure depth is zero for box1
depth = 1.7  # depth is in meters can be 0, 0.6, 0.9, or 1.7


net_rainfall = False  # Can be True or False

def main():
    model_name = 'PROPHET'
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
            excel_file_path = "./data/box1_net_rainfall.xlsx"
            target_sheet_name = 'Sheet1'
            regressors = ['Temperature', 'Rainfall']
        
        else:
            excel_file_path = "./data/box1.xlsx"
            target_sheet_name = 'Sheet1'
            regressors = ['Temperature', 'Rainfall']
            # regressors = ['Temperature']
        
    elif sample == 'box2':
        if depth == 0.6:
            excel_file_path = "./data/box2_depth6.xlsx"
            target_sheet_name = 'Sheet1'
            regressors = ['Soil Temperature', 'Water Content', 'Bulk EC']
        elif depth == 0.9:
            excel_file_path = "./data/box2_depth9.xlsx"
            target_sheet_name = 'Sheet1'
            regressors = ['Soil Temperature', 'Water Content', 'Bulk EC']
        elif depth == 1.7:
            excel_file_path = "./data/box2_depth17.xlsx"
            target_sheet_name = 'Sheet1'
            regressors = ['Soil Temperature', 'Water Content', 'Bulk EC']    
        else:
            raise ValueError("Invalid depth for box2. Must be one of [0.6, 0.9, 1.7]")
    else:
        raise ValueError("Invalid sample. Must be 'box1' or 'box2'.")

    # Read data from the specified sheet
    df = pd.read_excel( excel_file_path, sheet_name=target_sheet_name, engine='openpyxl', nrows=620)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.ffill()

    plt.plot(df['ds'], df['y'])
    plt.show()

    caos = np.random.randint(100, size=(len(df)))

    # Select the appropriate columns for training based on the model
    if model_name in ['PROPHET', 'NEURALPROPHET']:
        dftraining = df[['ds', 'y'] + regressors].copy()
    else:
        dftraining = df[['y'] + regressors].copy()
    
    sz = round(len(df)/100*10)
    dftrain = dftraining[:- sz].copy()
    dftest = dftraining[-sz:].copy()

    # shiftday = len(dftrain) // 10
    shiftday = len(dftest)

    if(norm):
        if model_name in ['ARIMA','SARIMAX']:
            scaler.fit(dftraining.iloc[:])
            dftrain.iloc[:] = scaler.transform(dftrain.iloc[:]).copy()
            dftest.iloc[:] = scaler.transform(dftest.iloc[:]).copy()
        else:
            scaler.fit(dftraining.iloc[:,1:])
            dftrain.iloc[:,1:] = scaler.transform(dftrain.iloc[:,1:]).copy()
            dftest.iloc[:,1:] = scaler.transform(dftest.iloc[:,1:]).copy()

    if sample == 'box1':
        # correlation_table = models.correlation(dftraining[['y','Temperature', 'Rainfall']])
        correlation_table = models.correlation(dftraining[['y','Temperature']])
    elif sample == 'box2':
        correlation_table = models.correlation(dftraining[['y','Soil Temperature', 'Bulk EC', 'Water Content']])

    utilis.instance_path.set_path(
    os.path.join(os.path.dirname(__file__), 'output', model_name),
    'tuning_' + model_name + 'sampel_' + sample + 'netrainfall_' + str(net_rainfall) + 'depth_' + str(depth) + '_norm_' + str(norm) + '_cross_valid_' + str(cross_valid))

    if cross_valid:
        if model_name == 'ARIMA':
            [best_params, best_model_df, hyper_parameter_table_results_list] = models.arima_tuning_workflow(None, dftrain, shiftday=shiftday)
        elif model_name == 'SARIMAX':
            [best_params, best_model_df, hyper_parameter_table_results_list] = models.sarimax_tuning_workflow(None, dftrain, shiftday=shiftday)
        elif model_name == 'PROPHET':
            [best_params, best_model_df, hyper_parameter_table_results_list] = models.prophet_tuning_workflow(None, dftrain, shiftday=shiftday)
        elif model_name == 'NEURALPROPHET':
            [best_params, best_model_df, hyper_parameter_table_results_list] = models.neuralprophet_tuning_workflow([], dftrain, shiftday=shiftday)


        utilis.plot_flow(correlation_table, tuning = True, best_params=best_params, best_model=best_model_df, hyper_parameter_table_results=hyper_parameter_table_results_list)
        
    else:
        if model_name == 'ARIMA':
            model_obj = models.model_arima(dftrain, test_set=dftest)
            model_obj.train()
            forecast = model_obj.forecast(dftest)
            forecast.rename(columns={'yhat': 'y'})
        if model_name == 'SARIMAX':
            model_obj = models.model_sarimax(dftrain, test_set=dftest)
            model_obj.train()
            forecast = model_obj.forecast(dftest)
            forecast.rename(columns={'yhat': 'y'})

        elif model_name == 'PROPHET':
            model_obj = models.model_prophet(dftrain, test_set=dftest)
            if sample == 'box1':
                model_obj.add_regressor('Temperature', models.hyper_parameters.regressor_weight['Temperature'], 'additive')
                model_obj.add_regressor('Rainfall', models.hyper_parameters.regressor_weight['Rainfall'], 'additive')
            elif sample == 'box2':
                model_obj.add_regressor('Soil Temperature', models.hyper_parameters.regressor_weight['Soil Temperature'], 'additive')
                model_obj.add_regressor('Water Content', models.hyper_parameters.regressor_weight['Water Content'], 'additive')
                model_obj.add_regressor('Bulk EC', models.hyper_parameters.regressor_weight['Bulk EC'], 'additive')
            model_obj.train()
            future = dftest.copy()
            future['y'] = np.nan
            forecast = model_obj.forecast(future)

        elif model_name == 'NEURALPROPHET':
            validation_split_index = int(len(dftrain) * 0.9)
            train_initial = dftrain.copy()
            train_end = dftest.copy()

            if sample == 'box1':
                model_obj = models.model_neuralprophet(
                    train_set=train_initial,
                    validation_set=train_end,
                    test_set=None,
                    kwargs=models.hyper_parameters.get_neuralprophet_hyperparameters()
                ).add_regressor(
                    'Temperature', models.hyper_parameters.regressor_weight['Temperature'], 'additive'
                )
                model_obj.train()
                regressors = dftest.copy()
                regressors['y'] = pd.Series([float('nan')] * len(dftest))
                future = model_obj.get_model().make_future_dataframe(
                    train_initial, periods=len(dftest), regressors_df=regressors, n_historic_predictions=False
                )
                forecast = model_obj.forecast(future)
                forecast = models.exctract_yhat(forecast, size=len(dftest))
                
            
            elif sample == 'box2':
                model_obj = models.model_neuralprophet(
                    train_set=train_initial,
                    validation_set=train_end,
                    test_set=None,
                    kwargs=models.hyper_parameters.get_neuralprophet_hyperparameters()
                ).add_regressor(
                    'Soil Temperature', models.hyper_parameters.regressor_weight['Soil Temperature'], 'additive'
                ).add_regressor(
                    'Water Content', models.hyper_parameters.regressor_weight['Water Content'], 'additive'
                ).add_regressor(
                    'Bulk EC', models.hyper_parameters.regressor_weight['Bulk EC'], 'additive'
                )
                model_obj.train()
                regressors = dftest.copy()
                regressors['y'] = pd.Series([float('nan')] * len(dftest))
                future = model_obj.get_model().make_future_dataframe(
                    train_initial, periods=len(dftest), regressors_df=regressors, n_historic_predictions=False
                )
                forecast = model_obj.forecast(future)
                forecast = models.exctract_yhat(forecast, len(dftest))
            
            
        if norm:
            if model_name in ['ARIMA', 'SARIMAX']:
                dftrain.iloc[:, :] = scaler.inverse_transform(dftrain.iloc[:, :]).copy()
                dftest.iloc[:, :] = scaler.inverse_transform(dftest.iloc[:, :]).copy()
                forecast.iloc[:, 1:] = scaler.inverse_transform(forecast.iloc[:, 1:]).copy()
            
            elif model_name in ['PROPHET']:
                dftrain.iloc[:,1:] = scaler.inverse_transform(dftrain.iloc[:,1:]).copy()
                dftest.iloc[:,1:] = scaler.inverse_transform(dftest.iloc[:,1:]).copy()
                forecast['yhat'] = invTransform(scaler, forecast['yhat'],'y',scaler.feature_names_in_)
                forecast['yhat_lower']=invTransform(scaler, forecast['yhat_lower'],'y',scaler.feature_names_in_)
                forecast['yhat_upper']=invTransform(scaler, forecast['yhat_upper'],'y',scaler.feature_names_in_)
            
            elif model_name in ['NEURALPROPHET']:
                dftrain.iloc[:,1:] = scaler.inverse_transform(dftrain.iloc[:,1:]).copy()
                dftest.iloc[:,1:] = scaler.inverse_transform(dftest.iloc[:,1:]).copy()
                forecast['yhat1'] = invTransform(scaler, forecast['yhat1'],'y',scaler.feature_names_in_)



        if model_name in ['ARIMA','SARIMAX']:
            dftrain['ds'] = df['ds'].iloc[:len(dftrain)]
            dftest['ds'] = df['ds'].iloc[len(dftrain):]
            forecast.index = dftest.index
            forecast['ds'] = dftest['ds'].values

        utilis.plot_flow(correlation_table, tuning = False, forecast = forecast, dftraining = dftrain, dftest = dftest, model_name = model_name.lower(), shift_data=False, model = model_obj)  
        
        namefolder = utilis.instance_path.get_path()

        if model_name in ['ARIMA','SARIMAX']:           
            forecast.to_excel(f'{namefolder}/selected_forecast_data.xlsx', index=False)
           
        elif model_name in ['PROPHET']:
            if sample == 'box1':
                forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower', 'Temperature', 'Rainfall']].to_excel(os.path.join(namefolder, 'selected_forecast_data.xlsx'), index=False)
            elif sample == 'box2':
                forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower', 'Soil Temperature', 'Water Content', 'Bulk EC']].to_excel(os.path.join(namefolder, 'selected_forecast_data.xlsx'), index=False)
            
        elif model_name in ['NEURALPROPHET']:
            if sample == 'box1':
                forecast[['ds', 'yhat1', 'Temperature', 'Rainfall']].to_excel(os.path.join(namefolder, 'selected_forecast_data.xlsx'), index=False)
            elif sample == 'box2':
                forecast[['ds', 'yhat1', 'Soil Temperature', 'Water Content', 'Bulk EC']].to_excel(os.path.join(namefolder, 'selected_forecast_data.xlsx'), index=False)

        '''plot con normalizzazione scaler'''
        f,ax2=plt.subplots(1)
        plt.plot(dftrain['ds'], (dftrain['y']),'k.',label='Training')
        plt.plot(dftest['ds'], (dftest['y']),'r-', label='Validation')
        if model_name in ['ARIMA','SARIMAX', 'PROPHET']:
            plt.plot(dftest['ds'], forecast['yhat'], ls='-', c='#0072B2', label='Prediction')
        else:
            plt.plot(dftest['ds'], forecast['yhat1'], ls='-', c='#0072B2', label='Prediction')
       

if __name__ == '__main__':
    main()

