from typing import Any
from concurrent.futures import ProcessPoolExecutor as Pool
import functools
import json
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet, set_random_seed
from prophet import Prophet
import statsmodels.api as sm
import itertools
from prophet.diagnostics import cross_validation, performance_metrics
import sys
import os
import time
from main import sample

class hyper_parameters:
    json_model, default_flag = [Any, Any]
    prophet = {
        'seasonality_mode':'multiplicative',
        'seasonality_prior_scale':70,
        'changepoint_range': 0.95,
        'changepoint_prior_scale':0.01
    }


    neuralprophet = {
        'yearly_seasonality':True,
        'weekly_seasonality':False,
        'daily_seasonality':False,
        'seasonality_mode':'additive',
        'seasonality_reg': 1,
        'n_lags':30,
        'n_forecasts':12,
        'impute_missing' : True,
        'num_hidden_layers': 5,
        'optimizer':'AdamW',
        'ar_reg': 0.5
    }

    sarimax = {
        'enforce_stationarity':True,
        'enforce_invertibility':True,
        #'freq':'D', 
        'seasonal_order':(0, 0, 2, 3),
        'order': (0, 0, 0)
    }

    arima = {
        'enforce_stationarity':True,
        'enforce_invertibility':True,
        'freq':'D', 
        'order':(1,1,1)
    }

    # regressor_weight = {
    #     'Temperature': 1
    # }
    
    regressor_weight = {
        'Temperature': 1,
        'Rainfall': 0,
        'Soil Temperature': 1,
        'Water Content': 1,
        'Bulk EC' : 1
    }

    @staticmethod
    def load_json_model():
        with open(os.path.dirname(__file__)+'/model.json') as file:
            hyper_parameters.json_model = json.load(file)
            hyper_parameters.default_flag = hyper_parameters.json_model['default-settings']
        
    @staticmethod
    def get_prophet_hyperparameters():
        hyper_parameters.load_json_model()
        if hyper_parameters.default_flag :
            return hyper_parameters.prophet
        else:
            return hyper_parameters.json_model['custom_hyper_parameters']['prophet']  
          
    @staticmethod
    def get_prophet_tuning_grid():
        hyper_parameters.load_json_model()
        return hyper_parameters.json_model['hyper-parameters-tuning']['prophet']['grid']

    @staticmethod
    def get_prophet_tuning_settings():
        hyper_parameters.load_json_model()
        return hyper_parameters.json_model['hyper-parameters-tuning']['prophet']['cross_validation'],\
            dict({'usage':hyper_parameters.json_model['hyper-parameters-tuning']['dataset-usage']})
    
    @staticmethod
    def get_neuralprophet_hyperparameters():
        hyper_parameters.load_json_model()
        if hyper_parameters.default_flag :
            return hyper_parameters.neuralprophet
        else:
            return hyper_parameters.json_model['custom_hyper_parameters']['neuralprophet']

    @staticmethod
    def get_neuralprophet_tuning_grid():
        hyper_parameters.load_json_model()
        return hyper_parameters.json_model['hyper-parameters-tuning']['neuralprophet']['grid']

    @staticmethod
    def get_neuralprophet_tuning_settings():
        hyper_parameters.load_json_model()
        return hyper_parameters.json_model['hyper-parameters-tuning']['neuralprophet']['cross_validation'],\
            dict({'usage':hyper_parameters.json_model['hyper-parameters-tuning']['dataset-usage']})
    
    @staticmethod
    def merge_dict(dictionary_s, dictionary_r):
        return {**dictionary_s,**dictionary_r}

    @staticmethod
    def get_combination_from_dict(dictionary):
        return [dict(zip(dictionary.keys(), v)) for v in itertools.product(*dictionary.values())]
    
    @staticmethod
    def get_arima_hyperparameters():
        hyper_parameters.load_json_model()
        if hyper_parameters.default_flag :
            return hyper_parameters.arima
        else:
            return hyper_parameters.json_model['custom_hyper_parameters']['arima']
        
    @staticmethod
    def get_arima_tuning_grid():
        hyper_parameters.load_json_model()
        return hyper_parameters.json_model['hyper-parameters-tuning']['arima']["grid"]
    
    @staticmethod
    def get_arima_tuning_settings():
        hyper_parameters.load_json_model()
        return hyper_parameters.json_model['hyper-parameters-tuning']['arima']['cross_validation'],\
            dict({'usage':hyper_parameters.json_model['hyper-parameters-tuning']['dataset-usage']})
    
    @staticmethod
    def get_sarimax_hyperparameters():
        hyper_parameters.load_json_model()
        if hyper_parameters.default_flag :
            return hyper_parameters.sarimax
        else:
            return hyper_parameters.json_model['custom_hyper_parameters']['sarimax']
        
    @staticmethod
    def get_sarimax_tuning_grid():
        hyper_parameters.load_json_model()
        return hyper_parameters.json_model['hyper-parameters-tuning']['sarimax']["grid"]
    
    @staticmethod
    def get_sarimax_tuning_settings():
        hyper_parameters.load_json_model()
        return hyper_parameters.json_model['hyper-parameters-tuning']['sarimax']['cross_validation'],\
            dict({'usage':hyper_parameters.json_model['hyper-parameters-tuning']['dataset-usage']})

    
class forecast_model:
    def __init__(self, train_set : pd.DataFrame, validation_set : pd.DataFrame, test_set : pd.DataFrame = None) -> None:
        self.train_set = train_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.fitted = None

    def get_model(self) -> Any :
        return self.model

    def get_train_data(self) -> pd.DataFrame :
        return self.train_set
    
    def get_validation_data(self) -> pd.DataFrame :
        return self.validation_set
    
class model_arima(forecast_model):
    def __init__(self,train_set : pd.DataFrame, \
                test_set : pd.DataFrame = None, kwargs=hyper_parameters.get_arima_hyperparameters()) -> Any:
        super().__init__(train_set, test_set)
        self.model = self.init_model(**kwargs)
    
    def init_model(self, **kwargs) -> Any :
        train_data = self.get_train_data().copy()
        endog_data = train_data[['y']]
        exog_data = train_data.drop(columns=['y'])
        return sm.tsa.arima.ARIMA(endog=endog_data, exog=exog_data, **kwargs)
    
    def train(self, **options) -> None :
        self.fitted = self.model.fit()
        return self

    def forecast(self, days) -> Any :
        exog_data = days.drop(columns=['y'])
        s_forecast = self.fitted.forecast(steps=len(days), exog=exog_data, alpha=0.05)
        if sample =='box1':
            return pd.DataFrame({'ds': s_forecast.index, 'yhat': s_forecast.reset_index(drop=True), 'Temperature': exog_data['Temperature'].reset_index(drop=True), 'Rainfall': exog_data['Rainfall'].reset_index(drop=True)})
        elif sample == 'box2':
            return pd.DataFrame({'ds': s_forecast.index, 'yhat': s_forecast.reset_index(drop=True), 'Soil Temperature': exog_data['Soil Temperature'].reset_index(drop=True), 'Water Content': exog_data['Water Content'].reset_index(drop=True), 'Bulk EC': exog_data['Bulk EC'].reset_index(drop=True)})


class model_sarimax(model_arima):
    def __init__(self, train_set: pd.DataFrame, test_set: pd.DataFrame = None, kwargs=hyper_parameters.get_sarimax_hyperparameters()) -> Any:
        super().__init__(train_set, test_set, kwargs)
        self.model = self.init_model(**kwargs)
    
    def init_model(self, **kwargs) -> Any:
        train_data = self.get_train_data().copy()
        endog_data = train_data[['y']]
        exog_data = train_data.drop(columns=['y'])
        return sm.tsa.SARIMAX(endog=endog_data, exog=exog_data, **kwargs, initialization='approximate_diffuse')

class model_prophet(forecast_model):
    def __init__(self, train_set: pd.DataFrame, test_set: pd.DataFrame = None,\
                  kwargs=hyper_parameters.get_prophet_hyperparameters()) -> Any:
        super().__init__(train_set, test_set)
        self.model = self.init_model(**kwargs)
    
    def init_model(self, **kwargs) -> Any:
        return Prophet(**kwargs)
    
    def train(self) -> Any :
        self.fitted = self.model.fit(self.get_train_data(), algorithm='Newton')
        return self

    def forecast(self, days, **options) -> Any :
        return self.fitted.predict(days)

    def add_regressor(self, name, prior_scale, mode) -> None :
        self.model.add_regressor(name, prior_scale=prior_scale, mode=mode)
        return self
    
    def add_seasonality(self, name, period, fourier_order) -> None :
        self.model.add_seasonality(name=name, period=period, fourier_order=fourier_order)
        return self

class model_neuralprophet(forecast_model):
    def __init__(self, train_set: pd.DataFrame, validation_set: pd.DataFrame, test_set: pd.DataFrame = None,\
                  kwargs=hyper_parameters.get_neuralprophet_hyperparameters()) -> Any:
        super().__init__(train_set, validation_set, test_set)
        self.model = self.init_model(**kwargs)
        set_random_seed(0)

    def init_model(self, **kwargs) -> Any :
        return NeuralProphet(**kwargs)

    def train(self) -> Any :
        self.fitted = self.model.fit(df=self.get_train_data(), \
            validation_df=self.get_validation_data(), epochs=100, batch_size=32, learning_rate=1e-3, \
            early_stopping=True, metrics=True)
        return self
    
    def forecast(self, days) -> Any :
        return self.model.predict(days)
    '''option and name di add seasonality da rivedere'''
    def add_regressor(self, name, options, mode) -> Any :
        self.model.add_future_regressor(name, mode=mode)
        return self
    
    def add_seasonality(self, name, period, fourier_order) -> Any :
        self.model.add_seasonality(name='np_yearly', period=period, fourier_order=fourier_order)
        return self

def generate_future(model_name, model_obj, dftraining, dftest, shift_data=None, shift_input=None, shiftday=None, stations=None):
    if model_name == 'prophet':
        future = model_obj.get_model().make_future_dataframe(periods=shiftday)
    elif model_name == 'neuralprophet':
        regressors = dftest.copy()
        regressors['y'] = pd.Series([float('nan')]*len(dftest))        
        future = model_obj.get_model().make_future_dataframe(dftraining, periods=shiftday, regressors_df=regressors, n_historic_predictions=True)
        return future
    if shift_data and shift_input:
        for col in stations.columns[:-2]:
            future[col]=''
            for i in range(0, len(future)):
                future[col][i] = stations[col][shiftday+i]
    else:
        for col in stations.columns[:-2]:
            future[col]=''
            future[col] = stations[col]
    return future.replace(np.nan,0)

def generate_regressor_future(dftraining, dftest):
    model_obj = model_prophet(dftraining, test_set=dftest)
    model_obj.train()
    return model_obj, model_obj.forecast(dftest)

def forecast_workflow(model_name, stations_dataframe_list, stations, dftraining, dftest, shiftday, shift_data, shift_input, stationsflag, esinoflag, gorgovivoflag):
    if model_name == 'PROPHET':
        model_obj = model_prophet(dftraining, test_set=dftest)
    elif model_name == 'NEURALPROPHET':
        df_test_lenght = len(dftraining)
        split_idx = int(df_test_lenght * 0.9)
        dftrain = dftraining.iloc[:split_idx]
        dfvalidation = dftraining.iloc[split_idx:]
        model_obj = model_neuralprophet(dftrain, validation_set=dfvalidation, test_set=dftest)
    elif model_name == 'ARIMA':
        model_obj = model_arima(dftraining, test_set=dftest)
    elif model_name == 'SARIMAX':
        model_obj = model_sarimax(dftraining, test_set=dftest)
    if model_name == 'PROPHET' or model_name == 'NEURALPROPHET':
        model_obj.add_seasonality('yearly', shiftday, 5)
        ''' aggiungo regressori con priorità specificata '''
        if stationsflag:
            for dataframe in stations_dataframe_list:
                model_obj.add_regressor(dataframe['quadrante_id'][0], hyper_parameters.regressor_weight['stations'], 'additive')
        if esinoflag:
            model_obj.add_regressor('esino', hyper_parameters.regressor_weight['esino'], 'additive')
        if gorgovivoflag:
            model_obj.add_regressor('gorgovivo', hyper_parameters.regressor_weight['portata'], 'additive')
        
        model_obj.train()
        future = generate_future(model_name, model_obj, dftraining, dftest, shift_data, shift_input, shiftday, stations)
        forecast = model_obj.forecast(future)
        fig = model_obj.get_model().plot(forecast)
        fig.savefig(f'forecast2_{model_name}.jpg')
        #forecast = forecast[['ds','yhat365']][-365:]
        #forecast.rename(columns={'yhat365':'yhat1'}, inplace=True)
        forecast = exctract_yhat(forecast, len(dftest))
        
    elif model_name == 'ARIMA' or model_name == 'SARIMAX':
        model_obj.train()
        forecast = model_obj.forecast(dftest)
    return model_obj, forecast

def prophet_tuning_workflow(stations_dataframe_list, dftraining, shiftday):
    grid = hyper_parameters.get_prophet_tuning_grid()
    cross_val_settings, usage = hyper_parameters.get_prophet_tuning_settings()
    setting_grid = hyper_parameters.get_combination_from_dict(hyper_parameters.merge_dict(cross_val_settings, usage))
    print(f'n° models settings : {len(setting_grid)}')
    mapes_list, best_model_df, hyper_parameter_table_results_list = [list(), pd.DataFrame(), dict()]
    for setting in setting_grid:
        num_rows = int(len(dftraining) * (1-setting['usage']))
        training_subset = dftraining.loc[num_rows:]
        kwargs = dict({'period':setting['period'], 'horizon':setting['horizon']})
        bm, hptr = prophet_hyperparameter_tuning(stations_dataframe_list, training_subset, shiftday, grid, **kwargs)
        best_model_df = pd.concat([best_model_df, bm])
        key = json.dumps(setting)
        hyper_parameter_table_results_list[key] = hptr
        mapes_list.append(hptr.loc[np.argmin(hptr['mape'])]['mape'])
        print(f'End cycle using : {setting["usage"]} of the dataset')
    best_model_df['setting'] = setting_grid
    best_model_df['mape'] = mapes_list
    best_params = best_model_df.iloc[np.argmin(mapes_list)]
    return best_params, best_model_df, hyper_parameter_table_results_list

def prophet_hyperparameter_tuning(stations_dataframe_list, dftraining, shiftday, param_grid, **kwargs):
    all_params = hyper_parameters.get_combination_from_dict(param_grid)
    print(f'n° models : {len(all_params)}')
    mapes = list() 
    
    for params in all_params:
        if sample == 'box1':
            m = model_prophet(dftraining, kwargs=params).\
                add_regressor('Temperature', hyper_parameters.regressor_weight['Temperature'], 'additive').\
                add_regressor('Rainfall', hyper_parameters.regressor_weight['Rainfall'], 'additive')
        elif sample == 'box2':
            m = model_prophet(dftraining, kwargs=params).\
                add_regressor('Soil Temperature', hyper_parameters.regressor_weight['Soil Temperature'], 'additive').\
                add_regressor('Water Content', hyper_parameters.regressor_weight['Water Content'], 'additive').\
                add_regressor('Bulk EC', hyper_parameters.regressor_weight['Bulk EC'], 'additive')
        
        model = m.train().get_model()
        initial = f'{int(len(dftraining) * 0.9)} days'
        df_cv = cross_validation(model, initial=initial, period=kwargs['period'], horizon=kwargs['horizon'], parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        mapes.append(df_p['mape'].values[0])

    tuning_results = pd.DataFrame(all_params)
    tuning_results['mape'] = mapes
    best_params = all_params[np.argmin(mapes)]
    best_params = pd.DataFrame.from_dict(best_params, orient='index').T
    return best_params, tuning_results

def neuralprophet_tuning_workflow(stations_dataframe_list, dftraining, shiftday):
    grid = hyper_parameters.get_neuralprophet_tuning_grid()
    cross_val_settings, usage = hyper_parameters.get_neuralprophet_tuning_settings()
    setting_grid = hyper_parameters.get_combination_from_dict(hyper_parameters.merge_dict(cross_val_settings, usage))
    print(f'n° models settings : {len(setting_grid)}')
    mapes_list, best_model_df, hyper_parameter_table_results_list = [list(), pd.DataFrame(), dict()]
    for setting in setting_grid:
        num_rows = int(len(dftraining) * (1-setting['usage']))
        training_subset = dftraining.loc[num_rows:]
        kwargs = dict({'period':setting['period'], 'horizon':setting['horizon']})
        bm, hptr = neuralprophet_hyperparameter_tuning(stations_dataframe_list, training_subset, shiftday, grid, **kwargs)
        best_model_df = pd.concat([best_model_df, bm])
        key = json.dumps(setting)
        hyper_parameter_table_results_list[key] = hptr
        mapes_list.append(hptr.loc[np.argmin(hptr['mape'])]['mape'])
        print(f'End cycle using : {setting["usage"]} of the dataset')
    best_model_df['setting'] = setting_grid
    best_model_df['mape'] = mapes_list
    best_params = best_model_df.iloc[np.argmin(mapes_list)]
    return best_params, best_model_df, hyper_parameter_table_results_list

def np_split(dftraining):
    df_test_lenght = len(dftraining)
    split_idx = int(df_test_lenght * 0.9) #careful
    dftrain = dftraining.iloc[:split_idx]
    dfvalidation = dftraining.iloc[split_idx:]
    return dftrain, dfvalidation

def neuralprophet_hyperparameter_tuning(stations_dataframe_list, dftraining, shiftday, param_grid, **kwargs):
    all_params = hyper_parameters.get_combination_from_dict(param_grid)
    print(f'n° models : {len(all_params)}')
    mapes = list()
    initial_comb = dftraining[:int(len(dftraining) * 0.9)]
    validation = dftraining[int(len(dftraining) * 0.9):]
    initial, initial_val = np_split(initial_comb)
    counter = len(all_params)
    for params in all_params:
        params['n_forecasts'] = params['n_forecasts']//(shiftday//kwargs['period'])
        loss, delta_time = neuralprophet_cross_validation(initial_comb, initial, initial_val, validation, params, shiftday, stations_dataframe_list, **kwargs)
        counter = counter-1
        print(f'--> ESTIMATED TIME UP COMPLETE : {counter*delta_time} seconds.')
        mapes.append(loss)
    tuning_results=pd.DataFrame(all_params)
    tuning_results['mape'] = mapes
    best_params = all_params[np.argmin(mapes)]
    best_params = {k: str(v) for k, v in best_params.items()}
    best_params = pd.DataFrame.from_dict(best_params, orient='index').T
    return best_params, tuning_results

def neuralprophet_cross_validation(initial_comb, initial, initial_val, validation, params, shiftday, stations_dataframe_list, **kwargs):
    if kwargs['horizon'] != params['n_forecasts']:
        print("Note: n_forecasts and horizon needs to be compatible the during cross-validation.",\
              "n_forecasts will be automatically adjusted.")
        params['n_forecasts'] = kwargs['horizon']
    next_horizon = validation[:kwargs['horizon']]
    losses = list()
    start_timer = time.time()
    for t in range(len(validation)//kwargs['period']):
        model_obj = neuralprophet_crossval_model(initial, initial_val, shiftday, params, stations_dataframe_list)
        horizon = generate_future('neuralprophet', model_obj, initial_comb, next_horizon)
        yhats = exctract_yhat(model_obj.train().forecast(horizon), kwargs['horizon'])
        loss = np.mean(np.abs((next_horizon['y'].values - yhats['yhat1'].values) / np.abs(next_horizon['y'].values)))
        losses.append(loss)
        updated_period = validation.iloc[t*kwargs['period']:(t+1)*kwargs['period']]
        initial_comb = pd.concat([initial_comb, updated_period], axis=0)
        initial, initial_val = np_split(initial_comb)
        next_horizon = validation[kwargs['period']*(t+1):kwargs['period']*(t+1)+kwargs['horizon']]
    delta_time = time.time() - start_timer
    print("One cycle of Cross-Validation took :", delta_time)
    mape = np.mean(losses)
    return mape, delta_time



def neuralprophet_crossval_model(initial, initial_val, shiftday, params, stations_dataframe_list):
    if sample == 'box1':
        model_obj = model_neuralprophet(initial, validation_set=initial_val, test_set=None, kwargs=params).\
            add_regressor('Temperature', hyper_parameters.regressor_weight['Temperature'], 'additive')
            
    elif sample == 'box2':
        model_obj = model_neuralprophet(initial, validation_set=initial_val, test_set=None, kwargs=params).\
            add_regressor('Soil Temperature', hyper_parameters.regressor_weight['Soil Temperature'], 'additive').\
            add_regressor('Water Content', hyper_parameters.regressor_weight['Water Content'], 'additive').\
            add_regressor('Bulk EC', hyper_parameters.regressor_weight['Bulk EC'], 'additive')
           
    
    for dataframe in stations_dataframe_list:
        model_obj.add_regressor(dataframe['quadrante_id'][0], hyper_parameters.regressor_weight['stations'], 'additive')
    return model_obj

def arima_tuning_workflow(stations_dataframe_list, dftraining, shiftday):
    grid = hyper_parameters.get_arima_tuning_grid()
    cross_val_settings, usage = hyper_parameters.get_arima_tuning_settings()
    setting_grid = hyper_parameters.get_combination_from_dict(hyper_parameters.merge_dict(cross_val_settings, usage))
    print(f'n° models settings : {len(setting_grid)}')
    mapes_list, best_model_df, hyper_parameter_table_results_list = [list(), pd.DataFrame(), dict()]
    for setting in setting_grid:
        num_rows = int(len(dftraining) * (1-setting['usage']))
        if (num_rows > 0):
            training_subset = dftraining.loc[num_rows:]
        else:
            training_subset = dftraining
        kwargs = dict({'period':setting['period'], 'horizon':setting['horizon']})
        bm, hptr = arima_hyperparameter_tuning(stations_dataframe_list, training_subset, shiftday, grid, **kwargs)
        best_model_df = pd.concat([best_model_df, bm])
        key = json.dumps(setting)
        hyper_parameter_table_results_list[key] = hptr
        mapes_list.append(hptr.loc[np.argmin(hptr['mape'])]['mape'])
        print(f'End cycle using : {setting["usage"]} of the dataset')
    best_model_df['setting'] = setting_grid
    best_model_df['mape'] = mapes_list
    best_params = best_model_df.iloc[np.argmin(mapes_list)]
    return best_params, best_model_df, hyper_parameter_table_results_list

def arima_hyperparameter_tuning(stations_dataframe_list, dftraining, shiftday, param_grid, **kwargs):
    all_params = {'order':list(itertools.product(param_grid['order']['p'], param_grid['order']['d'], param_grid['order']['q']))}
    all_params = hyper_parameters.get_combination_from_dict(all_params)
    print(f'n° models : {len(all_params)}')
    mapes = list()


    # 80% of the dataset is used for training and the remaining 10% for validation 
    initial_split_index = int(len(dftraining) * 0.8)
    initial = dftraining[:initial_split_index]
    validation = dftraining[initial_split_index:]

    for params in all_params:
        params['enforce_stationarity'] = True
        params['enforce_invertibility'] = True
        model_obj = model_arima(initial, kwargs=params)
        model = model_obj.train().get_model()
        loss = statsmodels_cross_validation(model_obj, initial, validation, **kwargs)
        mapes.append(loss)
    tuning_results=pd.DataFrame(all_params)
    tuning_results['mape'] = mapes
    best_params = all_params[np.argmin(mapes)]
    best_params = {k: str(v) for k, v in best_params.items()}
    best_params = pd.DataFrame.from_dict(best_params, orient='index').T
    return best_params, tuning_results

def sarimax_tuning_workflow(stations_dataframe_list, dftraining, shiftday):
    grid = hyper_parameters.get_sarimax_tuning_grid()
    cross_val_settings, usage = hyper_parameters.get_arima_tuning_settings()
    setting_grid = hyper_parameters.get_combination_from_dict(hyper_parameters.merge_dict(cross_val_settings, usage))
    print(f'n° models settings : {len(setting_grid)}')
    mapes_list, best_model_df, hyper_parameter_table_results_list = [list(), pd.DataFrame(), dict()]
    for setting in setting_grid:
        num_rows = int(len(dftraining) * (1-setting['usage']))
        training_subset = dftraining.loc[num_rows:]
        kwargs = dict({'period':setting['period'], 'horizon':setting['horizon']})
        bm, hptr = sarimax_hyperparameter_tuning(stations_dataframe_list, training_subset, shiftday, grid, **kwargs)
        best_model_df = pd.concat([best_model_df, bm])
        key = json.dumps(setting)
        hyper_parameter_table_results_list[key] = hptr
        mapes_list.append(hptr.loc[np.argmin(hptr['mape'])]['mape'])
        print(f'End cycle using : {setting["usage"]} of the dataset')
    best_model_df['setting'] = setting_grid
    best_model_df['mape'] = mapes_list
    best_params = best_model_df.iloc[np.argmin(mapes_list)]
    return best_params, best_model_df, hyper_parameter_table_results_list

def sarimax_hyperparameter_tuning(stations_dataframe_list, dftraining, shiftday, param_grid, **kwargs):
    order_params = {'order':list(itertools.product(param_grid['order']['p'], param_grid['order']['d'], param_grid['order']['q']))}
    seasonal_params = {'seasonal_order':list(itertools.product(param_grid['seasonal_order']['P'], param_grid['seasonal_order']['D'], param_grid['seasonal_order']['Q'], param_grid['seasonal_order']['M']))}
    all_params = hyper_parameters.merge_dict(seasonal_params, order_params)
    all_params = hyper_parameters.get_combination_from_dict(all_params)
    all_params = filter_sarimax_models(all_params)
    print(f'n° models : {len(all_params)}')
    mapes = list()


    # 80% of the dataset is used for training and the remaining 10% for validation 
    initial_split_index = int(len(dftraining) * 0.8)
    initial = dftraining[:initial_split_index]
    validation = dftraining[initial_split_index:]

    for params in all_params:
        print('params tested: ',params)
        params['enforce_stationarity'] = True
        params['enforce_invertibility'] = True
        model_obj = model_sarimax(initial, kwargs=params)
        model_obj.train()
        
        loss = statsmodels_cross_validation(model_obj, initial, validation, **kwargs)        
        mapes.append(loss)
    tuning_results=pd.DataFrame(all_params)
    tuning_results['mape'] = mapes
    best_params = all_params[np.argmin(mapes)]
    best_params = {k: str(v) for k, v in best_params.items()}
    best_params = pd.DataFrame.from_dict(best_params, orient='index').T
    return best_params, tuning_results

def statsmodels_cross_validation(model, initial, validation, **kwargs):
    mapes = list()
    horizon = validation[:kwargs['horizon']]
    forecasts = {}
    forecasts[initial.index[-1]] = model.forecast(horizon)
    for t in range(len(validation)//kwargs['period']):
        updated_endog = validation.iloc[t*kwargs['period']:(t+1)*kwargs['period']]
        endog, exog = endog_exog_split(updated_endog)
        model.fitted = model.fitted.append(endog, exog=exog, refit=True)
        next_horizon = validation[kwargs['period']*(t+1):kwargs['period']*(t+1)+kwargs['horizon']]
        forecasts[updated_endog.index[-1]] = model.forecast(next_horizon)
        loss = np.mean(np.abs((forecasts[updated_endog.index[-1]]['yhat'] - next_horizon['y'].values.tolist()) / np.abs(forecasts[updated_endog.index[-1]]['yhat'])))
        
        mapes.append(loss)
    
    mape = np.mean(mapes)
    return mape
    
def endog_exog_split(train_data):

    endog_data = train_data[['y']]
    exog_data = train_data.drop(columns=['y'])

    return endog_data, exog_data

def combine_forecast(df_l, df_r):
    df_l = df_l.combine_first(df_r)
    df_l = df_r.fillna(df_l)
    return df_l

def filter_sarimax_models(all_params):
    print(f'n° models before filtering: {len(all_params)}')
    new_params = list()
    for params in all_params:
        if params['order'][0] == 0 or params['order'][0] == 1:
            new_params.append(params)
        elif params['seasonal_order'][0] == 0:
            new_params.append(params)
        elif params['order'][0] % params['seasonal_order'][3] != 0:
            new_params.append(params)
    return new_params

def exctract_yhat(forecasts, size):
    # Identify yhat and regressor columns
    yhat_columns = [col for col in forecasts.columns if 'yhat' in col]
    regressor_columns = [col for col in forecasts.columns if 'future_regressor_' in col]
    
    # Create the initial newframe with all rows
    newframe = forecasts[['ds', 'yhat1']].copy()
    
    # Fill missing predictions
    for col in yhat_columns:
        if col != 'yhat1':
            newframe['yhat1'] = newframe['yhat1'].fillna(forecasts[col])
    
    # Add regressor columns
    for reg_col in regressor_columns:
        reg_name = reg_col.replace('future_regressor_', '')
        newframe[reg_name] = forecasts[reg_col]
    
    # Select the last `size` rows
    newframe = newframe.iloc[-size:].copy()
    return newframe

