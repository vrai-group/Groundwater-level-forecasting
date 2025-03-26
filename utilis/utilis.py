from functools import reduce
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
import models
import os
import time

#making the code workable on all operating systems by using os.path.join instead of using // or /
class instance_path:
    path = None
    @staticmethod
    def set_path(path, folder):
        if not os.path.exists(path):
            os.makedirs(path)
        instance_path.path = create_folder_with_timestamp(f'{path}/{folder}')
    @staticmethod
    def get_path():
        return instance_path.path

def create_folder_with_timestamp(name):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    folder_name = f"{name}_{timestamp}"
    os.mkdir(folder_name)
    return folder_name
class forecast_helper:
    def __init__(self, model_name, shift_data):
        self.model_name = model_name
        self.shift_data = shift_data

    def plot_helper(self):
        sns.set_style('darkgrid')
        fig, _ = plt.subplots(figsize=(12, 6), dpi=100)
        return fig

    def save_helper(self, fig, forecast):
        namefolder = instance_path.get_path()
        if self.shift_data:
            fig.savefig(f'{namefolder}/forecast_{self.model_name}.jpg')
            forecast.to_csv(f'{namefolder}/forecast_{self.model_name}.csv', index=False)
        else:
            fig.savefig(f'{namefolder}/validation_{self.model_name}.jpg')
            forecast.to_csv(f'{namefolder}/validation_{self.model_name}.csv', index=False)
class arima_like_helper(forecast_helper):
    def __init__(self, model_name, shift_data, forecast, dftest, dftraining):
        super().__init__(model_name, shift_data)
        self.forecast = forecast
        self.dftest = dftest
        self.dftraining = dftraining
        self.fig = None
    
    def save_helper(func):
        def wrapper(self):
            func(self)
            super().save_helper(self.fig, self.forecast)
        return wrapper

    @save_helper
    def plot_helper(self):
        fig = super().plot_helper()
        if self.shift_data:
            pass
        else:
            #plt.plot(self.dftraining['ds'], self.dftraining['y'], '.', color='black', label='Observed data points')
            plt.plot(self.forecast['ds'], self.forecast['yhat'], color='blue', label='forecast')
            plt.plot(self.dftest['ds'], self.dftest['y'], '.', color='orange', label='Validation')
        plt.legend(loc='upper right')
        self.fig = fig

class prophet_helper(forecast_helper):
    def __init__(self, model, model_name, shift_data, forecast, dftest, dftraining):
        super().__init__(model_name, shift_data)
        self.model = model
        self.forecast = forecast
        self.dftest = dftest
        self.dftraining = dftraining
        self.fig = None
    
    def save_helper(func):
        def wrapper(self):
            func(self)
            super().save_helper(self.fig, self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        return wrapper

    @save_helper
    def plot_helper(self):
        fig = super().plot_helper()
        model = self.model.get_model()
        print(model.component_modes)
        if self.shift_data:
            fig = model.plot(self.forecast, uncertainty=True)
        else :
            #fig = model.plot(self.forecast, uncertainty=True)
            plt.plot(self.dftraining['ds'], self.dftraining['y'], '.', color='black', label='Observed data points')
            plt.plot(self.forecast['ds'], self.forecast['yhat'], color='red', label='Forecast')
            plt.plot(self.dftest['ds'], self.dftest['y'], '.', color='blue', label='Validation')
        plt.xlabel('Time (yyyy-mm-dd)')
        plt.ylabel('Depth to water table (m)')
        plt.legend(loc='upper right')
        self.fig = fig

class neuralprophet_helper(forecast_helper):
    def __init__(self, model_name, shift_data, forecast, dftest, dftraining):
        super().__init__(model_name, shift_data)
        self.forecast = forecast
        self.dftest = dftest
        self.dftraining = dftraining
        self.fig = None
    
    def save_helper(func):
        def wrapper(self):
            func(self)
            super().save_helper(self.fig, self.forecast[['ds', 'yhat1']])
        return wrapper

    @save_helper
    def plot_helper(self):
        fig = super().plot_helper()
        if self.shift_data:
            pass
        else :
            #plt.plot(self.dftraining['ds'], self.dftraining['y'], '.', color='black', label='Observed data points')
            plt.xlabel('Time (yyyy-mm-dd)')
            plt.ylabel('Depth to water table (m)')
        plt.plot(self.forecast['ds'], self.forecast['yhat1'], color='blue', label='Forecast')
        plt.plot(self.dftest['ds'], self.dftest['y'], '.', color='orange', label='Validation')
        plt.legend(loc='upper right')
        self.fig = fig

def plot_flow(correlation, forecast=None, dftraining=None, dftest=None, model_name=None, shift_data=None, model=None, tuning=False,\
               best_params=None ,best_model=None, hyper_parameter_table_results=None):
    if tuning:
        best_params.to_csv(f'{instance_path.get_path()}/best_absolute_params.csv', index=True)
        best_model.to_csv(f'{instance_path.get_path()}/best_model_.csv', index=True)
        for usage in hyper_parameter_table_results:
            hyper_parameter_table_results[usage].to_csv(f'{instance_path.get_path()}/hyper_parameter_table_results_'+usage.replace('"','').replace(':','')+'.csv', index=False)
    else:
        if model_name == 'arima' or model_name == 'sarimax':
            arima_like_helper(model_name, shift_data, forecast, dftest, dftraining).plot_helper()
            if not shift_data:
                metrics = pd.DataFrame.from_dict(models.forecast_accuracy(forecast['yhat'].values, dftest['y'].values), orient='index').T
                print(metrics.head(1))
        elif model_name == 'prophet':
            prophet_helper(model, model_name, shift_data, forecast, dftest, dftraining).plot_helper()
            df_test_lenght = len(dftest)
            if not shift_data:
                metrics = pd.DataFrame.from_dict(models.forecast_accuracy(forecast['yhat'].iloc[-df_test_lenght:].values, dftest['y'].values), orient='index').T
                print(metrics.head(1))
        elif model_name == 'neuralprophet':
            neuralprophet_helper(model_name, shift_data, forecast, dftest, dftraining).plot_helper()
            if not shift_data:
                if len(forecast) == len(dftest):
                    metrics = pd.DataFrame.from_dict(models.forecast_accuracy(forecast['yhat1'].values, dftest['y'].values), orient='index').T
                    print(metrics.head(1))
        if not shift_data:
            if len(forecast) == len(dftest):
                metrics.to_csv(f'{instance_path.get_path()}/metrics.csv', index=False)
    correlation.to_csv(f'{instance_path.get_path()}/correlation.csv', index=False)

def invTransform(scaler, data, colName, colNames):
    dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
    dummy[colName] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
    return dummy[colName].values

def pipe(*functions):
    return lambda x: reduce(lambda v, f: f(v), functions, x)

def get_all_stations(config : dict) -> dict :
    return config['name-map']['stations']

def get_input_stations(config : dict) -> list :
    return config['trainedon']['stations']

def get_esogen_path(config : dict) -> str :
    return config['path']['esogen'] + config['name-map']['piezom']

def get_esino_path(config : dict) -> str :
    return config['path']['esino'] + config['name-map']['esino']

def get_gorgovivo_path(config : dict) -> str :
    return config['path']['gorgovivo'] + config['name-map']['gorgovivo']

def get_stations_paths(config : dict) -> list :
    return list(map(lambda x: config['path']['endogen'] + config['name-map']['stations'][x], get_input_stations(config)))

def load_to_dataframe(path : str) -> pd.DataFrame :
    if '.xlsx' in path :
        return pd.read_excel(path)
    else :
        return pd.read_csv(path, sep=';')

def check_trainedon(param : tuple) -> dict :
    config, check = param
    if not(config['trainedon'][check]) :
        raise ValueError(f'{check} is missing')
    else :
        return config
class configuration_class:
    __instance = None 
    esinoflag, gorgovivoflag, stationsflag = [None, None, None]
    input, esino, gorgovivo, stations_dataframe_list = [None, None, None, None]
    model_name, start_date, end_date = [None, None, None]
    shift_data, shift_input, months, shiftday = [None, None, None, None]
    tuning, logscale, lag_flag = [None, None, None]
    config = None

    def get_flags(cls):
        return cls.esinoflag, cls.gorgovivoflag, cls.stationsflag
    def get_input(cls):
        return cls.input
    def get_esino(cls):
        return cls.esino
    def get_portata(cls):
        return cls.gorgovivo
    def get_stationslist(cls):
        return cls.stations_dataframe_list
    def get_modelname(cls):
        return cls.model_name
    def get_period(cls):
        return cls.start_date, cls.end_date
    def get_timeshift(cls):
        return cls.shift_data, cls.shift_input, cls.months, cls.shiftday
    def get_options(cls):
        return cls.tuning, cls.logscale, cls.lag_flag

    def __new__(cls):
        if cls.__instance is None:
            cls.configuration_loader(cls)
            cls.__instance = \
                super(configuration_class, cls).__new__(cls)
        return cls.__instance 

    
    def configuration_loader(cls):
        if cls.__instance is None:
            with open("config.yml", "r") as stream:
                cls.config = yaml.safe_load(stream)

            cls.model_name = cls.config['options']['model']
            '''
            if cls.model_name is not ('prophet' or 'neuralprophet' or 'arima' or 'sarimax'):
                raise Exception('model name is incorrect')
            '''
            cls.start_date = datetime.strptime(cls.config['date']['start'], '20%y-%m-%d').date()
            cls.end_date = datetime.strptime(cls.config['date']['end'], '20%y-%m-%d').date()
            cls.months = cls.config['date']['months']
            cls.shiftday = round(365/12 * cls.months)
            cls.shift_input = cls.config['options']['shift_input']
            cls.tuning = cls.config['options']['hyperparameter-tuning']
            cls.logscale = cls.config['options']['logscale']
            cls.lag_flag = cls.config['options']['lag']
            cls.esinoflag = True
            cls.gorgovivoflag = True
            cls.stationsflag = True
            composed_esino = pipe(check_trainedon, get_esino_path, load_to_dataframe)
            composed_gorgovivo = pipe(check_trainedon, get_gorgovivo_path, load_to_dataframe)
            composed_stations = pipe(check_trainedon, get_stations_paths)
            cls.input = pd.read_excel(get_esogen_path(cls.config), sheet_name='PozzoInterno')

            try :
                cls.esino = composed_esino((cls.config, 'esino'))
            except Exception as error :
                cls.esinoflag = False
                print(error)
            try :
                cls.gorgovivo = composed_gorgovivo((cls.config, 'gorgovivo'))
            except Exception as error:
                cls.gorgovivoflag = False
                print(error)
            try :
                cls.stations_dataframe_list = list()
                stations_path = composed_stations((cls.config, 'endogen'))
                for st in stations_path :
                    cls.stations_dataframe_list.append(load_to_dataframe(st))
            except Exception as error:
                cls.stationsflag = False
                print(error)

            ''' Opzioni inserire successivamente
            norm = config['options']['normalize']
            plog = config['options']['plotlog']
            '''

            ''' filtri inserire successivamente
            filterIn7 = config['filter']['seven_in']
            filterIn14 = config['filter']['fourteen_in']
            filterOut7 = config['filter']['seven_out']
            filterOut14 = config['filter']['foruteen_out']
            '''
        else:
            raise Exception('Class already instantiated')