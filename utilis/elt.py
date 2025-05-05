from scipy import signal
import numpy as np
import pandas as pd
import models

def logscale_data(df, logscale):
    if logscale:
        cols_to_transform = df.columns.difference(['ds'])
        df[cols_to_transform] = np.log(df[cols_to_transform].values.astype(np.float128)+0.01).astype(np.float64)
    return df

def expscale_data(df, expscale):
    if expscale:
        cols_to_transform = df.columns.difference(['ds'])
        df[cols_to_transform] = np.exp(df[cols_to_transform].values.astype(np.float128))-0.01
        df[cols_to_transform] = df[cols_to_transform].astype(np.float64)
    return df

def train_test_split(stations, shift_data, shiftday, shift_input):
    if shift_data:
        '''modifica effettuata, dftest era dopo l'if questo faceva sì che i dati venissero cuttati'''
        dftest = stations[-shiftday:].copy()
        if not shift_input:
            stations.iloc[:,:-2] = stations.iloc[:,:-2].shift(periods=shiftday, fill_value=0)
        if shift_input:
            dftraining = stations[shiftday:-shiftday]
        else:
            dftraining = stations[:-shiftday]
    else:
        '''dati dal 2001'''
        dftest = stations[-shiftday:]
        dftraining = stations[:-shiftday]
    
    return dftraining, dftest

def etl_esino(stations, esino, esinoflag, start_date):
    '''
    Prendo i dati del fiume Esino dal 2001
    '''
    if esinoflag:
        stations['esino']=np.nan
        if start_date.year == 2000:
            for i in range(0,(stations.shape[0])):
                stations['esino'][i] = esino['livello assoluto fiume Esino'][(365+1)+i]
        else:
            stations['esino'] = esino['livello assoluto fiume Esino'].copy()
    return stations

def etl_mlsm(stations, input, start_date):
    stations['mslm']=np.nan
    if start_date.year == 2000:
        for i in range(0,stations.shape[0]):
            stations['mslm'][i] = input['mslm'][(365+1)+i]
    else:
        stations['mslm'] = input['mslm'].copy()
    return stations

def etl_stations(stations, input, shift_input, shift_data, shiftday, start_date, esinoflag):
    stations['data']=''
    if shift_input:
        addrows = pd.DataFrame(0, index=np.arange(shiftday), columns=stations.columns)
        stations = pd.concat([stations, addrows], ignore_index=True)
        stations.iloc[-shiftday:,:-2] = 6.4
        if esinoflag:
            stations['esino'].iloc[-shiftday:] = 158
    elif shift_data and not shift_input:
        '''generare i regressori futuri'''
        addrows = pd.DataFrame(0, index=np.arange(shiftday), columns=stations.columns)
        stations = pd.concat([stations, addrows], ignore_index=True)
    if start_date.year == 2000:
        for i in range(0,(input['data'][(365+1):]).size):
            stations['data'][i] = input['data'][(365+1)+i]
    else:
        stations['data'] = input['data'].copy()
    if shift_data and not shift_input:
        df_regressors = pd.DataFrame()
        df_regressors['ds'] = stations['data'].copy()
        for col in stations.columns[:-2]:
            train, test = train_test_split(stations[[col,'data']], shift_data, shiftday, shift_input)
            train.rename(columns = {col:'y', 'data':'ds'}, inplace = True)
            test.rename(columns = {col:'y', 'data':'ds'}, inplace = True)
            _, forecast = models.generate_regressor_future(train, test)
            df_regressors[col] = pd.concat([train['y'], forecast['yhat']], ignore_index=True)
    return stations

def etl_portata(stations, gorgovivo, gorgovivoflag, shift_data, shiftday):
    if gorgovivoflag:
        gorgovivo = gorgovivo.loc[gorgovivo['data'].iloc[:] >= pd.Timestamp('2007-01-01 00:00:00')].copy()
        gorgovivo = gorgovivo.reset_index(drop = True)
        if gorgovivo['data'].iloc[(len(gorgovivo)-1)] > stations['data'].iloc[(len(stations)-1)]:
            gorgovivo = gorgovivo.loc[gorgovivo['data'].iloc[:] <= stations['data'].iloc[(len(stations)-1)]].copy()
        if stations['data'][0] < pd.Timestamp('2007-01-01 00:00:00'):
            stations = stations.loc[stations['data'].iloc[:] >= pd.Timestamp('2007-01-01 00:00:00')].copy()
            stations = stations.reset_index(drop = True)
        if stations['data'][0] > gorgovivo['data'][0]:
            gorgovivo = gorgovivo.loc[gorgovivo['data'].iloc[:] >= pd.Timestamp(str(stations['data'][0].year)+'-01-01 00:00:00')].copy()
            gorgovivo = gorgovivo.reset_index(drop = True)
        if shift_data:
            addrows = pd.DataFrame(1200, index=np.arange(shiftday), columns=gorgovivo.columns)
            gorgovivo = pd.concat([gorgovivo, addrows], ignore_index=True)
            #gorgovivo = gorgovivo.append(pd.DataFrame(1500, index=np.arange(shiftday), columns=gorgovivo.columns), ignore_index=True)
        gorgovivo.iloc[:, 0] = stations['data'].copy()
        stations.insert(len(stations.columns) - 2, 'gorgovivo', gorgovivo.iloc[:, 1])
    return stations

def check_len(l1,l2):
    return len(l1) == len(l2)

def compute_lags(stations):
    lag_tot = list()
    concat_exogs = pd.DataFrame()
    #stations.data = pd.to_datetime(stations.data, unit='d')
    for i in range(0,len(stations.columns)-2):
        print(f'Computing lags for {stations.columns[i]}.')
        #print(stations.head(10))
        exogens_list = divide_dataframe(stations[[stations.columns[i],'data']])
        mslm_divided = divide_dataframe(stations[['mslm', 'data']])
        if not check_len(exogens_list, mslm_divided):
            print('Errore nello splitting dei dati')
            quit()
        else:
            for j in range(len(exogens_list)):
                lag = compute_correlation_lags(exogens_list[j][stations.columns[i]], mslm_divided[j]['mslm'])
                #print('Il lag calocolato è di :', lag, 'giorno/i')
                lag_tot.append(lag)
                shifted_exog = shift_dataframe_indices(lag_tot, exogens_list)
                duplicated_indices, adj_exog = concat_df_list(shifted_exog)
                exog = handle_tempsync(duplicated_indices, adj_exog)
            concat_exogs = pd.concat([exog, concat_exogs], axis=1)
        #print(f'Lags calcolati per stagione : {lag_tot}')
    concat_exogs.reset_index(inplace=True) 
    concat_exogs = pd.concat([concat_exogs, stations[stations.columns[-2::]]], axis=1)
    concat_exogs = merge_dataframes(concat_exogs, stations)
    #quit()
    return concat_exogs

def shift_dataframe_indices(lags, dataframes):
    shifted_dataframes = list()
    for lag, dataframe in zip(lags, dataframes):
        shifted_dataframe = dataframe.copy()
        shifted_dataframe[dataframe.columns[0]] = shifted_dataframe[dataframe.columns[0]].shift(-lag, freq='D')
        shifted_dataframes.append(shifted_dataframe)
    return shifted_dataframes

def compute_correlation_lags(df, mslm):
    diff_days = (df.index[0] - mslm.index[0]).days
    moving = df.values
    default = mslm.values
    cross_correlation = np.correlate(default, moving, mode='full')
    correlation_shift = np.argmax(cross_correlation) - (len(default) - 1) + diff_days
    return correlation_shift

def divide_dataframe(df):
    num_rows = df.shape[0]
    seasons_num = (num_rows*4)//(365)
    rows_per_seas = num_rows // seasons_num
    #print('num rows : ', num_rows, 'season num : ', seasons_num, 'rows per season : ', rows_per_seas)
    divided_dataframes = []
    for i in range(seasons_num):
        start_index = i * rows_per_seas
        end_index = start_index + rows_per_seas
        part_df = df.iloc[start_index:end_index]
        part_df.set_index('data', inplace=True)
        divided_dataframes.append(part_df)
    return divided_dataframes

def concat_df_list(shifted_df):
    concatenated_df = pd.concat(shifted_df)
    concatenated_df.sort_index(inplace=True)
    duplicated_indices = concatenated_df.index[concatenated_df.index.duplicated(keep=False)].unique()
    return duplicated_indices, concatenated_df

def handle_tempsync(duplicated_indices, concatenated_df):
    dfr = concatenated_df.copy()

    for index in duplicated_indices:
            rows = dfr.loc[index]
            num_null_values = rows.isnull().sum().values
            if num_null_values == 2:
                merged_row = rows[0]
            elif num_null_values == 1:  # Una riga ha valore nullo, l'altra ha un valore numerico definito
                merged_row = rows.dropna()
            else :  # Entrambe le righe hanno valori numerici diversi
                mean_row = rows.iloc[:1].copy()
                mean_values = rows.mean()
                mean_row.iloc[0] = mean_values.values
                merged_row = mean_row
            print(merged_row)
            dfr = dfr[dfr.index != index]
            dfr = pd.concat([dfr, merged_row])

    dfr.sort_index(inplace=True)
    dfr.interpolate(method='linear', limit_direction='both', inplace=True)
    return dfr

def merge_dataframes(df1, df2): #df1 è quello da aggiustare rispetto a df2 che è mslm
    merged_df = df1.merge(df2, how='outer', left_index=True, right_index=True)
    merged_df.drop(['data_x', 'data_y'], axis=1, inplace=True)
    merged_df['data'] = df2.data
    for column in merged_df.columns:
        if '_x' in column:
            if 'data' not in column:
                column_y = str(column[:-2]+'_y')
                merged_df[column].fillna(merged_df[column_y], inplace=True)
                merged_df.drop([column_y], axis=1, inplace=True)
                merged_df = merged_df.rename(columns={column: column[:-2]})
    return merged_df
'''
if not(norm):
    fig=model.plot(forecast, uncertainty=True)
    if(end_date.year() < 2021):
        plt.plot(dftest['data'], dftest['mslm'],'r-',label='Validation')
    plt.rc('xtick',labelsize=6)
    plt.xlabel('year', fontsize=14)
    plt.ylabel('level [m]', fontsize=14)
    plt.legend(loc="upper right")
    plt.show(block=False)

else:
    f,ax2=plt.subplots(1)
    dftraining.iloc[:, :-1] = scaler.inverse_transform(dftraining.iloc[:, :-1])
    dftest.iloc[:, :-1] = scaler.inverse_transform(dftest.iloc[:, :-1])
    plt.plot(dftraining['ds'], (dftraining['y']),'k.',label='Training')
    if not (shift_data):
        plt.plot(dftest['data'], (dftest['mslm']),'r-', label='Validation')
    forecast['yhat']=invTransform(scaler, forecast['yhat'],'mslm',scaler.feature_names_in_)
    plt.plot(forecast['ds'], forecast['yhat'], ls='-', c='#0072B2', label='Prediction')
    forecast['yhat_lower']=invTransform(scaler, forecast['yhat_lower'],'mslm',scaler.feature_names_in_)
    forecast['yhat_upper']=invTransform(scaler, forecast['yhat_upper'],'mslm',scaler.feature_names_in_)
    ax2.fill_between(forecast['ds'][-shiftDay:].dt.to_pydatetime(), (forecast['yhat_lower'][-shiftDay:]), (forecast['yhat_upper'][-shiftDay:]),color='#0072B2', alpha=0.2, label='Uncertainty interval')
    plt.rc('xtick',labelsize=6)
    plt.xlabel('year', fontsize=14)
    plt.ylabel('level [m]', fontsize=14)
    plt.title(f'Normalizzazione Scaler', fontsize=16)
    plt.legend(loc="upper right")
    plt.show(block=False)
'''
