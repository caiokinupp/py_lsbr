import os
import pandas as pd


path_profit_data = 'profit_data/'
path_ls_data = 'ls_data/'
col_name_asset = 'Ativo'
col_name_date = 'Data'
col_name_close = 'Fechamento'


def extractProfitData(path_profit_data, col_name_asset, col_name_date, col_name_close):
    # Extracting assets data from files
    lst_assets = []

    for file_name in os.listdir(path_profit_data):
        df_asset = pd.read_csv(path_profit_data+file_name, sep=';', encoding='latin-1')
        lst_assets.append(df_asset[[col_name_date, col_name_asset, col_name_close]])  
    
    return pd.concat(lst_assets)


def transformProfitData(df, col_name_asset, col_name_date, col_name_close):
    # Transforming String Close data to Float
    df[col_name_close] = df[col_name_close].str.replace(',','.').astype(float)
    
    # Transforming String Date to Datetime
    df[col_name_date] = pd.to_datetime(df[col_name_date], format='%d/%m/%Y')
    
    # Filtering the last 252 periods + 1 month
    min_date = df[col_name_date].max() - pd.DateOffset(years=1) - pd.DateOffset(months=1)
    df = df.loc[df[col_name_date] >= min_date]
    
    # Pivoting Data
    df = df.pivot(index=col_name_date, columns=col_name_asset, values=col_name_close)
    
    # Deleting assets with less than a year of data
    df = df.dropna(axis=1, thresh=252)
    
    # Fill days without negotiation
    df = df.fillna(method='ffill')
    
    # Renaming index
    df.index.names = ['date']
    return df


def loadProfitData(df, path_ls_data):
    max_date = str(df.index.max().date())
    file_name = 'profit_'+max_date
    df.to_csv(path_ls_data+file_name+'.csv')
    
    return None


def etlProfit(path_profit_data, col_name_asset, col_name_date, col_name_close, path_ls_data):
    df = extractProfitData(path_profit_data, col_name_asset, col_name_date, col_name_close)
    df = transformProfitData(df, col_name_asset, col_name_date, col_name_close)
    loadProfitData(df, path_ls_data)
    
    return None


etlProfit(path_profit_data, col_name_asset, col_name_date, col_name_close, path_ls_data)