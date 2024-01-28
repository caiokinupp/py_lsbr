import yfinance as yf
import pandas as pd
import os


path_ls_data = 'ls_data/'
col_name_asset = 'ticker'
col_name_date = 'Date'
col_name_close = 'Adj Close'


def extractYahooData(path_ls_data, col_name_close, col_name_asset):
    # Identifying Profit data files
    lst_profit_files = []
    for file_name in os.listdir(path_ls_data):
        if file_name.split("_")[0] == 'profit':
            lst_profit_files.append(file_name)
    
    # Identifying most recent data file for reference
    lst_profit_dates = []
    for profit_file in lst_profit_files:
        file_date_str = profit_file[7:-4]
        lst_profit_dates.append(pd.to_datetime(file_date_str, format='%Y-%m-%d'))

    
    # Time Window
    max_profit_date = max(lst_profit_dates)
    max_date = pd.to_datetime('today').normalize()
    min_date = max_date - pd.DateOffset(years=1) - pd.DateOffset(months=1)
    
    # Getting IBRX tickers
    df_profit = pd.read_csv(path_ls_data + 'profit_' + str(max_profit_date)[0:-9] + '.csv')
    assets_tickers = df_profit.columns[1:]
    
    lst_assets=[]
    for ticker in assets_tickers:
        asset_data = yf.download(ticker+'.SA', start = min_date, end = max_date)
        asset_data = asset_data[[col_name_close]]
        asset_data[col_name_asset] = ticker
        lst_assets.append(asset_data)
    
    
    return pd.concat(lst_assets)


def transformYahooData(df, col_name_asset, col_name_date, col_name_close):
    # Reseting index to pivoting
    df.reset_index(inplace = True)
    
    # Pivoting Data
    df = df.pivot(index=col_name_date, columns=col_name_asset, values=col_name_close)
    
    # Deleting assets with less than a year of data
    df = df.dropna(axis=1, thresh=252)
    
    # Renaming index
    df.index.names = ['date']
    return df


def loadYahooData(df, path_ls_data):
    max_date = str(df.index.max().date())
    file_name = 'yahoo_'+max_date
    df.to_csv(path_ls_data+file_name+'.csv')
    
    return None


def etlyahoo(path_ls_data, col_name_close, col_name_asset, col_name_date):
    df = extractYahooData(path_ls_data, col_name_close, col_name_asset)
    df = transformYahooData(df, col_name_asset, col_name_date, col_name_close)
    loadYahooData(df, path_ls_data)
    
    return None


etlyahoo(path_ls_data, col_name_close, col_name_asset, col_name_date)