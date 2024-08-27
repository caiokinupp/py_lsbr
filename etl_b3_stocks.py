import yfinance as yf
import pandas as pd

LST_B3_STOCKS = [
    "RRRP3",
    "ALOS3",
    "ALPA4",
    "ABEV3",
    "ASAI3",
    "AZUL4",
    "AZZA3",
    "B3SA3",
    "BBSE3",
    "BBDC3",
    "BBDC4",
    "BRAP4",
    "BBAS3",
    "BRKM5",
    "BRFS3",
    "BPAC11",
    "CRFB3",
    "CCRO3",
    "CMIG4",
    "COGN3",
    "CPLE6",
    "CSAN3",
    "CPFE3",
    "CMIN3",
    "CVCB3",
    "CYRE3",
    "DXCO3",
    "ELET3",
    "ELET6",
    "EMBR3",
    "ENGI11",
    "ENEV3",
    "EGIE3",
    "EQTL3",
    "EZTC3",
    "FLRY3",
    "GGBR4",
    "GOAU4",
    "NTCO3",
    "HAPV3",
    "HYPE3",
    "IGTI11",
    "IRBR3",
    "ITSA4",
    "ITUB4",
    "JBSS3",
    "KLBN11",
    "RENT3",
    "LREN3",
    "LWSA3",
    "MGLU3",
    "MRFG3",
    "BEEF3",
    "MRVE3",
    "MULT3",
    "PCAR3",
    "PETR3",
    "PETR4",
    "RECV3",
    "PRIO3",
    "PETZ3",
    "RADL3",
    "RAIZ4",
    "RDOR3",
    "RAIL3",
    "SBSP3",
    "SANB11",
    "SMTO3",
    "CSNA3",
    "SLCE3",
    "SUZB3",
    "TAEE11",
    "VIVT3",
    "TIMS3",
    "TOTS3",
    "TRPL4",
    "UGPA3",
    "USIM5",
    "VALE3",
    "VAMO3",
    "VBBR3",
    "VIVA3",
    "WEGE3",
    "YDUQ3",
]


def get_yahoo_asset_data(lst_stock_tickers, start_date, end_date):
    lst_assets = []
    for ticker in lst_stock_tickers:
        df_stock = yf.download(
            ticker + ".SA", start=start_date, end=end_date, progress=False
        )

        if len(df_stock) != 0:
            df_stock = df_stock.loc[:, ["Adj Close"]]
            df_stock.loc[:, "ticker"] = ticker
            lst_assets.append(df_stock)

    return pd.concat(lst_assets)


def pivote_asset_data(df):
    # Reseting index to pivoting
    df.reset_index(inplace=True)

    # Pivoting Data
    df = df.pivot(index="Date", columns="ticker", values="Adj Close")

    # Renaming index
    df.index.names = ["date"]

    # Fill NA
    df = df.fillna(method="ffill")

    return df


def create_long_short_dataset(
    lst_stock_tickers=LST_B3_STOCKS,
    start_date="2018-01-01",
    end_date=None,
    source="yahoo",
):
    if source == "yahoo":
        df = get_yahoo_asset_data(
            lst_stock_tickers=lst_stock_tickers,
            start_date=start_date,
            end_date=end_date,
        )
        df = pivote_asset_data(df)
        df.to_csv("ls_data/yahoo_long_short_data.csv")
    else:
        pass
