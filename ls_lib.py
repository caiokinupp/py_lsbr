# Python
# Third
# Property
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import warnings
import concurrent.futures

warnings.filterwarnings("ignore")

pd.set_option("display.max_rows", 150)

# Parameters
PATH_LS_DATA = "ls_data/"


def load_long_short_dataset(max_date=None, source="yahoo"):
    if source == "yahoo":
        if max_date == None:
            return pd.read_csv(
                PATH_LS_DATA + "yahoo_long_short_data.csv", index_col="date"
            ).reset_index()
        else:
            return (
                pd.read_csv(
                    PATH_LS_DATA + "yahoo_long_short_data.csv", index_col="date"
                )
                .loc[:max_date]
                .reset_index()
            )
    else:
        pass


def getPairs(df):
    assets = list(df.columns)
    assets.remove("date")

    lst_pairs = []
    for first in assets:
        for second in assets:
            if first != second:
                lst_pairs.append(first + "/" + second)

    return lst_pairs


def blog_half_life(ts):
    """
    Calculates the half life of a mean reversion
    """
    # make sure we are working with an array, convert if necessary
    ts = np.asarray(ts)

    ts = ts[::-1]

    # delta = p(t) - p(t-1)
    delta_ts = np.diff(ts)

    # calculate the vector of lagged values. lag = 1
    lag_ts = np.vstack([ts[:-1], np.ones(len(ts[:-1]))]).T

    # calculate the slope of the deltas vs the lagged values
    # Ref: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
    lambda_, const = np.linalg.lstsq(lag_ts, delta_ts, rcond=None)[0]

    # compute and return half life
    # negative sign to turn half life to a positive value
    return -np.log(2) / lambda_


def gpt_half_life(residuals):
    residuals = residuals[::-1]

    # Ensure residuals are in numpy array format
    residuals = np.array(residuals)

    # Fit an AR(1) model to the residuals
    model = AutoReg(residuals, lags=1).fit()

    # Get the autoregressive coefficient (phi)
    phi = model.params[1]

    # Calculate the half-life
    half_life = -np.log(2) / np.log(abs(phi))

    return half_life


def calculate_half_life(df_residuals, pair):
    hl_gpt = gpt_half_life(df_residuals[pair])
    hl_blog = blog_half_life(df_residuals[pair])
    return np.round(np.mean([hl_blog, hl_gpt]), 0)


def double_cointegration(df, check=False):
    if check == True:
        lst_pairs = list(df.pair)
        double_cointegration_pairs = []
        for pair in lst_pairs:
            inverted_pair = pair.split("/")[1] + "/" + pair.split("/")[0]
            if inverted_pair in lst_pairs:
                double_cointegration_pairs.append(pair)
        return df[df["pair"].isin(double_cointegration_pairs)]
    else:
        return df


def normalize_column(df, column_name):
    # Calculate mean and standard deviation of the column
    mean = df[column_name].mean()
    std = df[column_name].std()

    # Normalize the column
    df[column_name] = (df[column_name] - mean) / std

    return df[column_name]


def calculatePairStats(args):
    df_stocks = args["dataset"]
    pair = args["pair"]
    periods = args["periods"]
    date_col = args["date_col"]

    column_names = ["pair", "adf", "beta", "desv", "op_type", "halflife", "p_value"]

    asset1 = pair.split("/")[0]
    asset2 = pair.split("/")[1]

    df_res = df_stocks[[date_col, asset1, asset2]].sort_values(
        by=date_col, ascending=False
    )[:periods]

    X = df_res.iloc[:, 1].values.reshape(-1, 1)
    Y = df_res.iloc[:, 2].values.reshape(-1, 1)

    # Ignoring pairs with NaN values
    if df_res.iloc[:, 1].isna().sum() > 0 or df_res.iloc[:, 2].isna().sum() > 0:
        return None

    linear_regressor = LinearRegression()
    linear_regressor.fit(X, Y)

    Y_pred = linear_regressor.predict(X)
    df_res[pair] = Y - Y_pred  # Residuals

    df_res = df_res[[date_col, pair]]

    mean = df_res[pair].mean()
    std = df_res[pair].std()
    adf = adfuller(df_res[pair])
    beta = np.round(linear_regressor.coef_[0][0], 2)
    desv = np.round(((df_res[pair].iloc[0] - mean) / std), 2)
    op_type = "V" if desv > 0 else "V"
    half_life = calculate_half_life(df_res, pair)

    lst_values = [
        pair,
        adf[0],
        abs(beta),
        desv,
        op_type,
        half_life,
        np.round((1 - adf[1]) * 100, 2),
    ]
    df_stats = pd.DataFrame([lst_values], columns=column_names)

    return df_stats


def getPairsStats(args, workers):
    combined_stats = pd.DataFrame()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit tasks to the executor
        results = [executor.submit(calculatePairStats, arg) for arg in args]

    # Retrieve results as they become available
    for future in concurrent.futures.as_completed(results):
        df_stats = future.result()
        if df_stats is None:
            continue
        combined_stats = pd.concat([combined_stats, df_stats], ignore_index=True)

    return combined_stats
