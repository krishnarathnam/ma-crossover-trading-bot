from backtesting import Backtest, Strategy
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from fetch_data_td import fetch_data_td


# --------------------------------------------
# STEP 1 — Build Spread Model (cointegration)
# --------------------------------------------
def build_spread(df1, df2):
    df1, df2 = df1.align(df2, join="inner")

    X = df1["close"].values
    Y = df2["close"].values

    # Regression Y = aX + b
    a, b = np.polyfit(X, Y, 1)

    spread = Y - (a * X + b)
    spread_mean = spread.mean()
    spread_std = spread.std(ddof=1)
    upper = spread_mean + 2 * spread_std
    lower = spread_mean - 2 * spread_std

    adf_p = adfuller(spread)[1]
    print("\nADF p-value:", adf_p)

    df = pd.DataFrame({
        "X": X,
        "Y": Y,
        "spread": spread,
        "upper": upper,
        "lower": lower,
        "beta": a
    }, index=df1.index)

    return df


# --------------------------------------------
# STEP 2 — Generate Pair-Trading Signals
# --------------------------------------------
def generate_signals(df):
    df["long_spread"] = df["spread"] < df["lower"]
    df["short_spread"] = df["spread"] > df["upper"]

    # Exit when spread returns to mean
    df["exit"] = (df["spread"] * df["spread"].shift(1) < 0)

    return df


# --------------------------------------------
# STEP 3 — Pair Trading Backtest
# --------------------------------------------
def backtest_pairs(df, capital=10000, leverage=20):
    df = df.copy()

    df["pos_X"] = 0.0
    df["pos_Y"] = 0.0
    df["pnl"] = 0.0

    in_trade = False
    long_spread = False

    for i in range(1, len(df)):

        if not in_trade:

            # LONG SPREAD → long Y, short X
            if df["long_spread"].iloc[i]:
                in_trade = True
                long_spread = True
                df.loc[df.index[i], "pos_Y"] = +1
                df.loc[df.index[i], "pos_X"] = -df["beta"].iloc[i]

            # SHORT SPREAD → short Y, long X
            elif df["short_spread"].iloc[i]:
                in_trade = True
                long_spread = False
                df.loc[df.index[i], "pos_Y"] = -1
                df.loc[df.index[i], "pos_X"] = +df["beta"].iloc[i]

        else:
            # Exit when spread crosses mean
            if df["exit"].iloc[i]:
                in_trade = False
                df.loc[df.index[i], "pos_Y"] = 0
                df.loc[df.index[i], "pos_X"] = 0

            else:
                # carry positions forward
                df.loc[df.index[i], "pos_Y"] = df["pos_Y"].iloc[i-1]
                df.loc[df.index[i], "pos_X"] = df["pos_X"].iloc[i-1]

        # PnL calculation
        dX = df["X"].iloc[i] - df["X"].iloc[i - 1]
        dY = df["Y"].iloc[i] - df["Y"].iloc[i - 1]

        pnl = df["pos_Y"].iloc[i] * dY + df["pos_X"].iloc[i] * dX
        df.loc[df.index[i], "pnl"] = pnl

    df["cum_pnl"] = df["pnl"].cumsum()
    df["equity"] = capital + df["cum_pnl"] * leverage

    return df


# --------------------------------------------
# RUN EVERYTHING
# --------------------------------------------
def run_pair_backtest(sym1="EUR/USD", sym2="USD/CHF", interval="1day", bars=2000):
    print("\nFetching data...")
    df1 = fetch_data_td(sym1, interval, bars)
    df2 = fetch_data_td(sym2, interval, bars)

    model = build_spread(df1, df2)
    model = generate_signals(model)
    res = backtest_pairs(model)

    print("\nFinal Equity:", res["equity"].iloc[-1])
    print("Total Return: {:.2f}%".format(
        (res["equity"].iloc[-1] - res["equity"].iloc[0]) / res["equity"].iloc[0] * 100
    ))

    return res


# Run it
if __name__ == "__main__":
    df = run_pair_backtest("EUR/USD", "USD/CHF", "4h", 5000)
    print(df.tail())