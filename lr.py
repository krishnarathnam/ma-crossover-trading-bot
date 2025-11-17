from fetch_data_yf import fetch_data_yf
from fetch_data_td import fetch_data_td
import numpy as np
from statsmodels.tsa.stattools import adfuller

# ------ INPUT ------
# df1 = fetch_data_yf("HDFCBANK.NS", "5y", "1d")
# df2 = fetch_data_yf("ICICIBANK.NS", "5y", "1d")
df1 = fetch_data_td("EUR/USD", "1day", 1000)
df2 = fetch_data_td("USD/CHF", "1day", 1000)

# Align dataframes on common dates
df1_aligned, df2_aligned = df1.align(df2, join='inner', axis=0)

corr = df1_aligned['close'].corr(df2_aligned['close'])
print("Correlation:", corr)


def regression_stats(X, Y):
    """
    Linear regression Y = aX + b
    Returns: slope, intercept, residuals, SE, SE_intercept, error_ratio
    """

    X = np.array(X)
    Y = np.array(Y)

    # Regression
    a, b = np.polyfit(X, Y, 1)
    y_pred = a * X + b

    # Residuals
    residuals = Y - y_pred

    # -------- ADF TEST ONLY (no other tests) --------
    adf_result = adfuller(residuals)
    p_value = adf_result[1]
    print("\nADF p-value:", p_value)
    if p_value < 0.05:
        print("Residuals stationary → cointegration exists")
    else:
        print("Residuals NOT stationary → no cointegration")

    # Standard error of residuals
    SE = np.std(residuals, ddof=1)

    # SE(intercept)
    n = len(X)
    x_mean = np.mean(X)
    Sxx = np.sum((X - x_mean)**2)
    s = np.sqrt(np.sum(residuals**2) / (n - 2))
    SE_intercept = s * np.sqrt((1/n) + (x_mean**2)/Sxx)

    # Error Ratio
    error_ratio = SE_intercept / SE
    res_mean = np.mean(residuals)
    res_std  = np.std(residuals, ddof=1)

    upper_2sd = res_mean + 2 * res_std
    lower_2sd = res_mean - 2 * res_std
    current_residual = residuals[-1]

    return {
        "slope": a,
        "intercept": b,
        "residuals": residuals,
        "current_residual": current_residual,
        "SE": SE,
        "SE_intercept": SE_intercept,
        "error_ratio": error_ratio,
        "res_mean": res_mean,
        "res_std": res_std,
        "upper_2sd": upper_2sd,
        "lower_2sd": lower_2sd
    }


def choose_X_Y(df1, df2):
    print("\nCase 1: X = stock1, Y = stock2")
    r1 = regression_stats(df1, df2)
    # Get and print the SE for the most recent (latest) data point
    latest_index = -1  # assuming last value is latest
    print("SE (Case 1, latest):", r1["SE"])
    print("Current Residual (Case 1, latest):", r1["current_residual"])
    print("+2 SD:", r1["upper_2sd"])
    print("-2 SD:", r1["lower_2sd"])

    print("\nCase 2: X = stock2, Y = stock1")
    r2 = regression_stats(df2, df1)
    print("SE (Case 2, latest):", r2["SE"])
    print("Current Residual (Case 2, latest):", r2["current_residual"])
    print("+2 SD:", r2["upper_2sd"])
    print("-2 SD:", r2["lower_2sd"])

    if r1["error_ratio"] < r2["error_ratio"]:
        print("\nFINAL → Independent: stock1, Dependent: stock2")
        return {"X": "stock1", "Y": "stock2", "stats": r1}
    else:
        print("\nFINAL → Independent: stock2, Dependent: stock1")
        return {"X": "stock2", "Y": "stock1", "stats": r2}


# ---- RUN ----
choose_X_Y(df1_aligned["close"], df2_aligned["close"])