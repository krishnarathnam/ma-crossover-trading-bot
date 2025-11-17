from fetch_data_td import fetch_data_td
from fetch_data_yf import fetch_data_yf
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np


# df = fetch_data_td("GBP/USD", "1h", 5000)  # More data for better indicator calculations
# df2 = fetch_data_td("EUR/USD", "1h", 5000)  # More data for better indicator calculations
# df = fetch_data_yf("HDFCBANK.NS", "5y", "1d")
df = fetch_data_td("XAU/USD", "1day", 5000)
df2 = fetch_data_td("EUR/CHF", "1day", 5000)

# Align dataframes on common dates
df1_aligned, df2_aligned = df.align(df2, join='inner', axis=0)

corr = df1_aligned['close'].corr(df2_aligned['close'])
print("Correlation:", corr)

# df2 = fetch_data_yf("ICICIBANK.NS", "5y", "1d")
# df = calculate_signals_ou(df)

df['Daily Return'] = df['close'].pct_change()
df2['Daily Return'] = df2['close'].pct_change()

print(df['Daily Return'].corr(df2['Daily Return']))

ratio_df_df2 = df['close'] / df2['close']
ratio_df_df2 = ratio_df_df2.dropna()
kde = gaussian_kde(ratio_df_df2)
x_vals = np.linspace(ratio_df_df2.min(), ratio_df_df2.max(), 200)
density_vals = kde(x_vals)

mean_ratio_df_df2 = ratio_df_df2.mean()
median_ratio_df_df2 = ratio_df_df2.median()
mode_ratio_df_df2 = ratio_df_df2.mode()

std_ratio_df_df2 = ratio_df_df2.std()
std2_ratio_df_df2 = ratio_df_df2.std()**2
std3_ratio_df_df2 = ratio_df_df2.std()**3

neg_std_ratio_df_df2 = -ratio_df_df2.std()
neg_std2_ratio_df_df2 = -ratio_df_df2.std()**2
neg_std3_ratio_df_df2 = -ratio_df_df2.std()**3

# Calculate the daily change of the ratio of close values
# Suppress FutureWarning by specifying fill_method=None
plt.figure(figsize=(12,6))

plt.plot(ratio_df_df2.index, ratio_df_df2, label='Ratio (GBP/USD ÷ EUR/USD)', color='blue')

# Mean and deviation bands
mean_ratio = mean_ratio_df_df2
std_ratio = std_ratio_df_df2

plt.axhline(mean_ratio, color='green', linestyle='--', label='Mean')
plt.axhline(mean_ratio + std_ratio, color='red', linestyle='--', label='+1σ')
plt.axhline(mean_ratio - std_ratio, color='red', linestyle='--', label='-1σ')
plt.axhline(mean_ratio + 2*std_ratio, color='gray', linestyle=':', label='+2σ')
plt.axhline(mean_ratio - 2*std_ratio, color='gray', linestyle=':', label='-2σ')
# Mark even (2 and 4 and 6, etc) std3 in chart. Here we add +3σ, -3σ for illustration.
plt.axhline(mean_ratio + 3*std_ratio, color='orange', linestyle='-.', label='+3σ')
plt.axhline(mean_ratio - 3*std_ratio, color='orange', linestyle='-.', label='-3σ')
# If you need higher even stds:
# plt.axhline(mean_ratio + 6*std_ratio, color='brown', linestyle=':', label='+6σ')
# plt.axhline(mean_ratio - 6*std_ratio, color='brown', linestyle=':', label='-6σ')

plt.title('Pair Ratio with Mean Reversion Bands')
plt.xlabel('Date')
plt.ylabel('Ratio')
plt.show()