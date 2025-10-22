# urti_forecast.py
# Time Series Forecasting of Childhood URTI Cases

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# ----------------------------
# 1. Load and prepare dataset
# ----------------------------
# Observation data provided in a nested format: years as columns, months as rows
data_dict = {
    "Jan":[40,62,45,132,104,77,96,76,76,32,68,149,181,135,34,229],
    "Feb":[250,44,15,81,111,75,54,77,86,28,97,183,68,128,40,198],
    "Mar":[231,72,47,65,144,91,27,72,124,26,134,193,103,101,250,201],
    "Apr":[18,29,43,56,55,107,67,81,145,33,37,93,116,109,162,209],
    "May":[17,25,44,76,99,76,69,47,97,17,39,103,168,114,190,210],
    "Jun":[55,53,46,45,80,86,57,45,118,34,117,103,242,123,175,203],
    "Jul":[46,49,62,66,105,69,49,110,122,49,141,179,87,87,216],
    "Aug":[39,77,30,16,62,151,52,67,68,50,43,66,128,64,235],
    "Sep":[37,47,18,46,68,123,57,68,72,46,83,58,115,167,211],
    "Oct":[43,39,8,53,39,141,24,39,132,37,227,102,216,192,171],
    "Nov":[29,85,17,57,39,105,17,39,69,69,131,131,235,213,215],
    "Dec":[19,54,30,49,61,64,31,26,64,73,109,79,230,39,248]
}

years = list(range(2010, 2026))
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# Convert to long-format dataframe
records = []
for i, year in enumerate(years):
    for month in months:
        # Some years (2025) have fewer months (Jan-Jun)
        try:
            value = data_dict[month][i]
            records.append({"Year":year,"Month":month,"Cases":value})
        except IndexError:
            continue

df = pd.DataFrame(records)

# Create datetime index
df['Date'] = pd.to_datetime(df['Month'] + '-' + df['Year'].astype(str))
df.set_index('Date', inplace=True)
df = df[['Cases']].sort_index()

# ----------------------------
# 2. EDA
# ----------------------------
print("\nDescriptive Statistics:\n", df['Cases'].describe())
print("\nSkewness:", df['Cases'].skew())
print("Kurtosis:", df['Cases'].kurt())

# Plotting
plt.figure(figsize=(12,5))
plt.plot(df['Cases'], marker='o', linestyle='-')
plt.title("Monthly Childhood URTI Cases (Jan 2010 – Jun 2025)")
plt.xlabel("Date")
plt.ylabel("Cases")
plt.grid(True)
plt.show()

# ----------------------------
# 3. Stationarity Test
# ----------------------------
adf_result = adfuller(df['Cases'].diff().dropna())  # differenced for ARIMA(d=1)
print("\nADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
if adf_result[1] < 0.05:
    print("Series is stationary after differencing.")
else:
    print("Series is non-stationary.")

# ----------------------------
# 4. ARIMA Model Fitting
# ----------------------------
# Candidate models AIC/BIC comparison
models = [(1,1,1),(2,1,0),(3,1,0)]
for order in models:
    model = ARIMA(df['Cases'], order=order)
    model_fit = model.fit()
    print(f"\nARIMA{order} -> AIC: {model_fit.aic:.2f}, BIC: {model_fit.bic:.2f}")

# Fit the chosen model: ARIMA(3,1,0)
arima_model = ARIMA(df['Cases'], order=(3,1,0))
arima_result = arima_model.fit()
print("\nARIMA(3,1,0) Summary:\n", arima_result.summary())

# ----------------------------
# 5. Residual Diagnostics
# ----------------------------
residuals = arima_result.resid
plt.figure(figsize=(12,5))
plt.plot(residuals)
plt.title("Residuals from ARIMA(3,1,0) Model")
plt.show()

plt.figure(figsize=(12,5))
plt.hist(residuals, bins=20)
plt.title("Histogram of Residuals")
plt.show()

# ----------------------------
# 6. Forecasting
# ----------------------------
# Forecast Jan-Jun 2025 (6 months test)
forecast_6 = arima_result.get_forecast(steps=6)
forecast_6_df = pd.DataFrame({
    "Forecast": forecast_6.predicted_mean.round(0),
    "Lower CI": forecast_6.conf_int()['lower Cases'].round(0),
    "Upper CI": forecast_6.conf_int()['upper Cases'].round(0)
}, index=pd.date_range(start='2025-01-01', periods=6, freq='MS'))
print("\nForecast Jan-Jun 2025:\n", forecast_6_df)

# Forecast Jan-Dec 2025 (12 months extended)
forecast_12 = arima_result.get_forecast(steps=12)
forecast_12_df = pd.DataFrame({
    "Forecast": forecast_12.predicted_mean.round(0),
    "Lower CI": forecast_12.conf_int()['lower Cases'].round(0),
    "Upper CI": forecast_12.conf_int()['upper Cases'].round(0)
}, index=pd.date_range(start='2025-01-01', periods=12, freq='MS'))
print("\nExtended Forecast Jan-Dec 2025:\n", forecast_12_df)

# ----------------------------
# 7. Plot Forecast vs Actual
# ----------------------------
plt.figure(figsize=(12,5))
plt.plot(df['Cases'], label='Observed', marker='o')
plt.plot(forecast_12_df['Forecast'], label='Forecast', marker='x')
plt.fill_between(forecast_12_df.index,
                 forecast_12_df['Lower CI'],
                 forecast_12_df['Upper CI'], color='pink', alpha=0.3)
plt.title("ARIMA(3,1,0) Forecast vs Observed URTI Cases")
plt.xlabel("Date")
plt.ylabel("Cases")
plt.legend()
plt.show()
