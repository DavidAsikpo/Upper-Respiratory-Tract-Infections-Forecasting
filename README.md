# 🩺 Time Series Forecasting of Childhood Upper Respiratory Tract Infections (URTI)
**Author:** David Asikpo  
**Institution:** General Hospital, Ikot Ekpene, Akwa Ibom State, Nigeria. 
**Study Period:** January 2010 – June 2025  
**Model Used:** ARIMA(3,1,0)

---

## 📖 Project Overview

This project performs a **time series analysis** of monthly reported cases of **Childhood Upper Respiratory Tract Infections (URTI)** of children under the age of 14 at *General Hospital, Ikot Ekpene* from **January 2010 to June 2025**.  
The goal is to model past incidence trends, assess seasonality, and forecast future cases to support **public health planning and preventive interventions**, which will help the hospital plan better in resources and labour power for proper handling of future cases.

The analysis applies the **ARIMA (AutoRegressive Integrated Moving Average)** methodology — a robust statistical model for forecasting time-dependent data.

---

## 🎯 Objectives

1. **Fit an appropriate ARIMA model** to monthly URTI cases.  
2. **Assess the trend, variability, and distribution** of infections over 15 years.  
3. **Test for stationarity** using the Augmented Dickey-Fuller (ADF) test.  
4. **Forecast the incidence of URTI cases** for the next 6 months (Jan–Jun 2025).  
5. Provide actionable insights for **epidemiological response planning**.

---

## 🧠 Methodology

### 1. Data Description
- The dataset consists of monthly URTI case counts for children aged 0–14 years.  
- Total observations: **186 months (Jan 2010 – Jun 2025)**  
- Variables:
  - `Month` — month of observation  
  - `Year` — year of observation  
  - `Cases` — number of reported URTI cases  

---

### 2. Exploratory Data Analysis (EDA)
The exploratory analysis provided insight into:
- Seasonal variations — higher cases observed during **Dry season months (Jan–Mar)**.
- Decrease during **wet season (Apr–Aug)** due to higher humidity and cleaner air.
- Gradual increase again from **September to December**.

Statistic            | Value       |
|:-------------------|:-----------:|
| Mean                | 88.48      |
| Standard Error      | 4.39       |
| Median              | 69         |
| Mode                | 39         |
| Standard Deviation  | 58.83      |
| Sample Variance     | 3461.42    |
| Kurtosis            | 0.53       |
| Skewness            | 1.12       |
| Range               | 242        |
| Minimum             | 8          |
| Maximum             | 250        |
| Sum                 | 15926      |

---

### 3. Stationarity Testing
The **Augmented Dickey-Fuller (ADF) Test** was applied to the time series.

- **Null Hypothesis (H₀):** Series is non-stationary  
- **Alternative Hypothesis (H₁):** Series is stationary  
- ADF Statistic: -4.12  
- p-value: 0.01  

✅ Result: Since p < 0.05, the differenced series is **stationary**.

---

### 4. Model Identification & Selection
Candidate models were evaluated using **AIC** and **BIC** values:

| Model | AIC | BIC |
|:------|----:|----:|
| ARIMA(1,1,1) | 1875.21 | 1886.42 |
| ARIMA(2,1,0) | 1873.56 | 1884.31 |
| **ARIMA(3,1,0)** | **1870.30** | **1883.05** |

The **ARIMA(3,1,0)** model was chosen as optimal due to the lowest AIC and BIC values and well-behaved residuals.

---

### 5. Model Equation

\[
Y_t = -0.417Y_{t-1} - 0.431Y_{t-2} - 0.428Y_{t-3} + \epsilon_t
\]

---

### 6. Model Diagnostics
Residual diagnostic checks confirm:
- ACF/PACF of residuals are within ±0.149 → residuals are **white noise**.  
- Residuals are uncorrelated → **model is adequate**.  
- Ljung-Box test p-value > 0.05 → model residuals show no serial correlation.

---

### 7. Testing Model Accuracy
Using the **ARIMA(3,1,0)** model, forecasts for Jan–Jun 2025 were generated to test the accuracy of the model against actual data.

| Month     | Forecasted Cases | Test Data | 95% CI (Lower–Upper) |
|:---------|:----------------:|:---------:|:-------------------:|
| Jan 2025 | 232             | 229       | 146–319             |
| Feb 2025 | 206             | 198       | 106–306             |
| Mar 2025 | 210             | 201       | 105–314             |
| Apr 2025 | 226             | 209       | 121–332             |
| May 2025 | 229             | 210       | 114–344             |
| Jun 2025 | 219             | 203       | 94–344              |

---

### 6. Forecasting Results
Using the **ARIMA(3,1,0)** model,since we can the forecasts for Jan–Jun 2025 were very close to the actual data, a forcast of 12 months was further generated to boost business decisoons for the hospital.

| Month       | Forecasted Cases | 95% CI (Lower–Upper) |
|:-----------|:----------------:|:-------------------:|
| Jan 2025   | 232             | 146–319             |
| Feb 2025   | 206             | 106–306             |
| Mar 2025   | 210             | 105–314             |
| Apr 2025   | 226             | 121–332             |
| May 2025   | 229             | 114–344             |
| Jun 2025   | 219             | 94–344              |
| Jul 2025   | 215             | 84–346              |
| Aug 2025   | 220             | 85–354              |
| Sep 2025   | 224             | 85–363              |
| Oct 2025   | 222             | 76–367              |
| Nov 2025   | 219             | 68–370              |
| Dec 2025   | 219             | 64–374              |


📈 **Insight:** The trend indicates a gradual increase in URTI cases in 2025.

---

## 🧰 Tools & Libraries

| Category | Tools Used |
|-----------|-------------|
| Programming Language | Python 3.10 |
| Data Analysis | pandas, numpy |
| Time Series Modeling | statsmodels |
| Visualization | matplotlib |
| Model Diagnostics | ADF Test, AIC/BIC, Residual Analysis |

---

## 🧩 Folder Structure

