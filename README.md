# 📊 AAPL Long-Term Stock Price Prediction

This project aims to predict **Apple Inc. (AAPL)** stock prices using historical data and machine learning models. The notebook performs complete **data preprocessing**, **feature engineering**, **statistical analysis**, and **model evaluation** to forecast long-term stock trends.

---

## 1. Data Cleaning & Preprocessing

* **Data Source:** Historical AAPL stock data retrieved via *Yahoo Finance API (yfinance)*.
* **Initial Steps:**

  * Checked for **null or missing values** using:

    ```python
    df.isnull().sum()
    ```

    → No significant missing data was found.
  * Verified data types (Date, Float, Int) and converted the **Date** column to a proper datetime format.
  * Sorted the dataset chronologically and set the **Date** as index.

---

## 2. Statistical Analysis & Stationarity Tests

Before building predictive models, the time series was examined for **volatility**, **stationarity**, and **autocorrelation**:

* **Descriptive Statistics:**

  ```python
  df.describe()
  ```

  → Provided mean, standard deviation, min, max, and quartile spread of closing prices.

* **ADF (Augmented Dickey-Fuller) Test:**
  Used to test **stationarity** (null hypothesis: data is non-stationary).

  ```python
  from statsmodels.tsa.stattools import adfuller
  adf_result = adfuller(df['Close'])
  ```

  * **p-value > 0.05** indicated that the data was *non-stationary*, confirming the need for differencing or feature transformation.

* **Volatility Check:**
  Rolling standard deviation was plotted to visually confirm **price variance over time**.

  ```python
  df['volatility'] = df['Close'].rolling(window=30).std()
  ```

---

## 3. Feature Engineering

To help the model understand market dynamics, several **technical indicators** were added:

| Feature                           | Description                                                              |
| --------------------------------- | ------------------------------------------------------------------------ |
| **SMA_20, SMA_50**                | 20-day and 50-day Simple Moving Averages                                 |
| **EMA_20**                        | Exponential Moving Average to emphasize recent prices                    |
| **RSI (Relative Strength Index)** | Measures momentum and overbought/oversold conditions                     |
| **MACD**                          | Captures trend and momentum via short-term and long-term moving averages |
| **Volatility (30D)**              | Rolling standard deviation as a measure of risk                          |

Example code:

```python
df['SMA_20'] = df['Close'].rolling(20).mean()
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
```

---

## 4. Model Development

* **Data Split:**
  80% training, 20% testing.

* **Model Used:**
  An **LSTM (Long Short-Term Memory)** neural network trained to predict future closing prices based on the engineered features.

* **Training Parameters:**

  * Sequence length: 60 days
  * Optimizer: Adam
  * Loss function: Mean Squared Error (MSE)

---

## 5. Results & Visualization

### Prediction Graph

AAPL Stock Prediction<img width="1165" height="624" alt="output" src="https://github.com/user-attachments/assets/99938675-0f7e-4d0a-a964-fb6ce7a8a48b" />


* **Blue:** Actual Train Price
* **Orange:** Actual Test Price
* **Green:** Predicted Test Price

---

## 🧾 6. Model Evaluation

| Metric                             | Value  |
| ---------------------------------- | ------ |
| **Mean Squared Error (MSE)**       | 0.0218 |
| **Root Mean Squared Error (RMSE)** | 0.1476 |
| **R-squared (R²)**                 | 0.0183 |

🔍 **Interpretation:**

* The **low R² value** suggests that the model struggled to capture the full complexity of market movements — possibly due to limited feature scope or lack of macroeconomic data.
* However, the **RMSE** indicates that predictions were numerically close to actual normalized values, showing reasonable short-term trend tracking.

---

## 🚀 7. Future Work

To improve predictive accuracy:

* Integrate **macro indicators** (interest rates, inflation, earnings reports).
* Apply **Prophet** or **Transformer-based** time series models.
* Implement **hyperparameter tuning** with larger datasets and feature sets.
Would you like me to rewrite this in a more **academic style** (for inclusion in a report or thesis) or keep it **GitHub-style** (concise and formatted for code repositories)?
