import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = pd.read_csv('Nat_gas.csv') 
# Convert dates to datetime format
data['Dates'] = pd.to_datetime(data['Dates'])
data.set_index('Dates', inplace=True)
# Visualization of the historical data
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Prices'], label='Historical Prices')
plt.xlabel('Dates')
plt.ylabel('Natural Gas Price')
plt.title('Monthly Natural Gas Prices')
plt.legend()
plt.show()
# Extracting features and fitting a model for trend estimation
data['Month'] = data.index.month  # Month number to capture seasonality
X = np.array((data.index - data.index[0]).days).reshape(-1, 1)  # Days as feature
y = data['Prices'].values
model = LinearRegression()
model.fit(X, y)
# Extrapolate prices for the next 12 months
future_dates = [data.index[-1] + timedelta(days=30 * i) for i in range(1, 13)]
future_days = np.array([(date - data.index[0]).days for date in future_dates]).reshape(-1, 1)
future_prices = model.predict(future_days)
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Prices'], label='Historical Prices')
plt.plot(future_dates, future_prices, label='Extrapolated Prices', linestyle='--')
plt.xlabel('Dates')
plt.ylabel('Natural Gas Price')
plt.title('Historical and Extrapolated Natural Gas Prices')
plt.legend()
plt.show()
def estimate_price(date):
    date = pd.to_datetime(date)
    days_since_start = (date - data.index[0]).days
    estimated_price = model.predict(np.array([[days_since_start]]))[0]
    return estimated_price

# Example usage
date_input = '2024-12-15'  # Enter any date for estimation
predicted_price = estimate_price(date_input)
print(f"Estimated price on {date_input}: {predicted_price}")