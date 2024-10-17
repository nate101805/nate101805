import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Data input
data = {
    "Dates": [
        "10/31/20", "11/30/20", "12/31/20", "1/31/21", "2/28/21", "3/31/21", "4/30/21", "5/31/21", 
        "6/30/21", "7/31/21", "8/31/21", "9/30/21", "10/31/21", "11/30/21", "12/31/21", "1/31/22", 
        "2/28/22", "3/31/22", "4/30/22", "5/31/22", "6/30/22", "7/31/22", "8/31/22", "9/30/22", 
        "10/31/22", "11/30/22", "12/31/22", "1/31/23", "2/28/23", "3/31/23", "4/30/23", "5/31/23", 
        "6/30/23", "7/31/23", "8/31/23", "9/30/23", "10/31/23", "11/30/23", "12/31/23", "1/31/24", 
        "2/29/24", "3/31/24", "4/30/24", "5/31/24", "6/30/24", "7/31/24", "8/31/24", "9/30/24"
    ],
    "Prices": [
        10.1, 10.3, 11.0, 10.9, 10.9, 10.9, 10.4, 9.84, 10.0, 10.1, 10.3, 10.2, 10.1, 11.2, 11.4, 
        11.5, 11.8, 11.5, 10.7, 10.7, 10.4, 10.5, 10.4, 10.8, 11.0, 11.6, 11.6, 12.1, 11.7, 12.0, 
        11.5, 11.2, 10.9, 11.4, 11.1, 11.5, 11.8, 12.2, 12.8, 12.6, 12.4, 12.7, 12.1, 11.4, 11.5, 
        11.6, 11.5, 11.8
    ]
}

# Convert data to DataFrame
df = pd.DataFrame(data)
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')

# Convert dates to ordinal (days since 01-01-0001)
df['Date_Ordinal'] = df['Dates'].map(datetime.toordinal)

# Prepare input (X) and output (y) for the model
X = df[['Date_Ordinal']].values
y = df['Prices'].values

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Function to estimate gas prices for a given date and extrapolate into the future
def estimate_price(input_date_str, forecast_years=1):
    # Parse input date
    input_date = datetime.strptime(input_date_str, "%m/%d/%y")
    input_date_ordinal = input_date.toordinal()
    
    # Predict the price for the given date
    estimated_price = model.predict([[input_date_ordinal]])[0]
    
    # Extrapolate prices for the next 'forecast_years' into the future
    future_dates = [input_date + timedelta(days=365 * i) for i in range(forecast_years + 1)]
    future_prices = model.predict([[date.toordinal()] for date in future_dates])
    
    return estimated_price, list(zip(future_dates, future_prices))

# Example: Estimate the price for 10/01/23 and forecast one year into the future
estimate_date = '10/01/23'
estimated_price, future_predictions = estimate_price(estimate_date, forecast_years=1)

estimated_price, future_predictions
