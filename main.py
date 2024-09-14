import glob
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import pickle

# Create directory for saving models
os.makedirs('weights', exist_ok=True)

# Load datasets from multiple CSV files
print("Loading dataset from multiple CSV files...")
file_paths = glob.glob('dataset/*.csv')
data = pd.DataFrame()

for file_path in file_paths:
    print(f"Reading {file_path}...")
    temp_data = pd.read_csv(file_path)
    data = pd.concat([data, temp_data], ignore_index=True)

# Convert datetime column to correct format and set as index
print("Converting 'tpep_pickup_datetime' column to datetime format...")
data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])

print("Setting 'tpep_pickup_datetime' as index for time series analysis...")
data = data.set_index('tpep_pickup_datetime')

# Resample data to daily frequency
print("Resampling data to daily frequency...")
daily_data = data.resample('D').size()

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
train_size = int(len(daily_data) * 0.8)
train, test = daily_data[:train_size], daily_data[train_size:]

# Function to evaluate models with multiple metrics
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f'{model_name} - Mean Squared Error (MSE): {mse:.2f}')
    print(f'{model_name} - Mean Absolute Error (MAE): {mae:.2f}')
    return mse, mae

# Train ARIMA model with hyperparameter tuning
print("Training ARIMA model...")
best_arima_order = None
best_arima_mse = float('inf')
for p in range(1, 6):  # Example parameter range
    for d in range(0, 2):
        for q in range(0, 2):
            try:
                arima_model = ARIMA(train, order=(p, d, q))
                arima_fit = arima_model.fit()
                arima_forecast = arima_fit.forecast(steps=len(test))
                mse, mae = evaluate_model(test, arima_forecast, f"ARIMA (p={p}, d={d}, q={q})")
                if mse < best_arima_mse:
                    best_arima_mse = mse
                    best_arima_order = (p, d, q)
            except Exception as e:
                print(f"ARIMA (p={p}, d={d}, q={q}) failed: {e}")

print(f"Best ARIMA model: order={best_arima_order}, MSE={best_arima_mse:.2f}")
arima_model = ARIMA(train, order=best_arima_order)
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=len(test))
evaluate_model(test, arima_forecast, "Best ARIMA")
print("Saving ARIMA model...")
arima_fit.save('weights/arima_model.pkl')

# Train Exponential Smoothing (ETS) model with different seasonal periods
print("Training Exponential Smoothing (ETS) model...")
best_ets_mse = float('inf')
best_ets_params = None
for season_period in [7, 14, 30]:  # Different seasonal periods
    try:
        ets_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=season_period)
        ets_fit = ets_model.fit()
        ets_forecast = ets_fit.forecast(steps=len(test))
        mse, mae = evaluate_model(test, ets_forecast, f"ETS (Seasonal Period={season_period})")
        if mse < best_ets_mse:
            best_ets_mse = mse
            best_ets_params = season_period
    except Exception as e:
        print(f"ETS (Seasonal Period={season_period}) failed: {e}")

print(f"Best ETS model: Seasonal Period={best_ets_params}, MSE={best_ets_mse:.2f}")
ets_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=best_ets_params)
ets_fit = ets_model.fit()
ets_forecast = ets_fit.forecast(steps=len(test))
evaluate_model(test, ets_forecast, "Best Exponential Smoothing (ETS)")
print("Saving ETS model...")
ets_fit.save('weights/ets_model.pkl')

# Train Prophet model
print("Training Prophet model...")
prophet_data = daily_data.reset_index()
prophet_data.columns = ['ds', 'y']
prophet_model = Prophet()
prophet_model.fit(prophet_data.iloc[:train_size])
prophet_future = prophet_model.make_future_dataframe(periods=len(test))
prophet_forecast = prophet_model.predict(prophet_future)['yhat'][-len(test):]
evaluate_model(test.values, prophet_forecast, "Prophet")
print("Saving Prophet model...")
with open('weights/prophet_model.pkl', 'wb') as f:
    pickle.dump(prophet_model, f)

# Plotting results
print("Plotting results...")
plt.figure(figsize=(12, 8))
plt.plot(test.index, test, label='Actual', color='black')
plt.plot(test.index, arima_forecast, label='ARIMA Forecast')
plt.plot(test.index, ets_forecast, label='ETS Forecast')
plt.plot(test.index, prophet_forecast, label='Prophet Forecast')
plt.legend()
plt.title('Time Series Forecasting Comparison')
plt.show()

# Additional plot for Mean Absolute Error (MAE) comparison
plt.figure(figsize=(12, 8))
plt.bar(['ARIMA', 'ETS', 'Prophet'], [mean_absolute_error(test, arima_forecast), 
                                       mean_absolute_error(test, ets_forecast),
                                       mean_absolute_error(test, prophet_forecast)])
plt.title('Mean Absolute Error (MAE) Comparison')
plt.ylabel('MAE')
plt.show()
