from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler  # For feature scaling (optional)
from sklearn.impute import SimpleImputer  # For missing value imputation (optional)
import pandas as pd
import matplotlib.pyplot as plt

def fit_arima(data, order):
  model = ARIMA(data, order=order)
  fitted_model = model.fit()
  predictions = fitted_model.predict(start=1, end=len(data))  # Start from 1 to exclude the first element
  return predictions[1:]  # Return predictions from index 1 onwards

def predict_residuals_rf(data, features, rf_model):
  residuals = data.values[1:] - fit_arima(data, (1, 1, 1))
  if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame(features)
  predicted_residuals = rf_model.predict(features.iloc[-1:].reset_index(drop=True))
  print(type(features))
  return predicted_residuals[0]

data_file = "mem_use_20_Jan_NodeX9WS.csv"  # Replace with your actual CSV file path
data = pd.read_csv(data_file, index_col="time", parse_dates=True)
data = data["memory_usage"]

# Handle missing values 
imputer = SimpleImputer(strategy="mean")  # Replace with preferred imputation strategy
features = pd.DataFrame({'lag_1': imputer.fit_transform(data.values.reshape(-1, 1))[:, 0]})

# Feature scaling 
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.iloc[:-1])  # Exclude potentially added row

# Ensure consistent data lengths (check for discrepancies)
if len(data.values[1:]) != len(fit_arima(data, (1, 1, 1))):
  raise ValueError("Data and ARIMA prediction lengths  is not the same!")
rf_model = RandomForestRegressor()
rf_model.fit(features_scaled, data.values[1:] - fit_arima(data, (1, 1, 1)))

future_steps = 4
arima_preds = fit_arima(data, (1, 1, 1))[len(data):]
future_features = pd.DataFrame({'lag_1': imputer.transform(data.values[-future_steps:].reshape(-1, 1))[:, 0]})  # Impute missing values in future features
future_features_scaled = scaler.transform(future_features)  # Apply scaling if used
print(type(future_features_scaled))

residual_preds = predict_residuals_rf(data, future_features_scaled, rf_model)
final_predictions = arima_preds + residual_preds

plt.plot(data.index, data.values, label="Actual Memory Usage")
plt.plot(data.index[len(data):], arima_preds, label="ARIMA Predictions")
plt.plot(data.index[len(data):], final_predictions, label="Final Hybrid Predictions")
plt.legend()
plt.xlabel("Date/Time")
plt.ylabel("Memory Usage")
plt.title("Hybrid ARIMA-Random Forest Forecast for Memory Usage")
plt.xticks(rotation=45)
plt.tight_layout()  
plt.show()

