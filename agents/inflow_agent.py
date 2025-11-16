import requests
import pandas as pd
from datetime import datetime, timedelta
import lightgbm as lgb
from utils.helpers import load_and_clean_data, TUNNEL_CONSTANTS

class InflowAgent:
    def __init__(self, data_file="data/hsy_data.csv"):
        self.history_df = load_and_clean_data(data_file)
        # LightGBM is generally better than Random Forest here
        self.model = lgb.LGBMRegressor(objective='regression', metric='rmse', n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
        self.target_col = 'inflow_to_tunnel_f1' # m³/15 min

    def fetch_weather_forecast(self, n_steps=96):
        """
        Fetches 24-hour weather forecast data (precipitation) for Helsinki.
        """
        lat, lon = 60.17, 24.94 # Helsinki, FI (Approx. for HSY area)
        # Requesting precipitation and snow depth
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=precipitation,snowfall&timezone=Europe%2FHelsinki"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            weather_df = pd.DataFrame({
                'time_stamp': pd.to_datetime(data['hourly']['time']),
                'rain_mm': data['hourly']['precipitation'],
                'snow_cm': data['hourly']['snowfall']
            }).set_index('time_stamp')
            
            # Resample hourly data to 15-minute intervals and forward-fill
            weather_15min = weather_df.resample('15min').ffill().dropna()
            
            # Get the next 24 hours (96 steps) of forecast
            last_op_time = self.history_df.index[-1]
            future_index = pd.date_range(start=last_op_time + timedelta(minutes=15), 
                                         periods=n_steps, 
                                         freq='15min')
            
            return weather_15min[weather_15min.index.isin(future_index)].iloc[:n_steps]
            
        except Exception as e:
            print(f"Error fetching weather data: {e}. Using zero rainfall assumption for forecast steps.")
            future_index = pd.date_range(start=self.history_df.index[-1] + timedelta(minutes=15), 
                                         periods=n_steps, 
                                         freq='15min')
            return pd.DataFrame({'rain_mm': 0.0, 'snow_cm': 0.0}, index=future_index)

    def forecast_inflow(self, n_steps=96):
        """Trains an ML model to predict inflow based on time, lag, and weather."""
        
        # 1. Prepare Features: Historical data with lags and time
        df_hist = self.history_df.copy()
        df_hist['hour'] = df_hist.index.hour
        df_hist['dayofweek'] = df_hist.index.dayofweek
        
        # Lagged inflow is the best predictor
        for lag in [1, 4, 24]: # 15min, 1hr, 6hr (15min steps)
            df_hist[f'{self.target_col}_lag_{lag}'] = df_hist[self.target_col].shift(lag)
            
        # For simplicity, we assume historical weather data is not perfectly available,
        # so we train primarily on time and lag features, which capture the daily cycle.
        df_hist = df_hist.dropna()
        
        features = [col for col in df_hist.columns if 'lag' in col or col in ['hour', 'dayofweek']]
        
        # 2. Train the model
        X_train = df_hist[features]
        # print(f"Training features shape: {X_train}")
        
        y_train = df_hist[self.target_col]
        self.model.fit(X_train, y_train)
        print("Inflow forecast model trained.")

        # 3. Create prediction input with guaranteed n_steps rows
        # Build the time index for the forecast period
        last_op_time = self.history_df.index[-1]
        forecast_index = pd.date_range(start=last_op_time + timedelta(minutes=15), 
                                       periods=n_steps, 
                                       freq='15min')
        
        # Create X_pred with time features (guaranteed to have n_steps rows)
        X_pred = pd.DataFrame(index=forecast_index)
        X_pred['hour'] = X_pred.index.hour
        X_pred['dayofweek'] = X_pred.index.dayofweek
        
        # Optionally fetch weather, but don't fail if it returns fewer rows
        try:
            weather_forecast = self.fetch_weather_forecast(n_steps)
            # Only add weather columns if they exist and align
            for col in weather_forecast.columns:
                X_pred[col] = weather_forecast[col]
        except Exception as e:
            print(f"Weather forecast skipped ({e}). Using time features only.")

        # Autoregressive prediction: use predicted inflow to predict the next step
        last_values = df_hist[features].iloc[-1].to_dict()
        inflow_forecast = []
        
        for i in range(n_steps):
            # Combine current time/lag features
            current_features = {k: [last_values[k]] for k in features if 'lag' in k}
            current_features['hour'] = [X_pred.iloc[i]['hour']]
            current_features['dayofweek'] = [X_pred.iloc[i]['dayofweek']]
            
            # Predict the next inflow value
            next_inflow = self.model.predict(pd.DataFrame(current_features))[0]
            inflow_forecast.append(max(0, next_inflow)) # Inflow cannot be negative
            
            # Update lag features for the next iteration
            for lag in sorted([24, 4, 1], reverse=True):
                # Shift lags: lag_2 becomes lag_3, lag_1 becomes lag_2
                lag_key = f'{self.target_col}_lag_{lag}'
                prev_lag_key = f'{self.target_col}_lag_{lag-1}' if lag > 1 else None
                
                if prev_lag_key in last_values:
                    last_values[lag_key] = last_values[prev_lag_key]
                elif lag_key in last_values:
                    last_values[lag_key] = next_inflow # Fallback to new prediction
            
            last_values[f'{self.target_col}_lag_1'] = next_inflow
        
        return pd.Series(inflow_forecast, index=X_pred.index, name='inflow_forecast_m3_15min')