import pandas as pd
import requests
from datetime import datetime, timedelta
import lightgbm as lgb
from utils.helpers import load_and_clean_data, TUNNEL_CONSTANTS
import pytz 
from nordpool import elspot # Import the dedicated library

class PriceAgent:
    def __init__(self, data_file="data/hsy_data.csv"):
        self.history_df = load_and_clean_data(data_file)
        self.model = lgb.LGBMRegressor(objective='regression', metric='rmse', n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
        self.target_col = 'electricity_price_1_high' # EUR/kWh
        self.area_code = TUNNEL_CONSTANTS["NORDPOOL_AREA"] # 'FI'
        self.local_tz = pytz.timezone('Europe/Helsinki')

    def fetch_nordpool_prices(self, n_steps=96):
        """
        Fetches 24-hour Day-Ahead prices (hourly) using the Nord Pool library
        and converts them to the required 15-minute intervals.
        """
        print(f"Fetching Nord Pool Day-Ahead prices for area {self.area_code}...")
        
        # 1. Initialize the Nord Pool client
        prices_spot = elspot.Prices()
        
        # 2. Determine the fetch period
        # We need prices for tomorrow. Nord Pool releases prices around 14:00 EET for the next day.
        # The library defaults to fetching tomorrow's prices if 'end_date' is not specified.
        
        # We will attempt to fetch today's and tomorrow's prices to maximize available data
        today = datetime.now(self.local_tz).date()
        # Fetch prices from today up to 48 hours out.
        fetch_start = today 
        fetch_end = today + timedelta(days=2) 
        
        try:
            # The .fetch() method typically returns EUR/MWh
            data = prices_spot.fetch(
                areas=[self.area_code]
                , 
                resolution=15
                # end_date=fetch_end.strftime('%Y-%m-%d')
            )
            
            if not data or 'areas' not in data or self.area_code not in data['areas']:
                raise ValueError("Nord Pool data structure error or area not available.")
            
            # 3. Process the data (already 15-minute resolution)
            values = data['areas'][self.area_code]['values']
            print(f"Fetched {len(values)} price entries from Nord Pool (15-min intervals).")
            price_list = []
            
            for item in values:
                # Timestamps are typically localized to the target timezone by the library
                dt = item['start'].astimezone(self.local_tz)
                # Convert price from EUR/MWh to EUR/kWh
                price = item['value'] / 1000.0
                price_list.append({'timestamp': dt, 'price': price})

            # 4. Create DataFrame from 15-minute prices (no resampling needed)
            prices_15min = pd.DataFrame(price_list).set_index('timestamp')['price'].sort_index()
            
            # Directly use the fetched 15-minute prices (already 96 steps for 24 hours)
            forecast = prices_15min.iloc[:n_steps]
            
            if len(forecast) < n_steps:
                print(f"Warning: Nord Pool API provided less than {n_steps} forecast steps. Falling back to ML forecast.")
                return self._forecast_with_ml(n_steps)

            return forecast.rename('price_forecast_eur_kwh').tz_localize(None) # Remove timezone for simplicity in MPC
            
        except Exception as e:
            print(f"Nord Pool API Error: {e}. Falling back to ML model for forecast.")
            return self._forecast_with_ml(n_steps)

    # --- Fallback ML Model Methods (Kept for Robustness) ---
    
    def _create_features(self, df):
        """Creates time and lag features for ML forecasting."""
        df_new = df.copy()
        df_new['hour'] = df_new.index.hour
        df_new['dayofweek'] = df_new.index.dayofweek
        
        for lag in [1, 4, 24]: # 15min, 1hr, 6hr (15min steps)
            df_new[f'{self.target_col}_lag_{lag}'] = df_new[self.target_col].shift(lag)
        return df_new.dropna()

    def _forecast_with_ml(self, n_steps):
        """Fallback ML forecast using autoregressive LightGBM."""
        df_features = self._create_features(self.history_df)
        features = [col for col in df_features.columns if 'lag' in col or col in ['hour', 'dayofweek']]
        
        X_train = df_features[features]
        y_train = df_features[self.target_col]
        
        # Check if enough data to train
        if len(X_train) == 0:
            print("Insufficient historical data for ML fallback. Returning constant last known price.")
            last_price = self.history_df[self.target_col].iloc[-1]
            future_index = pd.date_range(start=self.history_df.index[-1] + timedelta(minutes=15), periods=n_steps, freq='15min')
            return pd.Series(last_price, index=future_index, name='price_forecast_eur_kwh')


        self.model.fit(X_train, y_train)

        # Autoregressive prediction steps
        X_pred = pd.DataFrame(index=pd.date_range(start=df_features.index[-1] + timedelta(minutes=15), periods=n_steps, freq='15min'))
        X_pred['hour'] = X_pred.index.hour
        X_pred['dayofweek'] = X_pred.index.dayofweek
        
        last_values = df_features[features].iloc[-1].to_dict()
        
        price_forecast = []
        for i in range(n_steps):
            current_features = {k: [last_values[k]] for k in features if 'lag' in k}
            current_features['hour'] = [X_pred.iloc[i]['hour']]
            current_features['dayofweek'] = [X_pred.iloc[i]['dayofweek']]
            
            next_price = self.model.predict(pd.DataFrame(current_features))[0]
            price_forecast.append(next_price)
            
            # Update lag features for the next iteration
            for lag in sorted([24, 4, 1], reverse=True):
                lag_key = f'{self.target_col}_lag_{lag}'
                prev_lag_key = f'{self.target_col}_lag_{lag-1}' if lag > 1 else None
                
                if prev_lag_key in last_values:
                    last_values[lag_key] = last_values[prev_lag_key]
                elif lag_key in last_values:
                    last_values[lag_key] = next_price
            
            last_values[f'{self.target_col}_lag_1'] = next_price
        
        return pd.Series(price_forecast, index=X_pred.index, name='price_forecast_eur_kwh')