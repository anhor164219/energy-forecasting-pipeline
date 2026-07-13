import os
import pandas as pd
import xgboost as xgb
from src.features.build_features import build_features_pipeline
from src.models.train import prepare_matrices

def load_production_model(model_path: str) -> xgb.XGBRegressor:
    """
    Loads the persistent JSON model artifact from disk into memory.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f" No model found at {model_path}. Run train.py first!")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    print(" Production model loaded successfully from disk.")
    return model

def generate_forecasts(model: xgb.XGBRegressor, X_new: pd.DataFrame) -> pd.DataFrame:
    """
    Uses the loaded model to predict prices and formats the output for the database.
    """
    predictions = model.predict(X_new)
    
    forecast_df = pd.DataFrame(index=X_new.index)
    forecast_df["price_area"] = X_new["price_area"].astype("str")
    forecast_df["predicted_price"] = predictions
    
    return forecast_df

if __name__ == "__main__":
    print(" Starting Live Production Inference Loop...")
    
    # 1. Path to your saved asset
    MODEL_PATH = "src/models/artifacts/xgboost_energy_model.json"
    
    # 2. Load the trained model artifact using your new function
    model = load_production_model(MODEL_PATH)
    
    # 3. Simulate getting fresh live data by pulling the latest feature pipeline matrices
    # (In a real system, this would look at a small 'latest data' slice from June 2026)
    print(" Gathering latest grid features...")
    full_df = build_features_pipeline()
    X_live, _ = prepare_matrices(full_df)
    
    # Take just the final 10 records as a mock 'live delivery window'
    X_live_snapshot = X_live.tail(10)
    
    final_forecasts = generate_forecasts(model, X_live_snapshot)
    
    print("\n LIVE FORECAST TABLE READY FOR DATABASE INSERTION:")
    print("=" * 60)
    print(final_forecasts)
    print("=" * 60)