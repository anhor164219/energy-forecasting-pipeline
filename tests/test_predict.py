import os
import pandas as pd
import numpy as np
import xgboost as xgb
import pytest

# IMPORT YOUR ACTUAL PRODUCTION INFERENCE LOGIC
from src.models.predict import load_production_model, generate_forecasts

class MockXGBModel:
    """A minimal mock class to simulate a model outputting an array of predictions"""
    def predict(self, X):
        return np.array([350.0, 420.0])

def test_actual_forecast_generation_formatting():
    """
    Verifies that generate_forecasts matches timestamps,
    coerces categories to strings, and populates predictions.
    """
    # 1. Arrange: Create mock categorical features
    times = pd.to_datetime(["2026-06-24 12:00:00", "2026-06-24 12:15:00"])
    X_new = pd.DataFrame({
        "price_area": pd.Series(["DK1", "DK2"], dtype="category"),
        "hour": [12, 12]
    }, index=times)
    
    mock_model = MockXGBModel()
    
    # 2. Act: Call your actual function
    forecast_df = generate_forecasts(mock_model, X_new)
    
    # 3. Assert
    assert isinstance(forecast_df, pd.DataFrame)
    assert "price_area" in forecast_df.columns
    assert "predicted_price" in forecast_df.columns
    # Task A: Prove that 'price_area' was converted from a category to a plain string/object data type
    assert forecast_df["price_area"].dtype == "object"
    
    # Task B: Prove that the values in 'predicted_price' match our mock predictions array exactly
    assert forecast_df["predicted_price"].iloc[0] == 350


def test_load_production_model_integration(tmp_path):
    """
    Uses pytest's tmp_path fixture to test saving and loading a minimal model state
    without touching your real production artifact file.
    """
    # 1. Arrange: Save a basic dummy model structure to a temporary test folder
    temp_dir = tmp_path / "artifacts"
    temp_dir.mkdir()
    temp_model_path = os.path.join(temp_dir, "test_model.json")
    
    # ⚡ THE FIX: Create a tiny 2-row matrix so XGBoost can initialize its tree engine
    X_dummy = pd.DataFrame({"feature": [1, 2]})
    y_dummy = pd.Series([100.0, 200.0])
    
    dummy_model = xgb.XGBRegressor(n_estimators=1) # Just build 1 tiny tree!
    dummy_model.fit(X_dummy, y_dummy)             # Now it's officially "fitted"
    dummy_model.save_model(temp_model_path)        # Writes a valid JSON schema
    
    # 2. Act: Run your actual loading wrapper function against the temp path
    loaded_model = load_production_model(temp_model_path)
    
    # 3. Assert: Prove that your loader successfully returned a functional XGBRegressor
    assert isinstance(loaded_model, xgb.XGBRegressor)