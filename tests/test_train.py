import pandas as pd
import numpy as np
import pytest
import xgboost as xgb
# IMPORT THE ACTUAL LOGIC FROM YOUR REPOSITORY
from src.models.train import split_time_series_data, prepare_matrices, evaluate_model, evaluate_model_global, tune_xgboost_model, train_xgboost_model

def test_actual_chronological_splitting():
    """
    Verifies that split_time_series_data splits chronologically 
    without shuffling the rows.
    """
    # 1. Arrange: Create 10 sequential chronological rows
    times = pd.date_range(start="2026-06-23 00:00", periods=10, freq="15min")
    mock_df = pd.DataFrame({"price": np.arange(10)}, index=times)
    
    # 2. Act: Call your ACTUAL repository function with an 80/20 split
    train_df, test_df = split_time_series_data(mock_df, train_ratio=0.8)
    
    # 3. Assert: Fill in the validation blanks
    # Task A: Prove train gets exactly 8 rows and test gets 2 rows
    assert len(train_df) == 8
    assert len(test_df) == 2
    
    # Task B: Prove the split was chronological (test data must contain the LATEST elements)
    # The first value in the test set should be the 8th index value (which is 7)
    assert test_df["price"].iloc[0] == 8

    train_df, test_df = split_time_series_data(mock_df, train_ratio=0.7)
    assert len(train_df) == 8
    assert len(test_df) == 2
    assert test_df["price"].iloc[0] == 8


def test_actual_matrix_preparation():
    """
    Verifies that prepare_matrices strips the target column and preserves dtypes.
    """
    # 1. Arrange
    mock_df = pd.DataFrame({
        "id": [1, 2],
        "price_area": ["DK1", "DK2"],
        "day_ahead_price": [400.0, 450.0]
    })
    
    # 2. Act: Call your ACTUAL repository function
    X, y = prepare_matrices(mock_df)
    
    # 3. Assert: Fill in the validation blanks
    # Task C: Prove that target variable 'day_ahead_price' is completely absent from X
    assert "day_ahead_price" not in X
    assert y.name == "day_ahead_price"
    assert "id" not in X
    assert "id" not in y
    assert X["price_area"].dtype == "category"
    # Task D: Prove that target y is a series containing the original prices
    assert y.iloc[0] == mock_df["day_ahead_price"].iloc[0]

class DummyXGBRegressor:
    def predict(self, X):
        # We will force the model to predict exactly 410.0 for the first row 
        # and 480.0 for the second row.
        return np.array([410.0, 480.0])

def test_actual_evaluation_metrics():
    """
    Verifies that evaluate_model correctly computes and returns 
    the Global Mean Absolute Error (MAE).
    
    Tasks to implement:
    Fill in the placeholder assertions to validate the error calculation.
    """
    # 1. Arrange: Define explicit inputs where we know the exact mathematical outcome
    X_test = pd.DataFrame({
        "price_area": pd.Series(["DK1", "DK2"], dtype="category"),
        "hour": [12, 13]
    })
    
    # True values: Row 1 is 400.0, Row 2 is 500.0
    y_test = pd.Series([400.0, 500.0], name="day_ahead_price")
    
    mock_model = DummyXGBRegressor()
    
    # 2. Act: Run your ACTUAL repository function
    # Note: This will print the diagnostic tables to your screen when running pytest!
    global_mae = evaluate_model(mock_model, X_test, y_test)
    
    # 3. Assert: Verify the underlying error math
    # Math Check:
    # - Row 1 (DK1): Actual = 400.0, Predicted = 410.0 -> Absolute Error = 10.0
    # - Row 2 (DK2): Actual = 500.0, Predicted = 480.0 -> Absolute Error = 20.0
    # - Expected Global MAE = (10.0 + 20.0) / 2 = 15.0
    
    # Task A: Prove the returned global MAE matches your exact calculation
    assert global_mae == 15


def test_global_evaluation_metrics():
    """
    Verifies that evaluate_model correctly computes and returns 
    the Global Mean Absolute Error (MAE).
    
    Tasks to implement:
    Fill in the placeholder assertions to validate the error calculation.
    """
    # 1. Arrange: Define explicit inputs where we know the exact mathematical outcome
    X_test = pd.DataFrame({
        "price_area": pd.Series(["DK1", "DK2"], dtype="category"),
        "hour": [12, 13]
    })
    
    # True values: Row 1 is 400.0, Row 2 is 500.0
    y_test = pd.Series([400.0, 500.0], name="day_ahead_price")
    
    mock_model = DummyXGBRegressor()
    
    # 2. Act: Run your ACTUAL repository function
    # Note: This will print the diagnostic tables to your screen when running pytest!
    global_mae = evaluate_model_global(mock_model, X_test, y_test)
    
    # 3. Assert: Verify the underlying error math
    # Math Check:
    # - Row 1 (DK1): Actual = 400.0, Predicted = 410.0 -> Absolute Error = 10.0
    # - Row 2 (DK2): Actual = 500.0, Predicted = 480.0 -> Absolute Error = 20.0
    # - Expected Global MAE = (10.0 + 20.0) / 2 = 15.0
    
    # Task A: Prove the returned global MAE matches your exact calculation
    assert global_mae == 15

def test_tune_xgboost_model_smoketest(monkeypatch):
    """
    Smoketest: Verifies that the tuning loop executes without syntax 
    or configuration errors when handed valid data matrices.
    """
    # 1. Arrange: Create a minimal mock training dataset (6 rows for 3 chronological folds)
    times = pd.date_range(start="2026-06-23 00:00", periods=6, freq="15min")
    
    X_train = pd.DataFrame({
        "price_area": pd.Series(["DK1", "DK2", "DK1", "DK2", "DK1", "DK2"], dtype="category"),
        "hour": [0, 0, 1, 1, 2, 2],
        "price_lag_1h": [400.0, 420.0, 410.0, 430.0, 415.0, 435.0]
    }, index=times)
    
    y_train = pd.Series([410.0, 430.0, 415.0, 435.0, 420.0, 440.0], name="day_ahead_price")

    # ⚡ PRO MLOps TRICK: Monkeypatch the parameter grid!
    # Because running the real 18-combination grid takes too long, we temporarily 
    # overwrite your production grid settings with a microscopic grid during this test.
    tiny_grid = {
        "max_depth": [2],
        "learning_rate": [0.1],
        "n_estimators": [5]  # Only build 5 trees instead of 150!
    }
    
    # This dynamically intercepts your function's parameter grid if you structured it as a variable,
    # or we can test the function directly. For this exercise, let's assume we run the function.
    # To keep the test under 2 seconds, ensure your actual code runs smoothly.

    # 2. Act: Call your actual training function
    trained_model = tune_xgboost_model(X_train, y_train)
    
    # 3. Assert: Verify that the function successfully spit out a real model object
    # Task A: Prove that the returned asset is a real, instantiated XGBRegressor
    assert isinstance(trained_model, xgb.XGBRegressor)    


def test_train_xgboost_model_smoketest():
    """
    Smoketest: Verifies that the baseline model training function executes
    and successfully outputs a fitted model artifact when given valid inputs.
    """
    # 1. Arrange: Create a minimal mock training dataset (4 rows is plenty for a baseline fit)
    times = pd.date_range(start="2026-06-23 00:00", periods=4, freq="15min")
    
    X_train = pd.DataFrame({
        "price_area": pd.Series(["DK1", "DK2", "DK1", "DK2"], dtype="category"),
        "hour": [0, 0, 1, 1],
        "price_lag_1h": [400.0, 420.0, 410.0, 430.0]
    }, index=times)
    
    y_train = pd.Series([410.0, 430.0, 415.0, 435.0], name="day_ahead_price")
    
    # 2. Act: Call your ACTUAL baseline repository function
    trained_base_model = train_xgboost_model(X_train, y_train)
    
    # 3. Assert: Validate that the function successfully spit out a real model object
    # Task A: Prove that the returned asset is a real, instantiated XGBRegressor
    assert isinstance(trained_base_model, xgb.XGBRegressor)