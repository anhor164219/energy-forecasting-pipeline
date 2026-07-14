import pandas as pd
import xgboost as xgb
from src.features.build_features import build_features_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import os

def split_time_series_data(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronologically splits a DataFrame into training and testing sets.
    Ensures no future data leaks into the training phase.
    
    Args:
        df (pd.DataFrame): The fully processed, feature-rich DataFrame.
        train_ratio (float): The proportion of data to use for training (default 0.8).
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    split_index = int(train_ratio * len(df))
    if split_index % 2 == 1:
        split_index += 1
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    return train_df, test_df


def prepare_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepares the data matrix for XGBoost by handling data types 
    and splitting features from targets.
    """
    df = df.copy()
    df = df.drop(columns=["id"], errors="ignore")
    df["price_area"] = df["price_area"].astype("category")
    X = df.drop(columns=["day_ahead_price"])
    y = df["day_ahead_price"]
    
    return X, y

def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    """
    Initializes and trains an XGBoost Regressor.
    """
    print(" Training XGBoost model...")
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        tree_method="hist",      
        enable_categorical=True,  
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print(" Model training complete!")
    return model

def tune_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    """
    Performs hyperparameter tuning using TimeSeriesSplit cross-validation
    to find the optimal combination of parameters for XGBoost.
    Returns:
        xgb.XGBRegressor: The fully optimized and retrained model artifact.
    """
    print(" Initializing hyperparameter tuning loop...")
    
    model = xgb.XGBRegressor(
        tree_method="hist",
        enable_categorical=True,
        random_state=42
    )
    param_grid = { "max_depth": [4, 6], "learning_rate": [0.05, 0.1], "n_estimators": [100, 150]}
    Time_split = TimeSeriesSplit(n_splits=3)
    Grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_absolute_error", n_jobs=-1, cv=Time_split)
    Grid_search.fit(X_train, y_train)
    print(f" Best parameters found: {Grid_search.best_params_}")
    print(f" Best Cross-Validation MAE: {abs(Grid_search.best_score_):.2f} DKK / MWh")
    return Grid_search.best_estimator_


def evaluate_model(model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Generates predictions on the test set and calculates the Mean Absolute Error.
    
    Returns:
        float: The calculated Mean Absolute Error score.
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f" Test Set Mean Absolute Error: {mae:.2f} DKK")
    return mae

def evaluate_model_global(model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Generates predictions on the test set, calculates the global MAE,
    and breaks down the error scores individually for DK1 and DK2.
    """
    y_pred = model.predict(X_test)
    global_mae = mean_absolute_error(y_test, y_pred)
    print("=" * 45)
    print(f" GLOBAL TEST SET MAE: {global_mae:.2f} DKK / MWh")
    print("=" * 45)
    eval_df = X_test.copy()
    eval_df["actual"] = y_test
    eval_df["predicted"] = y_pred
    
    for area in eval_df["price_area"].cat.categories:
        area_df = eval_df[eval_df["price_area"] == area]
        
        area_mae = mean_absolute_error(area_df["actual"], area_df["predicted"])
        print(f" {area} Price Area MAE : {area_mae:.2f} DKK / MWh")
        
    print("=" * 45)
    return global_mae


if __name__ == "__main__":
    full_df = build_features_pipeline()
    train_data, test_data = split_time_series_data(full_df, train_ratio=0.8)
    
    X_train, y_train = prepare_matrices(train_data)
    X_test, y_test = prepare_matrices(test_data)
    
    #  CHOSEN CHALLENGE: Run the tuning loop!
    model = tune_xgboost_model(X_train, y_train)
    
    # Evaluate the optimized model on the completely hidden test set
    y_pred = model.predict(X_test)
    evaluate_model_global(model, X_test, y_test)
    #evaluate_model(model, X_test, y_test)
    
    # Define the path where you want to keep your production artifacts
    model_dir = "src/models/artifacts"
    os.makedirs(model_dir, exist_ok=True)  # Creates the folder if it doesn't exist yet
    model_path = os.path.join(model_dir, "xgboost_energy_model.json")

    # Save the model natively
    model.save_model(model_path)
    print(f" Production model successfully saved to: {model_path}")
