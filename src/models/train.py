import pandas as pd
import xgboost as xgb
from src.features.build_features import build_features_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import matplotlib.pyplot as plt
import os

def split_time_series_data(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronologically splits a DataFrame into training and testing sets.
    Ensures no future data leaks into the training phase.
    
    Tasks to implement:
    1. Calculate the integer split index based on the train_ratio (e.g., 80% mark).
    2. Slice the DataFrame up to the split index for the training set.
    3. Slice the DataFrame from the split index to the end for the testing set.
    
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
    
    Tasks to implement:
    1. Create a defensive copy.
    2. Drop the database 'id' column if it exists (it's random noise for an ML model).
    3. Convert the 'price_area' column to the pandas 'category' dtype.
    4. Separate the DataFrame into features X (everything except 'day_ahead_price')
       and target y ('day_ahead_price').
    """
    df = df.copy()
    
    # YOUR CODE HERE
    # 1. Drop 'id' column safely if it exists
    
    # 2. Convert 'price_area' to 'category'
    
    # 3. Separate X and y
    df = df.drop(columns=["id"], errors="ignore")
    df["price_area"] = df["price_area"].astype("category")
    X = df.drop(columns=["day_ahead_price"])
    y = df["day_ahead_price"]
    
    return X, y

def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    """
    Initializes and trains an XGBoost Regressor.
    """
    print("🚀 Training XGBoost model...")
    
    # Initialize the model to enable native categorical data processing!
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        tree_method="hist",       # Required or highly recommended for categorical support
        enable_categorical=True,  # Tells XGBoost to automatically read our pandas 'category' column
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("🎯 Model training complete!")
    return model

def tune_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    """
    Performs hyperparameter tuning using TimeSeriesSplit cross-validation
    to find the optimal combination of parameters for XGBoost.
    
    Tasks to implement:
    1. Initialize a base xgb.XGBRegressor with 'tree_method="hist"', 
       'enable_categorical=True', and a 'random_state=42'.
    2. Define a parameter dictionary 'param_grid' testing:
       - max_depth: 4, 6
       - learning_rate: 0.05, 0.1
       - n_estimators: 100, 150
    3. Instantiate a 'TimeSeriesSplit' engine with 3 chronological folds.
    4. Set up a 'GridSearchCV' engine using your base model, param_grid, and time-splitter.
       CRITICAL: Use scoring="neg_mean_absolute_error".
    5. Fit the grid search purely on the training data.
    6. Print out 'grid_search.best_params_' and return the 'grid_search.best_estimator_'.
    
    Returns:
        xgb.XGBRegressor: The fully optimized and retrained model artifact.
    """
    print("🔬 Initializing hyperparameter tuning loop...")
    
    # YOUR CODE HERE
    # 1. Base Model setup
    model = xgb.XGBRegressor(
        tree_method="hist",
        enable_categorical=True,
        random_state=42
    )
    # 2. Param Grid setup
    param_grid = { "max_depth": [4, 6], "learning_rate": [0.05, 0.1], "n_estimators": [100, 150]}
    # 3. Time Series Splitter setup
    Time_split = TimeSeriesSplit(n_splits=3)
    
    # 4. Grid Search setup (Hint: set n_jobs=-1 to use all CPU cores!)
    Grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_absolute_error", n_jobs=-1, cv=Time_split)
    # 5. Fit the Search
    Grid_search.fit(X_train, y_train)
    
    # 6. Extract results and return the best estimator
    print(f"👑 Best parameters found: {Grid_search.best_params_}")
    print(f"🏆 Best Cross-Validation MAE: {abs(Grid_search.best_score_):.2f} DKK / MWh")
    return Grid_search.best_estimator_


def evaluate_model(model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Generates predictions on the test set and calculates the Mean Absolute Error.
    
    Tasks to implement:
    1. Use model.predict() on X_test to get the array of predicted prices.
    2. Use mean_absolute_error() to compare the real prices (y_test) with your predictions.
    3. Print out the final MAE score.
    
    Returns:
        float: The calculated Mean Absolute Error score.
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"📊 Test Set Mean Absolute Error: {mae:.2f} DKK")
    return mae

def evaluate_model_global(model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Generates predictions on the test set, calculates the global MAE,
    and breaks down the error scores individually for DK1 and DK2.
    """
    # 1. Generate predictions
    y_pred = model.predict(X_test)
    
    # 2. Calculate global metric
    global_mae = mean_absolute_error(y_test, y_pred)
    print("=" * 45)
    print(f"🏆 GLOBAL TEST SET MAE: {global_mae:.2f} DKK / MWh")
    print("=" * 45)
    
    # 3. Create a temporary evaluation DataFrame to split regions
    eval_df = X_test.copy()
    eval_df["actual"] = y_test
    eval_df["predicted"] = y_pred
    
    # 4. Loop through each price area and score them independently
    for area in eval_df["price_area"].cat.categories:
        area_df = eval_df[eval_df["price_area"] == area]
        
        area_mae = mean_absolute_error(area_df["actual"], area_df["predicted"])
        print(f"📍 {area} Price Area MAE : {area_mae:.2f} DKK / MWh")
        
    print("=" * 45)
    return global_mae

def plot_predictions(X_test: pd.DataFrame, y_test: pd.Series, y_pred: list) -> None:
    """
    Generates a beautiful line chart comparing actual prices against 
    model predictions for a readable 3-day window.
    """
    # 1. Combine everything into a temporary DataFrame for easy filtering
    plot_df = X_test.copy()
    plot_df["actual"] = y_test
    plot_df["predicted"] = y_pred
    
    # 2. Filter down to just one region and a clear 3-day snapshot (4 intervals * 24h * 3 days)
    snapshot_df = plot_df[plot_df["price_area"] == "DK1"].tail(4 * 24 * 3)
    
    # 3. Build the plot
    plt.figure(figsize=(14, 6))
    
    plt.plot(snapshot_df.index, snapshot_df["actual"], label="Actual Price", color="#1f77b4", linewidth=2)
    plt.plot(snapshot_df.index, snapshot_df["predicted"], label="Predicted Price", color="#ff7f0e", linestyle="--", linewidth=2)
    
    # 4. Styling and labels
    plt.title("⚡ DK1 Price Forecast Validation (3-Day Snapshot)", fontsize=14, fontweight="bold")
    plt.xlabel("Time (UTC)", fontsize=12)
    plt.ylabel("Price (DKK / MWh)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    
    # Rotates the timestamp labels on the bottom so they look clean
    plt.gcf().autofmt_xdate() 
    
    print("📊 Rendering chart window...")
    plt.show()

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
    print(f"💾 Production model successfully saved to: {model_path}")
    # RUN THE VISUALIZER!
    #plot_predictions(X_test, y_test, y_pred)