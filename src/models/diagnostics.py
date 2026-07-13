import pandas as pd
import xgboost as xgb
from src.features.build_features import build_features_pipeline
from src.models.train import prepare_matrices
from src.models.predict import load_production_model

def extract_feature_importance(model: xgb.XGBRegressor, feature_names: list) -> pd.Series:
    """
    Extracts the feature importances from a trained XGBoost model,
    pairs them with their true column names, and sorts them.
    """
    raw_importances = model.feature_importances_
    
    importance_series = pd.Series(raw_importances, index=feature_names)
    sorted_series = importance_series.sort_values(ascending=False)
    
    return sorted_series

if __name__ == "__main__":
    print(" Running Model Diagnostics...")
    
    MODEL_PATH = "src/models/artifacts/xgboost_energy_model.json"
    model = load_production_model(MODEL_PATH)
    
    full_df = build_features_pipeline()
    X, _ = prepare_matrices(full_df)
    feature_columns = X.columns.tolist()
    
    # Run your diagnostic challenge!
    ranked_features = extract_feature_importance(model, feature_columns)
    
    print("\n TOP INFLUENTIAL FEATURES IN YOUR XGBOOST MODEL:")
    print("=" * 50)
    for rank, (feature, weight) in enumerate(ranked_features.items(), 1):
        print(f"{rank:2d}. {feature:<25} -> {weight*100:>6.2f}% impact")
    print("=" * 50)