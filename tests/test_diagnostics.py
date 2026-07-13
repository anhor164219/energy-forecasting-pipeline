import pandas as pd
import numpy as np
import pytest

# IMPORT YOUR ACTUAL DIAGNOSTIC LOGIC
from src.models.diagnostics import extract_feature_importance

class DiagnosticMockModel:
    def __init__(self):
        # Mock raw feature importance rankings (sum to 1.0)
        # index 0 matches hour, index 1 matches price_lag_1h, index 2 matches price_area
        self.feature_importances_ = np.array([0.15, 0.70, 0.15])

def test_extract_feature_importance_sorting():
    """
    Verifies that extract_feature_importance correctly pairs columns 
    with their weights and sorts them in descending order.
    """
    # 1. Arrange
    feature_names = ["hour", "price_lag_1h", "price_area"]
    mock_model = DiagnosticMockModel()
    
    # 2. Act: Call your actual diagnostic sorting function
    sorted_importances = extract_feature_importance(mock_model, feature_names)
    assert isinstance(sorted_importances, pd.Series)
    # 3. Assert
    # Task D: Prove that the champion feature ('price_lag_1h') was sorted to the very top slot (index 0)
    assert sorted_importances.index[0] == "price_lag_1h"
    
    # Task E: Prove that the top weight evaluates to exactly 0.70
    assert sorted_importances.iloc[0] == 0.7