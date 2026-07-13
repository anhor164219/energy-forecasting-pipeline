import pandas as pd
import numpy as np
import pytest

# IMPORT THE ACTUAL LOGIC FROM YOUR COMPONENT PIPELINE
from src.features.build_features import add_lag_features, add_temporal_features, add_rolling_features, impute_missing_intervals, remove_duplicates, clean_time_series

def test_actual_lag_isolation_by_region():
    """
    Verifies that add_lag_features calculates a 1-hour lag (4 steps back)
    independently per price area using a valid historical runway.
    """
    # 1. Arrange: Build a 5-step historical runway (5 timestamps x 2 areas = 10 rows)
    times = pd.date_range(start="2026-06-24 00:00:00", periods=5, freq="15min")
    
    # Repeat the times for both zones to create interleaved rows
    interleaved_times = np.repeat(times, 2) 
    
    mock_df = pd.DataFrame({
        "price_area": ["DK1", "DK2"] * 5,
        # Distinct linear sequences so we know exactly what value belongs where
        # DK1 prices will be: 10, 11, 12, 13, 14
        # DK2 prices will be: 50, 51, 52, 53, 54
        "day_ahead_price": [10.0, 50.0, 11.0, 51.0, 12.0, 52.0, 13.0, 53.0, 14.0, 54.0]
    }, index=interleaved_times)
    mock_df.index.name = "time_dk"
    
    # 2. Act: Run your actual code
    processed_df = add_lag_features(mock_df)
    
    # Extract the very last row for each region (the 5th time step)
    # At 01:00:00, a 1-hour lag should point exactly to 00:00:00
    dk1_latest = processed_df[(processed_df["price_area"] == "DK1") & (processed_df.index == "2026-06-24 01:00:00")].iloc[0]
    dk2_latest = processed_df[(processed_df["price_area"] == "DK2") & (processed_df.index == "2026-06-24 01:00:00")].iloc[0]
    
    # 3. Assert
    # Math Verification: 
    # DK1 current price is 14.0. 4 steps back (1 hour ago) its price was 10.0.
    assert dk1_latest["price_lag_1h"] == 10.0
    
    # DK2 current price is 54.0. 4 steps back (1 hour ago) its price was 50.0.
    # If boundary isolation fails, this might accidentally pull a DK1 price (like 14.0)
    assert dk2_latest["price_lag_1h"] == 50.0

def test_actual_temporal_extraction():
    """
    Verifies that extract_temporal_features correctly parses the pandas DatetimeIndex
    to create numeric hour and flag representations.
    """
    # 1. Arrange: Create explicit dates (a Wednesday afternoon and a Saturday morning)
    times = pd.to_datetime(["2026-06-24 14:00:00", "2026-06-27 06:00:00"])
    mock_df = pd.DataFrame({"dummy": [0, 0]}, index=times)
    
    # 2. Act: Execute your actual repository function
    processed_df = add_temporal_features(mock_df)
    # 3. Assert
    # Task C: Prove that row 0 accurately evaluated the hour as 14
    assert processed_df["hour"].iloc[0] == 14
    
    # Task D: Prove that row 0 (Wednesday) is marked as a weekday (is_weekend == 0)
    assert processed_df["is_weekend"].iloc[0] == 0
    
    # Task E: Prove that row 1 (Saturday) is marked as a weekend (is_weekend == 1)
    assert processed_df["is_weekend"].iloc[1] == 1
    
    assert processed_df["month"].iloc[0] == 6
    assert processed_df["day_of_week"].iloc[0] == 2

def test_actual_rolling_features():
    """
    Verifies that add_rolling_features calculates rolling statistics
    correctly over time and maintains strict boundaries between price areas.
    """
    # 1. Arrange: Create a 3-step sequence for both DK1 and DK2 (6 rows total)
    times = pd.date_range(start="2026-06-24 00:00:00", periods=98, freq="15min")
    interleaved_times = np.repeat(times, 2)
    day_ahead_price = []
    for i in range(98*2):
        if i % 2 == 0:
            day_ahead_price.append(100*i)
        else:
            day_ahead_price.append(10*i)
    mock_df = pd.DataFrame({
        "price_area": ["DK1", "DK2"] * 98,
        # DK1 prices: 10.0, 20.0, 30.0  -> (Averages should accumulate: 10.0, 15.0, 20.0)
        # DK2 prices: 100.0, 200.0, 300.0 -> (Averages should accumulate: 100.0, 150.0, 200.0)
        "day_ahead_price": day_ahead_price 
    }, index=interleaved_times)
    mock_df.index.name = "time_dk"
    
    # Sort index to ensure perfect chronological interleaving for the test
    mock_df = mock_df.sort_index()

    # 2. Act: Call your actual feature transformation function
    processed_df = add_rolling_features(mock_df)
    
    # Extract the rows for the final timestamp (00:30:00) to check the fully accumulated window
    dk1_final = processed_df[(processed_df["price_area"] == "DK1") & (processed_df.index == "2026-06-25 00:00:00")].iloc[0]
    dk2_final = processed_df[(processed_df["price_area"] == "DK2") & (processed_df.index == "2026-06-25 00:00:00")].iloc[0]
    
    # 3. Assert: Verify the rolling average column
    # (Adjust the column string key name if your production code labels it differently, e.g., 'price_rolling_mean_24h')
    
# Fixed Task A: Slice elements 1 to 96 for DK1 (maps to raw indices 2 through 192)
    expected_dk1_mean = np.mean(day_ahead_price[2:193:2])
    assert dk1_final["price_rolling_mean_24h"] == expected_dk1_mean
    
    # Fixed Task B: Slice elements 1 to 96 for DK2 (maps to raw indices 3 through 193)
    expected_dk2_mean = np.mean(day_ahead_price[3:194:2])
    assert dk2_final["price_rolling_mean_24h"] == expected_dk2_mean


def test_actual_impute_missing_intervals():
    """
    Verifies that impute_missing_intervals identifies gaps in the 15-minute grid,
    re-inserts the missing rows, and fills values independently per price area.
    """
    # 1. Arrange: Build a dataset where '2026-06-24 00:15:00' is missing ONLY for DK1
    times_dk1 = pd.to_datetime(["2026-06-24 00:00:00", "2026-06-24 00:30:00"])  # Missing 00:15!
    times_dk2 = pd.to_datetime(["2026-06-24 00:00:00", "2026-06-24 00:15:00", "2026-06-24 00:30:00"])
    
    df_dk1 = pd.DataFrame({
        "price_area": ["DK1", "DK1"],
        "day_ahead_price": [10.0, 30.0]
    }, index=times_dk1)
    
    df_dk2 = pd.DataFrame({
        "price_area": ["DK2", "DK2", "DK2"],
        "day_ahead_price": [100.0, 115.0, 130.0]
    }, index=times_dk2)
    
    # Combine them into a single, incomplete interleaved dataframe
    mock_incomplete_df = pd.concat([df_dk1, df_dk2]).sort_index()
    mock_incomplete_df.index.name = "time_dk"
    
    # Verify our setup is correct: the raw input should only have 5 rows total
    assert len(mock_incomplete_df) == 5

    # 2. Act: Call your actual imputation code
    processed_df = impute_missing_intervals(mock_incomplete_df)
    
    # 3. Assert: Verify the grid was fixed and filled properly
    
    # Task A: Prove the dataframe grew to 6 rows (meaning the missing DK1 row was restored)
    assert len(processed_df) == 6
    
    # Extract the specific row that used to be missing
    dk1_restored_row = processed_df[
        (processed_df["price_area"] == "DK1") & 
        (processed_df.index == "2026-06-24 00:15:00")
    ].iloc[0]
    
    # Task B: Prove that the price is no longer NaN
    assert not pd.isna(dk1_restored_row["day_ahead_price"])
    
    # Task C: Verify the imputation value based on your repository's strategy.
    # If your code uses Forward Fill (ffill), the price at 00:15 should copy 00:00 (which is 10.0).
    # If boundary isolation fails, it might accidentally copy DK2's 00:15 price (115.0)!
    assert dk1_restored_row["day_ahead_price"] == 20.0

def test_actual_remove_duplicates():
    """
    Verifies that remove_duplicates detects row collisions on the same timestamp
    and price area, resolving them cleanly without affecting valid rows.
    """
    # 1. Arrange: Provide 'time_dk' as a standard COLUMN to match production logic
    mock_duplicate_df = pd.DataFrame({
        "time_dk": pd.to_datetime([
            "2026-06-24 00:00:00", 
            "2026-06-24 00:00:00",  # Collision row
            "2026-06-24 00:00:00"   # Valid row (different area)
        ]),
        "price_area": ["DK1", "DK1", "DK2"],
        "day_ahead_price": [10.0, 99.0, 100.0]  
    })
    
    # Confirm the raw setup contains 3 rows before processing
    assert len(mock_duplicate_df) == 3

    # 2. Act: Call your actual repository cleanup function
    processed_df = remove_duplicates(mock_duplicate_df)
    
    # 3. Assert: Verify the collision was resolved cleanly
    # The duplicate row should be dropped (total rows should drop from 3 to 2)
    assert len(processed_df) == 2
    
    # Isolate the remaining data records for evaluation
    dk1_remaining = processed_df[processed_df["price_area"] == "DK1"]
    dk2_remaining = processed_df[processed_df["price_area"] == "DK2"]
    
    # Verify that DK2 was completely untouched by the deduplication step
    assert len(dk2_remaining) == 1
    assert dk2_remaining["day_ahead_price"].iloc[0] == 100.0
    
    # Verify that the first entry (10.0) was preserved for DK1
    assert len(dk1_remaining) == 1
    assert dk1_remaining["day_ahead_price"].iloc[0] == 10.0



def test_clean_time_series_defaults_and_timezone_shift():
    """
    Verifies that clean_time_series shifts Danish Summer Time (CEST) 
    2 hours back to UTC and defaults to clipping between -500 and 5000.
    """
    # Arrange: Create local June text timestamps with extreme and normal values
    mock_df = pd.DataFrame({
        "time_dk": ["2026-06-24 02:00:00", "2026-06-24 03:00:00"],
        "price_area": ["DK1", "DK1"],
        "day_ahead_price": [-1000.0, 6000.0]  # Out of default bounds
    })

    # Act: Use default parameters
    processed_df = clean_time_series(mock_df)

    # Assert: Timezone translation check (02:00 CEST -> 00:00 UTC)
    assert processed_df.index[0].hour == 0

    # Assert: Default bounding threshold checks
    assert processed_df["day_ahead_price"].iloc[0] == -500.0
    assert processed_df["day_ahead_price"].iloc[1] == 5000.0

def test_clean_time_series_custom_bounds():
    """Verifies clean_time_series enforces user-supplied lower and upper clipping limits."""
    # Arrange
    mock_df = pd.DataFrame({
        "time_dk": ["2026-06-24 12:00:00"],
        "price_area": ["DK2"],
        "day_ahead_price": [500.0]
    })

    # Act: Enforce a strict custom upper bound of 200.0
    processed_df = clean_time_series(mock_df, lower=0.0, upper=200.0)

    # Assert: Confirm the value was clamped down to the custom upper threshold
    assert processed_df["day_ahead_price"].iloc[0] == 200.0

def test_add_lag_features_defaults():
    """Verifies default list initialization [1, 24] generates standard lag columns."""
    # Arrange: Needs a 97-row dataset per region to populate a 24h lag (96 steps)
    times = pd.date_range(start="2026-06-24 00:00:00", periods=97, freq="15min")
    mock_df = pd.DataFrame({
        "price_area": ["DK1"] * 97,
        "day_ahead_price": np.arange(97, dtype=float)
    }, index=times)

    # Act: Rely on the safe None-to-default [1, 24] mapper
    processed_df = add_lag_features(mock_df, lag_hours=None)

    # Assert: Verify both default columns were built
    assert "price_lag_1h" in processed_df.columns
    assert "price_lag_24h" in processed_df.columns
    
    # 24 hours ago (96 periods back) from element 96 (value 96.0) is element 0 (value 0.0)
    assert processed_df["price_lag_24h"].iloc[96] == 0.0


def test_add_lag_features_custom_and_isolation():
    """Verifies that custom short lookbacks isolate regional groupings without cross-leakage."""
    # Arrange: Interleaved 2-step sequence
    times = pd.to_datetime([
        "2026-06-24 00:00:00", "2026-06-24 00:00:00",
        "2026-06-24 00:15:00", "2026-06-24 00:15:00"
    ])
    mock_df = pd.DataFrame({
        "price_area": ["DK1", "DK2", "DK1", "DK2"],
        "day_ahead_price": [10.0, 50.0, 15.0, 55.0]
    }, index=times)

    # Act: Request a custom 15-minute window (0.25 hours = 1 period shift)
    processed_df = add_lag_features(mock_df, lag_hours=[0.25])

    # Isolate step 2 rows (00:15:00) where the 1-period lag is populated
    dk1_step2 = processed_df[(processed_df["price_area"] == "DK1") & (processed_df.index == "2026-06-24 00:15:00")].iloc[0]
    dk2_step2 = processed_df[(processed_df["price_area"] == "DK2") & (processed_df.index == "2026-06-24 00:15:00")].iloc[0]

    # Assert: Verify column generation and regional firewalling
    assert "price_lag_0.25h" in processed_df.columns
    assert dk1_step2["price_lag_0.25h"] == 10.0
    assert dk2_step2["price_lag_0.25h"] == 50.0    


def test_add_rolling_features_defaults():
    """Verifies default list initialization [24] generates standard rolling mean column."""
    # Arrange: Create short mock sequence
    times = pd.date_range(start="2026-06-24 00:00:00", periods=3, freq="15min")
    mock_df = pd.DataFrame({
        "price_area": ["DK1"] * 3,
        "day_ahead_price": [10.0, 20.0, 30.0]
    }, index=times)

    # Act: Rely on the default [24] fallback window
    processed_df = add_rolling_features(mock_df, rolling_hours=None)

    # Assert
    assert "price_rolling_mean_24h" in processed_df.columns
    # min_periods=1 ensures that even short sequences yield rolling lookups immediately
    assert processed_df["price_rolling_mean_24h"].iloc[2] == 20.0  # (10 + 20 + 30) / 3


def test_add_rolling_features_custom_window():
    """Verifies custom rolling windows map correct aggregated lookups independently per region."""
    # Arrange: 3 steps, sorted chronologically
    times = pd.date_range(start="2026-06-24 00:00:00", periods=3, freq="15min")
    interleaved_times = np.repeat(times, 2)
    mock_df = pd.DataFrame({
        "price_area": ["DK1", "DK2"] * 3,
        "day_ahead_price": [10.0, 100.0, 20.0, 200.0, 30.0, 300.0] 
    }, index=interleaved_times).sort_index()

    # Act: Execute a 30-minute rolling window (0.5 hours = 2 periods)
    processed_df = add_rolling_features(mock_df, rolling_hours=[0.5])

    # Extract step 3 observations (00:30:00)
    dk1_final = processed_df[(processed_df["price_area"] == "DK1") & (processed_df.index == "2026-06-24 00:30:00")].iloc[0]
    dk2_final = processed_df[(processed_df["price_area"] == "DK2") & (processed_df.index == "2026-06-24 00:30:00")].iloc[0]

    # Assert: 2-period rolling window checks
    # DK1 at 00:30 evaluates 00:15 (20.0) and 00:30 (30.0) -> Mean = 25.0
    assert dk1_final["price_rolling_mean_0.5h"] == 25.0
    # DK2 at 00:30 evaluates 00:15 (200.0) and 00:30 (300.0) -> Mean = 250.0
    assert dk2_final["price_rolling_mean_0.5h"] == 250.0