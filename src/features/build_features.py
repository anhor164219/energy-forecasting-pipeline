import pandas as pd
from sqlmodel import create_engine
from typing import List, Union, Optional
# Connection string to your local Docker Postgres
DATABASE_URL: str = "postgresql://energy_user:energy_password@localhost:5432/energy_data"
engine = create_engine(DATABASE_URL)
def load_data_from_db() -> pd.DataFrame:
    """
    Loads all records from the energy_prices table into a Pandas DataFrame.
    
    Returns:
        pd.DataFrame: Raw data containing columns: id, time_dk, price_area, day_ahead_price
    """
    
    records = pd.read_sql_table(table_name="energy_prices", con=engine)
    return records


def clean_time_series(df: pd.DataFrame, lower: float =-500.0, upper: float =5000.0) -> pd.DataFrame:
    """
    Cleans the energy DataFrame for time-series modeling, 
    properly shifting Danish local time to UTC and clipping price outliers.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The raw input energy data containing 'time_dk' and 'day_ahead_price'.
    lower : float, default -500.0
        The lower bound threshold for clamping price values.
    upper : float, default 5000.0
        The upper bound threshold for clamping price values.
    """
    df = df.copy()
    
    df["time_dk"] = pd.to_datetime(df["time_dk"])
    df["time_dk"] = df["time_dk"].dt.tz_localize("Europe/Copenhagen").dt.tz_convert("UTC").dt.tz_localize(None)

    df["day_ahead_price"] = df["day_ahead_price"].clip(lower=lower, upper=upper)
    
    df = df.set_index("time_dk")
    df = df.sort_index(kind="stable")

    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies and removes any duplicate rows for the same timestamp and price area.
    Args:
        df (pd.DataFrame): DataFrame with possible duplicate values in timestamp and price area.
    Returns:
        pd.DataFrame: Cleaned DataFrame with guaranteed unique records.
    """
    df = df.copy()
    df = df.drop_duplicates(subset=["time_dk", "price_area"], keep="first")
    return df

def impute_missing_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects missing 15-minute intervals per price area, inserts empty rows, 
    and estimates the missing prices using linear interpolation.
    Args:
        df (pd.DataFrame): Cleaned DataFrame with a single DatetimeIndex.
        
    Returns:
        pd.DataFrame: DataFrame with a complete time grid and no missing intervals.
    """
    df = df.copy()
    df_resampled = df.groupby("price_area")["day_ahead_price"].resample("15min").asfreq()
    df_resampled = df_resampled.interpolate(method="linear")
    df = df_resampled.reset_index(level="price_area")
    df = df.sort_index(kind="stable")
    return df

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time-based features from the DataFrame's DatetimeIndex.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame with a DatetimeIndex.
        
    Returns:
        pd.DataFrame: DataFrame with the new temporal feature columns added.
    """
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype("int32")
    return df



def add_lag_features(
    df: pd.DataFrame, 
    lag_hours: Optional[List[Union[int, float]]] = None
) -> pd.DataFrame:
    df = df.copy()
    
    if lag_hours is None:
        lag_hours = [1, 24]
        
    for h in lag_hours:
        periods = int(h * 4)
        column_name = f"price_lag_{h}h"
        df[column_name] = df.groupby("price_area")["day_ahead_price"].shift(periods=periods)
        
    return df



def add_rolling_features(
    df: pd.DataFrame, 
    rolling_hours: Optional[List[Union[int, float]]] = None
) -> pd.DataFrame:
    """
    Creates rolling window mean features safely across different price areas.
    
    Because the data resolution is 15-minute intervals:
    - 24 hours = rolling window of 96 intervals (24 * 4)
    
    Args:
        df (pd.DataFrame): DataFrame with 'price_area' and 'day_ahead_price'.
        rolling_hours (List[int/float], optional): Explicit list of hour windows to 
                                                   compute rolling means for.
                                                   Defaults to [4, 12, 24].
        
    Returns:
        pd.DataFrame: DataFrame with dynamically generated rolling mean columns.
    """
    df = df.copy()
    
    if rolling_hours is None:
        rolling_hours = [24]
        
    for h in rolling_hours:
        periods = int(h * 4)
        column_name = f"price_rolling_mean_{h}h"
        df[column_name] = (
            df.groupby("price_area")["day_ahead_price"]
            .transform(lambda group: group.rolling(window=periods, min_periods=1).mean())
        )
        
    return df

def build_features_pipeline(lag_hours: Optional[List[Union[int, float]]] = None, rolling_hours: Optional[List[Union[int, float]]] = None) -> pd.DataFrame:
    """
    Orchestrates the entire feature engineering pipeline from database to final ML-ready data.
    Returns:
        pd.DataFrame: A completely clean, feature-rich DataFrame ready for training.
    """

    df = load_data_from_db()
    df = remove_duplicates(df)
    df = clean_time_series(df)
    df = impute_missing_intervals(df)
    df = add_temporal_features(df)
    df = add_lag_features(df, lag_hours)
    df = add_rolling_features(df, rolling_hours)
    df = df.dropna()
    return df


if __name__ == "__main__":
    # Test implementation
    raw_df = load_data_from_db()
    print("Raw Data Shape:", raw_df.shape)
    
    cleaned_df = clean_time_series(raw_df)
    print("Cleaned Data Index Type:", type(cleaned_df.index))
    print(cleaned_df.head())

    temporal_feature_df = add_temporal_features(cleaned_df)
    print(" Added temporal features:")
    print(temporal_feature_df.head(n=200))

    temporal_lag_feature_df = add_lag_features(temporal_feature_df)
    print(" Added temporal features:")
    print(temporal_lag_feature_df.iloc[[0, 1, 2, 3, 4, 5]])
    print(temporal_lag_feature_df.iloc[[8, 9, 10, 11, 12, 13]])
    print(temporal_lag_feature_df.iloc[[192, 193, 194, 195, 196, 197]])

    final_df = build_features_pipeline()
    
    # This counts how many NaNs are left in each column
    print("Missing values per column:")
    print(final_df.isna().sum())
    print(final_df.head())

