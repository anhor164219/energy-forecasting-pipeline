"""Fetch 15-minute energy prices from Energinet and save them to PostgreSQL."""

import json
import requests
from datetime import datetime, date
from typing import List, Dict, Any, Union
from sqlmodel import Session, create_engine, select

# Internal imports
from src.models.models import EnergyPrice

DATABASE_URL: str = "postgresql://energy_user:energy_password@localhost:5432/energy_data"
engine = create_engine(DATABASE_URL)

def fetch_raw_prices(start: Union[str, datetime, date] = "now-P2D") -> List[Dict[str, Any]]:
    """
    Fetches raw price data from Energinet using precise filtering and sorting.
    
    Args:
        start (Union[str, datetime, date]): The starting boundary for data collection.
            - Can be a relative string offset (e.g., "now-P2D" for 2 days, "now-P1M" for 1 month).
            - Can be an absolute text string (e.g., "2026-07-11").
            - Can be a native Python datetime/date object.
            Defaults to "now-P2D".
            
    Returns:
        List[Dict[str, Any]]: A list of raw record dictionaries returned from the API.
    """
    url = "https://api.energidataservice.dk/dataset/DayAheadPrices"
    
    # Polymorphic type check: convert native date objects to valid ISO strings automatically
    if isinstance(start, (datetime, date)):
        start_str = start.strftime("%Y-%m-%dT%H:%M")
    else:
        start_str = str(start)
        
    query_params = {
        "start": start_str,
        "filter": json.dumps({"PriceArea": ["DK1", "DK2"]}), 
        "sort": "TimeDK ASC",
        "columns": "TimeDK,PriceArea,DayAheadPriceDKK"  # Minimize network bandwidth
    }
    
    print(f"Requesting data from Energinet (start filter: {start_str})...")
    response = requests.get(url, params=query_params)
    response.raise_for_status()
    
    data = response.json()
    return data.get("records", [])

def save_prices_to_db(records: List[Dict[str, Any]]) -> int:
    """Parses and upserts records into the database."""
    new_count = 0
    with Session(engine) as session:
        for rec in records:
            # We check existence using the new 'time_dk' name
            statement = select(EnergyPrice).where(
                EnergyPrice.time_dk == rec["TimeDK"],
                EnergyPrice.price_area == rec["PriceArea"]
            )
            existing = session.exec(statement).first()
            
            if not existing:
                new_price = EnergyPrice(
                    time_dk=rec["TimeDK"],
                    price_area=rec["PriceArea"],
                    day_ahead_price=rec["DayAheadPriceDKK"]
                )
                session.add(new_price)
                new_count += 1
        
        session.commit()
    return new_count

if __name__ == "__main__":
    try:
        raw_data = fetch_raw_prices("2026-03-01")
        added = save_prices_to_db(raw_data)
        print(f" Ingestion complete! Added {added} new 15-min intervals to the DB.")
    except Exception as e:
        print(f" Ingestion failed: {e}")