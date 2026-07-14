"""Fetch energy prices from Energinet and save them to PostgreSQL."""

import json
import sys
import logging
import requests
from datetime import datetime, date
from typing import List, Dict, Any, Union

from sqlalchemy.exc import ArgumentError, OperationalError
from sqlmodel import Session, create_engine, select, text
from src.models.models import EnergyPrice

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ingestion_pipeline")

DATABASE_URL: str = "postgresql://energy_user:energy_password@localhost:5432/energy_data"


def get_verified_engine(database_url: str):
    """
    Safely instantiates a SQLModel engine and verifies database connectivity
    before allowing downstream processing to start.
    """
    try:
        engine = create_engine(database_url)
    except ArgumentError as err:
        logger.error(f"DATABASE_URL Syntax Error! The connection string format is invalid. Details: {err}")
        sys.exit(1)
    except ImportError as err:
        logger.error(f"Missing Database Driver! Ensure you have the required DB driver installed. Details: {err}")
        sys.exit(1)

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Successfully query from database. Engine verified!")
        return engine
    except OperationalError as err:
        logger.error(
            "Database Unreachable! The connection string is syntactically correct, "
            f"but the connection physically timed out or was rejected.\nDetails: {err}"
        )
        sys.exit(1)


engine = get_verified_engine(DATABASE_URL)


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
    
    if isinstance(start, (datetime, date)):
        start_str = start.strftime("%Y-%m-%dT%H:%M")
    else:
        start_str = str(start)
        
    query_params = {
        "start": start_str,
        "filter": json.dumps({"PriceArea": ["DK1", "DK2"]}), 
        "sort": "TimeDK ASC",
        "columns": "TimeDK,PriceArea,DayAheadPriceDKK"  
    }
    
    logger.info(f"Contacting Energinet API... Requesting start window: {start_str}")
    
    try:
        response = requests.get(url, params=query_params)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred while calling Energinet API: {e}")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"A network/request error occurred while contacting Energinet: {e}")
        return []
    
    try:
        data = response.json()
        records = data.get("records", [])
        logger.info(f"API Response retrieved successfully. Pulled {len(records)} candidate raw entries.")
        return records
    except ValueError as e:
        logger.error(f"Failed to parse API response as JSON payload: {e}")
        return []


def save_prices_to_db(records: List[Dict[str, Any]]) -> int:
    """
    Parses and upserts records into the database while avoiding duplicate violations.
    
    Returns:
        int: The total count of newly inserted records.
    """
    if not records:
        logger.warning("No records were supplied for database insertion. Skipping database step.")
        return 0

    new_count = 0
    logger.info(f"Scanning {len(records)} potential records against existing database logs...")
    
    try:
        with Session(engine) as session:
            for idx, rec in enumerate(records):
                if not all(key in rec for key in ["TimeDK", "PriceArea", "DayAheadPriceDKK"]):
                    logger.warning(f"Record at index {idx} is malformed and missing key fields. Skipping: {rec}")
                    continue

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
            
            if new_count > 0:
                logger.info(f"Committing {new_count} new unique records to database...")
                session.commit()
                logger.info("Database transaction committed successfully.")
            else:
                logger.info("Zero new records found. No commit transaction required.")
                
        return new_count

    except Exception as e:
        logger.error(f"Database operation failed during ingestion: {e}")
        raise


# =====================================================================
# 4. RUN ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    logger.info("Initializing energy pricing ingestion workflow pipeline...")
    try:
        raw_data = fetch_raw_prices()
        added = save_prices_to_db(raw_data)
        logger.info(f"Ingestion process completed successfully! Loaded {added} new intervals into the database.")
    except Exception as e:
        logger.critical(f"Ingestion pipeline crashed unexpectedly during execution runtime: {e}")
        sys.exit(1)