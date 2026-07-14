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

# =====================================================================
# 1. LOGGING & CONFIGURATION SETUP
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ingestion_pipeline")

DATABASE_URL: str = "postgresql://energy_user:energy_password@localhost:5432/energy_data"


# =====================================================================
# 2. VERIFIED DATABASE CONNECTION INJECTION (LAZY-LOADED SINGLETON)
# =====================================================================
_engine = None  # Internal placeholder to cache our verified engine

def get_engine():
    """
    Lazy-loads and returns a verified database engine singleton.
    This guarantees that database pings are deferred until actual write/read execution.
    """
    global _engine
    if _engine is not None:
        return _engine

    logger.info("Initializing database connection engine...")
    # Phase A: Validate connection string formatting and dialect drivers
    try:
        engine = create_engine(DATABASE_URL)
    except ArgumentError as err:
        logger.error(f"DATABASE_URL Syntax Error! The connection format is invalid. Details: {err}")
        sys.exit(1)
    except ImportError as err:
        logger.error(f"Missing Database Driver! Ensure you have psycopg2-binary installed. Details: {err}")
        sys.exit(1)

    # Phase B: Verify physical server and credential reachability
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Successfully handshaked with database. Engine verified!")
        _engine = engine
        return _engine
    except OperationalError as err:
        logger.error(
            "Database Unreachable! The connection string is syntactically correct, "
            f"but the connection physically timed out or was rejected.\nDetails: {err}"
        )
        sys.exit(1)


# =====================================================================
# 3. CORE INGESTION PIPELINE FUNCTIONS
# =====================================================================
def fetch_raw_prices(start: Union[str, datetime, date] = "now-P2D") -> List[Dict[str, Any]]:
    """Fetches raw price data from Energinet using precise filtering and sorting."""
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
    """Parses and upserts records into the database while avoiding duplicate violations."""
    if not records:
        logger.warning("No records were supplied for database insertion. Skipping database step.")
        return 0

    new_count = 0
    logger.info(f"Scanning {len(records)} potential records against existing database logs...")
    
    try:
        # Fetch the lazy-loaded engine here when database write action is called
        db_engine = get_engine()
        
        with Session(db_engine) as session:
            for idx, rec in enumerate(records):
                if not all(key in rec for key in ["TimeDK", "PriceArea", "DayAheadPriceDKK"]):
                    logger.warning(f"Record at index {idx} is malformed. Skipping: {rec}")
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
        logger.error(f"Database operation failed during bulk ingestion: {e}")
        raise


# =====================================================================
# 4. RUN ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    logger.info("Initializing energy pricing ingestion workflow pipeline...")
    try:
        # Explicitly invoke connection validation before starting fetch execution
        get_engine()
        raw_data = fetch_raw_prices()
        added = save_prices_to_db(raw_data)
        logger.info(f"Ingestion process completed successfully! Loaded {added} new intervals.")
    except Exception as e:
        logger.critical(f"Ingestion pipeline crashed unexpectedly: {e}")
        sys.exit(1)