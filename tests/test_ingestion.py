import sys
import pytest
from datetime import datetime, date
from unittest.mock import patch, MagicMock

# =====================================================================
# 🛡️ THE DATABASE SHIELD (PRE-IMPORT PATCHING)
# Because 'src/ingestion/ingest.py' runs a database verification ping 
# on import, we MUST mock `create_engine` and `text` BEFORE importing 
# any functions. This prevents unit tests from crashing on offline DBs.
# =====================================================================
with patch("src.ingestion.ingest.create_engine") as mock_create, \
     patch("src.ingestion.ingest.text") as mock_text:
    
    # Configure our mocked DB engine to act like a valid active database
    mock_engine = MagicMock()
    mock_connect = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_connect
    mock_create.return_value = mock_engine
    
    # Safely import the target functions now that the import side-effects are bypassed
    from src.ingestion.ingest import fetch_raw_prices, save_prices_to_db


# =====================================================================
# 1. TESTS FOR: fetch_raw_prices (Ingestion API)
# =====================================================================

@patch("src.ingestion.ingest.requests.get")
def test_fetch_raw_prices_default_handling(mock_get):
    """Verifies the default duration string 'now-P2D' is correctly transmitted inside query parameters."""
    # Arrange: Build a dummy API JSON payload structure
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "records": [{"TimeDK": "2026-06-24 00:00", "PriceArea": "DK1", "DayAheadPriceDKK": 150.0}]
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    # Act: Call ingestion using default fallback parameters
    records = fetch_raw_prices()

    # Assert: Verify internal dict extraction
    assert len(records) == 1
    assert records[0]["PriceArea"] == "DK1"

    # Assert: Check that requests.get was invoked with the exact default payload mapping
    mock_get.assert_called_once()
    called_kwargs = mock_get.call_args[1]
    
    assert called_kwargs["params"]["start"] == "now-P2D"
    assert "DK1" in called_kwargs["params"]["filter"]


@patch("src.ingestion.ingest.requests.get")
def test_fetch_raw_prices_polymorphic_inputs(mock_get):
    """Verifies that python datetime and date objects are seamlessly formatted to string stamps."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"records": []}
    mock_get.return_value = mock_response

    # Test Case A: Inputting a native Python date object
    target_date = date(2026, 7, 11)
    fetch_raw_prices(start=target_date)
    
    # Extract the string query param passed to requests during the first execution
    first_call_params = mock_get.call_args_list[0][1]["params"]
    assert first_call_params["start"] == "2026-07-11T00:00"

    # Test Case B: Inputting a strict hardcoded text duration string
    fetch_raw_prices(start="now-P1M")
    second_call_params = mock_get.call_args_list[1][1]["params"]
    assert second_call_params["start"] == "now-P1M"


@patch("src.ingestion.ingest.requests.get")
def test_fetch_raw_prices_network_failures(mock_get):
    """Verifies that requests.exceptions safely return an empty list instead of uncaught crashes."""
    from requests.exceptions import RequestException
    
    # Force the API request to raise a Network Timeout / RequestException
    mock_get.side_effect = RequestException("Connection Timed Out")
    
    # Act: Call the function
    records = fetch_raw_prices()
    
    # Assert: The function caught the error internally and returned an empty array cleanly
    assert records == []


# =====================================================================
# 2. TESTS FOR: save_prices_to_db (Database Upserts)
# =====================================================================

def test_save_prices_to_db_empty_records():
    """Verifies that passing no records exits early and returns 0 without calling the database."""
    # Act
    inserted_count = save_prices_to_db([])
    
    # Assert
    assert inserted_count == 0


@patch("src.ingestion.ingest.Session")
def test_save_prices_to_db_inserts_new_records(mock_session_class):
    """Verifies that completely new records are successfully staged and committed to the database."""
    # Arrange: Mock the Session context manager flow
    mock_session_instance = MagicMock()
    mock_session_class.return_value.__enter__.return_value = mock_session_instance
    
    # Return None for search queries (indicating the record is new/does not exist)
    mock_result = MagicMock()
    mock_result.first.return_value = None
    mock_session_instance.exec.return_value = mock_result
    
    # Create input data records to insert
    mock_records = [
        {"TimeDK": "2026-06-24 00:00", "PriceArea": "DK1", "DayAheadPriceDKK": 150.0},
        {"TimeDK": "2026-06-24 00:15", "PriceArea": "DK1", "DayAheadPriceDKK": 160.0}
    ]
    
    # Act
    inserted_count = save_prices_to_db(mock_records)
    
    # Assert: 2 new records should be successfully inserted
    assert inserted_count == 2
    assert mock_session_instance.add.call_count == 2
    mock_session_instance.commit.assert_called_once()


@patch("src.ingestion.ingest.Session")
def test_save_prices_to_db_skips_duplicates(mock_session_class):
    """Verifies that records already existing in the database are ignored and not committed."""
    # Arrange: Mock the Session context manager flow
    mock_session_instance = MagicMock()
    mock_session_class.return_value.__enter__.return_value = mock_session_instance
    
    # Simulate that the record already exists in the database
    from src.models.models import EnergyPrice
    existing_price_mock = MagicMock(spec=EnergyPrice)
    
    mock_result = MagicMock()
    mock_result.first.return_value = existing_price_mock
    mock_session_instance.exec.return_value = mock_result
    
    # Input a record that matches our database "mock"
    mock_records = [
        {"TimeDK": "2026-06-24 00:00", "PriceArea": "DK1", "DayAheadPriceDKK": 150.0}
    ]
    
    # Act
    inserted_count = save_prices_to_db(mock_records)
    
    # Assert: No database actions should be added or committed
    assert inserted_count == 0
    mock_session_instance.add.assert_not_called()
    mock_session_instance.commit.assert_not_called()