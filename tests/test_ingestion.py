import pytest
from datetime import datetime, date
from unittest.mock import patch, MagicMock

# Safe imports now that lazy-loading protects the import sequence!
from src.ingestion.ingest import fetch_raw_prices, save_prices_to_db


# =====================================================================
# 1. TESTS FOR: fetch_raw_prices (Ingestion API)
# =====================================================================

@patch("src.ingestion.ingest.requests.get")
def test_fetch_raw_prices_default_handling(mock_get):
    """Verifies the default duration string 'now-P2D' is correctly transmitted inside query parameters."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "records": [{"TimeDK": "2026-06-24 00:00", "PriceArea": "DK1", "DayAheadPriceDKK": 150.0}]
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    records = fetch_raw_prices()

    assert len(records) == 1
    assert records[0]["PriceArea"] == "DK1"

    mock_get.assert_called_once()
    called_kwargs = mock_get.call_args[1]
    assert called_kwargs["params"]["start"] == "now-P2D"


@patch("src.ingestion.ingest.requests.get")
def test_fetch_raw_prices_polymorphic_inputs(mock_get):
    """Verifies that python datetime and date objects are seamlessly formatted to string stamps."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"records": []}
    mock_get.return_value = mock_response

    # Test Case A: Inputting a native Python date object
    fetch_raw_prices(start=date(2026, 7, 11))
    first_call_params = mock_get.call_args_list[0][1]["params"]
    assert first_call_params["start"] == "2026-07-11T00:00"

    # Test Case B: Inputting a strict hardcoded text duration string
    fetch_raw_prices(start="now-P1M")
    second_call_params = mock_get.call_args_list[1][1]["params"]
    assert second_call_params["start"] == "now-P1M"


@patch("src.ingestion.ingest.requests.get")
def test_fetch_raw_prices_network_failures(mock_get):
    """Verifies that requests.exceptions safely return an empty list instead of uncaught crashes."""
    import requests
    mock_get.side_effect = requests.exceptions.RequestException("Connection Timed Out")
    
    records = fetch_raw_prices()
    assert records == []


# =====================================================================
# 2. TESTS FOR: save_prices_to_db (Database Upserts)
# =====================================================================

def test_save_prices_to_db_empty_records():
    """Verifies that passing no records exits early and returns 0 without calling the database."""
    assert save_prices_to_db([]) == 0


@patch("src.ingestion.ingest.Session")
@patch("src.ingestion.ingest.get_engine")  # Intercept get_engine to prevent real connection attempts
def test_save_prices_to_db_inserts_new_records(mock_get_engine, mock_session_class):
    """Verifies that completely new records are successfully staged and committed to the database."""
    # Setup mock session context manager
    mock_session_instance = MagicMock()
    mock_session_class.return_value.__enter__.return_value = mock_session_instance
    
    # Return None for search queries (indicating the record is new/does not exist)
    mock_result = MagicMock()
    mock_result.first.return_value = None
    mock_session_instance.exec.return_value = mock_result
    
    mock_records = [
        {"TimeDK": "2026-06-24 00:00", "PriceArea": "DK1", "DayAheadPriceDKK": 150.0}
    ]
    
    inserted_count = save_prices_to_db(mock_records)
    
    assert inserted_count == 1
    mock_session_instance.add.assert_called_once()
    mock_session_instance.commit.assert_called_once()


@patch("src.ingestion.ingest.Session")
@patch("src.ingestion.ingest.get_engine")  # Intercept get_engine
def test_save_prices_to_db_skips_duplicates(mock_get_engine, mock_session_class):
    """Verifies that records already existing in the database are ignored and not committed."""
    mock_session_instance = MagicMock()
    mock_session_class.return_value.__enter__.return_value = mock_session_instance
    
    from src.models.models import EnergyPrice
    existing_price_mock = MagicMock(spec=EnergyPrice)
    
    mock_result = MagicMock()
    mock_result.first.return_value = existing_price_mock
    mock_session_instance.exec.return_value = mock_result
    
    mock_records = [
        {"TimeDK": "2026-06-24 00:00", "PriceArea": "DK1", "DayAheadPriceDKK": 150.0}
    ]
    
    inserted_count = save_prices_to_db(mock_records)
    
    assert inserted_count == 0
    mock_session_instance.add.assert_not_called()
    mock_session_instance.commit.assert_not_called()