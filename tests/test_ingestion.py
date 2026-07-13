import json
import pytest
from datetime import datetime, date
from unittest.mock import patch, MagicMock

from src.ingestion.ingest import fetch_raw_prices

@patch("src.ingestion.ingest.requests.get")
def test_fetch_raw_prices_default_handling(mock_get):
    """Verifies the default duration string 'now-P2D' is correctly transmitted inside query parameters."""
    # Arrange: Build a dummy API JSON payload structure
    mock_response = MagicMock()
    mock_response.json.return_value = {"records": [{"TimeDK": "2026-06-24 00:00", "PriceArea": "DK1", "DayAheadPriceDKK": 150.0}]}
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