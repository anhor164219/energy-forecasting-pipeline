from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field, UniqueConstraint

class EnergyPrice(SQLModel, table=True):
    """
    Represents a 15-minute interval energy price record in the database.

    This class maps to the 'energy_prices' table and handles the schema for 
    Danish day-ahead spot prices. It includes a composite unique constraint 
    to prevent duplicate entries for the same timestamp and region.

    Attributes:
        id: The primary key, auto-incremented by the database.
        time_dk: The local Danish timestamp (CET/CEST) for the price interval.
        price_area: The bidding zone identifier (e.g., 'DK1' for West, 'DK2' for East).
        day_ahead_price: The electricity spot price in DKK per MWh.
    """
    __tablename__: str = "energy_prices"

    id: Optional[int] = Field(
        default=None, 
        primary_key=True,
        description="Unique identifier for the record"
    )
    
    time_dk: datetime = Field(
        index=True, 
        nullable=False,
        description="The local Danish date and time for the 15-min interval"
    )
    
    price_area: str = Field(
        max_length=3, 
        nullable=False,
        description="The Danish price area: DK1 or DK2"
    )
    
    day_ahead_price: float = Field(
        nullable=False,
        description="The spot price in DKK"
    )

    __table_args__ = (
        UniqueConstraint("time_dk", "price_area", name="unique_price_per_area"),
    )