DROP TABLE IF EXISTS energy_prices;

CREATE TABLE energy_prices (
    id SERIAL PRIMARY KEY,
    hour_dk TIMESTAMP NOT NULL,
    price_area VARCHAR(3) NOT NULL,
    day_ahead_price FLOAT NOT NULL,
    
    -- This ensures we only have ONE price per area per hour
    CONSTRAINT unique_price_per_area UNIQUE (hour_dk, price_area)
);