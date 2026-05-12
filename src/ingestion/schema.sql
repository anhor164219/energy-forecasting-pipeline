CREATE TABLE IF NOT EXISTS "energy_prices"(
    "id" SERIAL PRIMARY KEY,
    "hour_dk" TIMESTAMP NOT NULL,
    "price_area" VARCHAR(3) NOT NULL,
    "day_ahead_price" FLOAT NOT NULL,
    CONSTRAINT unique_price_per_area UNIQUE (hour_dk, price_area)
)
