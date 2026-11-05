CREATE TABLE processed.weather_expected (
    target_season_start_year SMALLINT NOT NULL,
    county_name VARCHAR(100) NOT NULL,
    crop VARCHAR(30) NOT NULL,
    expected_season_temperature_c NUMERIC(6,3),
    expected_season_precipitation_mm NUMERIC(8,2),
    PRIMARY KEY (target_season_start_year, county_name, crop)
)