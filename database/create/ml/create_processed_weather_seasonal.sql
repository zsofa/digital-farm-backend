CREATE TABLE processed.weather_seasonal (
    season_start_year SMALLINT NOT NULL,
    county_name VARCHAR(100) NOT NULL,
    crop VARCHAR(30) NOT NULL,

    season_mean_temperature_c NUMERIC(6,3),
    season_precipitation_mm NUMERIC(8,2),

    PRIMARY KEY (season_start_year, county_name, crop)
)