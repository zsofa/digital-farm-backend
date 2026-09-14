CREATE TABLE processed.weather (
    year SMALLINT NOT NULL,
    county_name VARCHAR(100) NOT NULL,

    annual_mean_temperature_c NUMERIC(6,3),
    annual_precipitation_mm NUMERIC(8,2),

    gridpoint_count SMALLINT NOT NULL,

    PRIMARY KEY (year, county_name)
);