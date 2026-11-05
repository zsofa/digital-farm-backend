DROP TABLE IF EXISTS ml.training_dataset;

CREATE TABLE ml.training_dataset (
    yield_year SMALLINT NOT NULL,
    season_start_year SMALLINT NOT NULL,
    county_name VARCHAR(100) NOT NULL,
    crop VARCHAR(30) NOT NULL,

    ---Parcel/environmental features
    soil_ph NUMERIC(5,2),
    soil_soc_g_kg NUMERIC(8,2),
    soil_clay_pct NUMERIC(6,2),

    --- Weather known/estimated before target season
    expected_season_temperature_c NUMERIC(6,3),
    expected_season_precipitation_mm NUMERIC(8,2),

    ---- Historical local yield context
    recent_yield_mean_t_ha NUMERIC(8,3),

    ---ML target
    average_yield_t_ha NUMERIC(8,3),

    PRIMARY KEY (yield_year, county_name, crop)
)