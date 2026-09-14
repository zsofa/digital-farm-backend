CREATE TABLE processed.irrigation (
    year SMALLINT NOT NULL,
    county_name VARCHAR(100) NOT NULL,

    irrigated_area_ha NUMERIC,
    irrigated_water_per_ha_m3 NUMERIC,

    PRIMARY KEY (year, county_name)
);