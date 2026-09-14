CREATE TABLE processed.crop_production (
    year SMALLINT NOT NULL,
    county_name VARCHAR(100) NOT NULL,
    crop VARCHAR(30) NOT NULL,

    harvested_area_ha NUMERIC,
    harvested_production_t NUMERIC,
    average_yield_t_ha NUMERIC(8,3),

    PRIMARY KEY (year, county_name, crop)
);