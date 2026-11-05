CREATE TABLE processed.fertilization (
    year SMALLINT NOT NULL,
    county_name VARCHAR(100) NOT NULL,

    fertilized_area_ha NUMERIC,
    fertilizer_quantity_per_ha NUMERIC,

    PRIMARY KEY (year, county_name)
);