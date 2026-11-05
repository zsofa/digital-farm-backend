CREATE TABLE processed.recent_yield (
    target_year SMALLINT NOT NULL,
    county_name VARCHAR(100) NOT NULL,
    crop VARCHAR(30) NOT NULL,
    recent_yield_mean_t_ha NUMERIC(8,3),
    PRIMARY KEY (target_year, county_name, crop)
);