CREATE TABLE raw.ksh_crop_production (
    id BIGSERIAL PRIMARY KEY,
    crop VARCHAR(30) NOT NULL,
    metric VARCHAR(100) NOT NULL,
    county_name VARCHAR(100) NOT NULL,
    year SMALLINT NOT NULL,
    value NUMERIC,
    unit VARCHAR(30),
    source_file VARCHAR(150)
);