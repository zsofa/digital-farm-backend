 CREATE TABLE raw.ksh_irrigation (
    id BIGSERIAL PRIMARY KEY,
    metric VARCHAR(100) NOT NULL,
    county_name VARCHAR(30) NOT NULL,
    year SMALLINT NOT NULL,
    value NUMERIC,
    unit VARCHAR(30),
    source_file VARCHAR(200)
 )