CREATE_TABLE processed.soil (
    county_name VARCHAR(100) PRIMARY KEY,

    soil_ph NUMERIC,
    soil_soc_g_kg NUMERIC(8,2),
    soil_clay_pct NUMERIC(6,2)
)