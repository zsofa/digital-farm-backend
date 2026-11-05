CREATE TABLE app.yield_simulation (
    id BIGSERIAL PRIMARY KEY,

    parcel_season_id BIGINT NOT NULL
        REFERENCES app.parcel_season(id)
        ON DELETE CASCADE,

    county_name VARCHAR(100) NOT NULL,

    soil_source VARCHAR(20) NOT NULL,
    soil_ph NUMERIC(5,2) NOT NULL,
    soil_soc_g_kg NUMERIC(8,2) NOT NULL,
    soil_clay_pct NUMERIC(6,2) NOT NULL,

    recent_yield_mean_t_ha NUMERIC(8,3) NOT NULL,

    model_type VARCHAR(100) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);