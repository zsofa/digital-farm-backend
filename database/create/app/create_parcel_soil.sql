CREATE TABLE app.parcel_soil (
    id BIGSERIAL PRIMARY KEY,

    parcel_id BIGINT NOT NULL
        REFERENCES app.parcel(id)
        ON DELETE CASCADE,

    source VARCHAR(20) NOT NULL
        CHECK (source IN ('soilgrids', 'user')),

    soil_ph NUMERIC(5,2) NOT NULL,
    soil_soc_g_kg NUMERIC(8,2) NOT NULL,
    soil_clay_pct NUMERIC(6,2) NOT NULL,

    measured_at DATE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_parcel_soil_parcel_id
ON app.parcel_soil(parcel_id);