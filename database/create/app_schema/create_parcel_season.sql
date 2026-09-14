CREATE TABLE app.parcel_season (
    id BIGSERIAL PRIMARY KEY,

    parcel_id BIGINT NOT NULL
        REFERENCES app.parcel(id)
        ON DELETE CASCADE,

    season_start_year SMALLINT NOT NULL,

    crop VARCHAR(30) NOT NULL
        CHECK (crop IN ('wheat', 'barley', 'maize')),

    farming_strategy VARCHAR(30) NOT NULL
        CHECK (
            farming_strategy IN (
                'conventional',
                'reduced'
            )
        ),

    is_irrigated BOOLEAN NOT NULL DEFAULT FALSE,

    machine_use VARCHAR(20)
        CHECK (
            machine_use IS NULL
            OR machine_use IN ('low', 'medium', 'high')
        ),

    fertilizer_type VARCHAR(20)
        CHECK (
            fertilizer_type IS NULL
            OR fertilizer_type IN ('AN', 'urea', 'CAN')
        ),

    fertilizer_quantity_kg_ha NUMERIC(8,2)
        CHECK (
            fertilizer_quantity_kg_ha IS NULL
            OR fertilizer_quantity_kg_ha >= 0
        ),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (parcel_id, season_start_year)
);