CREATE TABLE app.parcel (
    id BIGSERIAL PRIMARY KEY,
    farm_id BIGINT NOT NULL
        REFERENCES app.farm(id)
        ON DELETE CASCADE,

    name VARCHAR(150) NOT NULL,

    geometry GEOMETRY(POLYGON, 4326) NOT NULL,

    official_area_ha NUMERIC(10,3) NOT NULL
        CHECK (official_area_ha > 0),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


CREATE INDEX idx_parcel_geometry
ON app.parcel
USING GIST (geometry);


INSERT INTO app.parcel (
    farm_id,
    name,
    geometry,
    official_area_ha
)
VALUES (
    1,
    'Test parcel',
    ST_GeomFromText(
        'POLYGON((
            17.60 47.68,
            17.61 47.68,
            17.61 47.69,
            17.60 47.69,
            17.60 47.68
        ))',
        4326
    ),
    10.0
);