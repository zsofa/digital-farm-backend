CREATE TABLE processed.county_boundary (
    county_name VARCHAR(100) PRIMARY KEY,
    geometry GEOMETRY(MULTIPOLYGON, 4326) NOT NULL
);

CREATE INDEX idx_county_boundary_geometry
ON processed.county_boundary
USING GIST (geometry);