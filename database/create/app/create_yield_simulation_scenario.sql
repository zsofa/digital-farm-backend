CREATE TABLE app.yield_simulation_scenario (
    id BIGSERIAL PRIMARY KEY,

    simulation_id BIGINT NOT NULL
        REFERENCES app.yield_simulation(id)
        ON DELETE CASCADE,

    scenario VARCHAR(20) NOT NULL
        CHECK (
            scenario IN (
                'dry_hot',
                'expected',
                'wet_cool'
            )
        ),

    temperature_c NUMERIC(6,3) NOT NULL,
    precipitation_mm NUMERIC(8,2) NOT NULL,

    predicted_yield_t_ha NUMERIC(8,3) NOT NULL,
    estimated_total_t NUMERIC(12,3) NOT NULL,

    UNIQUE (simulation_id, scenario)
);