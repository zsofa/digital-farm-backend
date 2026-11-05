INSERT INTO ml.training_dataset (
    yield_year,
    season_start_year,
    county_name,
    crop,

    soil_ph,
    soil_soc_g_kg,
    soil_clay_pct,

    expected_season_temperature_c,
    expected_season_precipitation_mm,

    recent_yield_mean_t_ha,

    average_yield_t_ha
)
SELECT
    c.year AS yield_year,

    CASE
        WHEN c.crop = 'maize'
            THEN c.year
        WHEN c.crop IN ('wheat', 'barley')
            THEN c.year - 1
    END AS season_start_year,

    c.county_name,
    c.crop,

    s.soil_ph,
    s.soil_soc_g_kg,
    s.soil_clay_pct,

    w.expected_season_temperature_c,
    w.expected_season_precipitation_mm,

    r.recent_yield_mean_t_ha,

    c.average_yield_t_ha

FROM processed.crop_production c

INNER JOIN processed.soil s
    ON s.county_name = c.county_name

INNER JOIN processed.recent_yield r
    ON r.target_year = c.year
    AND r.county_name = c.county_name
    AND r.crop = c.crop

INNER JOIN processed.weather_expected
    ON w.county_name = c.county_name
    AND w.crop = c.crop
    AND w.target_season_start_year =
        CASE
            WHEN c.crop = 'maize'
                THEN c.year
            WHEN c.crop IN ('wheat', 'barley')
                THEN c.year - 1
        END

WHERE c.average_yield_t_ha IS NOT NULL;