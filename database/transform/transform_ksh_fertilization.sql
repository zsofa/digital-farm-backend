<!------- Put all metric data in one row with the given county and year ------->

INSERT INTO processed.fertilization (
    year,
    county_name,
    fertilized_area_ha,
    fertilizer_quantity_per_ha
)
SELECT
    year,
    county_name,

    MAX(value) FILTER (
        WHERE metric = 'fertilized_area'
    ) AS fertilized_area_ha,

    MAX(value) FILTER (
        WHERE metric = 'fertilizer_quantity_per_ha'
    ) AS fertilizer_quantity_per_ha

FROM raw.ksh_fertilization

GROUP BY
    year,
    county_name

ORDER BY
    year,
    county_name;
