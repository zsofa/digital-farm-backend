<!------- Put all metric data in one row with the given county and year ------->

INSERT INTO processed.irrigation (
    year,
    county_name,
    irrigated_area_ha,
    irrigated_water_per_ha_m3
)
SELECT
    year,
    county_name,

    MAX(value) FILTER (
        WHERE metric = 'irrigated_area'
    ) AS irrigated_area_ha,

    MAX(value) FILTER (
        WHERE metric = 'irrigated_water_per_ha'
    ) AS irrigated_water_per_ha_m3

FROM raw.ksh_irrigation

GROUP BY
    year,
    county_name

ORDER BY
    year,
    county_name;
