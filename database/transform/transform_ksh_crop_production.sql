<!------- Put all metric data in one row with the given crop, county and year ------->
<!-------- Convert average yield kg/ha to t/ha ----------------->

INSERT INTO processed.crop_production (
    year,
    county_name,
    crop,
    harvested_area_ha,
    harvested_production_t,
    average_yield_t_ha
)
SELECT
    year,
    county_name,
    crop,

    MAX(value) FILTER (
        WHERE metric = 'harvested_area'
    ) AS harvested_area_ha,

    MAX(value) FILTER (
        WHERE metric = 'harvested_production'
    ) AS harvested_production_t,

    MAX(value) FILTER (
        WHERE metric = 'average_yield'
    ) / 1000.0 AS average_yield_t_ha

FROM raw.ksh_crop_production

GROUP BY
    year,
    county_name,
    crop

ORDER BY
    year,
    county_name,
    crop;
