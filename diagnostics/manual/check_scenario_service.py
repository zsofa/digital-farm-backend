from pprint import pprint

from ml.scenarios.scenario_service import simulate_yield_scenarios


result = simulate_yield_scenarios(
    county_name="Győr-Moson-Sopron",
    crop="maize",
    area_ha=10,
)

pprint(result)