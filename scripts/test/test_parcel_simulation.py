from pprint import pprint

from services.parcel_simulation_service import simulate_parcel_season


result = simulate_parcel_season(1)

pprint(result)