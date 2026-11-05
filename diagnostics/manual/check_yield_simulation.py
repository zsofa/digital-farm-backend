from pprint import pprint

from services.yield_simulation_service import create_yield_simulation


result = create_yield_simulation(1)

pprint(result)