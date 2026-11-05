from pprint import pprint

from services.parcel_soil_service import generate_parcel_soil
from services.parcel_soil_service import resolve_parcel_soil


#soil = generate_parcel_soil(5)
soil = resolve_parcel_soil(5)

pprint(soil)