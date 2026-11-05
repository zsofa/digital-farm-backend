from services.sustainability_service import (
    calculate_parcel_season_sustainability,
)


PARCEL_SEASON_ID = 3
USER_ID = 1


result = (
    calculate_parcel_season_sustainability(
        PARCEL_SEASON_ID,
        USER_ID,
    )
)

print(result)