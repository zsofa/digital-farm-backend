from datetime import date


SUPPORTED_CROPS = {
    "wheat",
    "barley",
    "maize",
}


CROP_SEASONS = {
    "wheat": {
        "start": (9, 1),
        "end": (7, 31),
        "end_year_offset": 1,
    },
    "barley": {
        "start": (9, 1),
        "end": (7, 31),
        "end_year_offset": 1,
    },
    "maize": {
        "start": (3, 1),
        "end": (10, 31),
        "end_year_offset": 0,
    },
}


def resolve_next_full_season(
    crop,
    current_date=None,
):
    if crop not in SUPPORTED_CROPS:
        raise ValueError("Invalid crop.")

    today = current_date or date.today()

    season = CROP_SEASONS[crop]

    start_month, start_day = season["start"]
    end_month, end_day = season["end"]

    start_this_year = date(
        today.year,
        start_month,
        start_day,
    )

    # Only a not-yet-started season can be selected.
    if today < start_this_year:
        start_year = today.year
    else:
        start_year = today.year + 1

    end_year = (
        start_year
        + season["end_year_offset"]
    )

    start_date = date(
        start_year,
        start_month,
        start_day,
    )

    end_date = date(
        end_year,
        end_month,
        end_day,
    )

    return {
        "crop": crop,
        "season_start_year": start_year,
        "start_date": start_date,
        "end_date": end_date,
    }


def resolve_next_season_start_year(
    crop,
    current_date=None,
):
    season = resolve_next_full_season(
        crop,
        current_date,
    )

    return season["season_start_year"]