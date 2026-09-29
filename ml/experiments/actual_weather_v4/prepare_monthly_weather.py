import calendar
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.import_hungaromet_weather_seasonal import (
    MAPPING_FILE,
    PRECIPITATION_FILE,
    TEMPERATURE_FILE,
    load_mapping,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "monthly_weather.csv"

START_YEAR = 1971
END_YEAR = 2025
EXPECTED_GRIDPOINTS = 1233
EXPECTED_COUNTIES = 19


def build_positions(header, mapping):
    grid_indices = [int(value) for value in header.split()]

    if len(grid_indices) != EXPECTED_GRIDPOINTS:
        raise ValueError("Unexpected number of header gridpoints.")

    if len(set(grid_indices)) != len(grid_indices):
        raise ValueError("Duplicate gridpoint indices in header.")

    missing_indices = set(mapping) - set(grid_indices)

    if missing_indices:
        raise ValueError(
            f"Mapped gridpoints missing from header: {missing_indices}"
        )

    positions = {}

    for position, grid_index in enumerate(grid_indices):
        county = mapping.get(grid_index)

        if county is not None:
            positions.setdefault(county, []).append(position)

    if len(positions) != EXPECTED_COUNTIES:
        raise ValueError("Expected 19 counties.")

    return {
        county: np.asarray(indices, dtype=int)
        for county, indices in positions.items()
    }


def process_file(file_path, mapping, variable):
    monthly = {}
    seen_dates = set()
    last_date = None

    print(f"\nReading {file_path.name}...", flush=True)

    with file_path.open("r", encoding="ascii") as source:
        positions = build_positions(source.readline(), mapping)

        for line_number, line in enumerate(source, start=2):
            if not line.strip():
                continue

            current_date = date(
                int(line[0:4]),
                int(line[4:6].strip()),
                int(line[6:8].strip()),
            )

            if not START_YEAR <= current_date.year <= END_YEAR:
                continue

            if current_date in seen_dates:
                raise ValueError(
                    f"{file_path.name}: duplicate date {current_date}."
                )

            if last_date is not None and current_date < last_date:
                raise ValueError(
                    f"{file_path.name}: dates are not chronological."
                )

            seen_dates.add(current_date)
            last_date = current_date

            values = np.asarray(
                [float(value) for value in line[8:].split()],
                dtype=float,
            )

            if len(values) != EXPECTED_GRIDPOINTS:
                raise ValueError(
                    f"{file_path.name}, line {line_number}: "
                    f"expected {EXPECTED_GRIDPOINTS} values, "
                    f"found {len(values)}."
                )

            # Same missing-value convention as the original importer.
            values[values <= -900] = np.nan

            if np.isinf(values).any():
                raise ValueError(
                    f"{file_path.name}: infinite value on {current_date}."
                )

            for county, indices in positions.items():
                county_values = values[indices]
                valid_count = int(np.isfinite(county_values).sum())
                gridpoint_count = len(indices)

                if valid_count == 0:
                    raise ValueError(
                        f"No valid {variable} values for "
                        f"{county} on {current_date}."
                    )

                if variable == "precipitation":
                    if (county_values < 0).any():
                        raise ValueError(
                            f"Negative precipitation for "
                            f"{county} on {current_date}."
                        )

                daily_mean = float(np.nanmean(county_values))
                key = (current_date.year, current_date.month, county)

                if key not in monthly:
                    monthly[key] = {
                        "total": 0.0,
                        "days": 0,
                        "gridpoints": gridpoint_count,
                        "partial_missing_days": 0,
                        "min_valid_gridpoints": gridpoint_count,
                    }

                record = monthly[key]
                record["total"] += daily_mean
                record["days"] += 1
                record["partial_missing_days"] += int(
                    valid_count < gridpoint_count
                )
                record["min_valid_gridpoints"] = min(
                    record["min_valid_gridpoints"],
                    valid_count,
                )

            if current_date.month == 12 and current_date.day == 31:
                print(f"  Completed year {current_date.year}", flush=True)

    expected_day_count = (
        date(END_YEAR + 1, 1, 1) - date(START_YEAR, 1, 1)
    ).days

    if len(seen_dates) != expected_day_count:
        raise ValueError(
            f"{file_path.name}: expected {expected_day_count} unique days, "
            f"found {len(seen_dates)}."
        )

    records = []

    for (year, month, county), record in sorted(monthly.items()):
        required_days = calendar.monthrange(year, month)[1]

        if record["days"] != required_days:
            raise ValueError(
                f"Incomplete month: {county}, {year}-{month:02d}; "
                f"{record['days']} instead of {required_days} days."
            )

        if variable == "temperature":
            value = record["total"] / record["days"]
            value_column = "mean_temperature_c"
        else:
            value = record["total"]
            value_column = "precipitation_mm"

        records.append({
            "year": year,
            "month": month,
            "county_name": county,
            value_column: value,
            f"{variable}_days": record["days"],
            f"{variable}_gridpoints": record["gridpoints"],
            f"{variable}_partial_missing_days": (
                record["partial_missing_days"]
            ),
            f"{variable}_min_valid_gridpoints": (
                record["min_valid_gridpoints"]
            ),
        })

    result = pd.DataFrame(records)
    expected_rows = (END_YEAR - START_YEAR + 1) * 12 * EXPECTED_COUNTIES

    if len(result) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} monthly rows, found {len(result)}."
        )

    return result


def main():
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}. "
            "Use another output filename to preserve the previous result."
        )

    for file_path in [
        TEMPERATURE_FILE,
        PRECIPITATION_FILE,
        MAPPING_FILE,
    ]:
        if not file_path.is_file():
            raise FileNotFoundError(file_path)

    mapping = load_mapping()

    temperature = process_file(
        TEMPERATURE_FILE,
        mapping,
        "temperature",
    )
    precipitation = process_file(
        PRECIPITATION_FILE,
        mapping,
        "precipitation",
    )

    monthly = temperature.merge(
        precipitation,
        on=["year", "month", "county_name"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not monthly["_merge"].eq("both").all():
        raise ValueError("Temperature and precipitation keys differ.")

    monthly = (
        monthly.drop(columns="_merge")
        .sort_values(["year", "month", "county_name"])
        .reset_index(drop=True)
    )

    if not monthly["temperature_days"].eq(
        monthly["precipitation_days"]
    ).all():
        raise ValueError("Temperature and precipitation day counts differ.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "x", encoding="utf-8", newline=""
    ) as output:
        monthly.to_csv(output, index=False, float_format="%.8f")

    print("\n" + "=" * 80)
    print("MONTHLY WEATHER PREPARATION COMPLETED")
    print("=" * 80)
    print(f"Years: {START_YEAR}-{END_YEAR}")
    print(f"Counties: {monthly['county_name'].nunique()}")
    print(f"Monthly rows: {len(monthly)}")

    for variable in ["temperature", "precipitation"]:
        column = f"{variable}_partial_missing_days"
        affected_months = int(monthly[column].gt(0).sum())

        print(
            f"{variable}: county-months with partial gridpoint "
            f"missingness: {affected_months}"
        )
        print(
            f"{variable}: total affected county-days: "
            f"{int(monthly[column].sum())}"
        )

    print("\nMonthly value ranges:")
    print(
        monthly[["mean_temperature_c", "precipitation_mm"]]
        .agg(["min", "mean", "max"])
        .round(3)
        .to_string()
    )

    print(f"\nSaved to:\n{OUTPUT_FILE}")
    print("No database tables or model files were modified.")
    print("No model training or evaluation was performed.")


if __name__ == "__main__":
    main()