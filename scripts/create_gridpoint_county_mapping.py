from pathlib import Path

import geopandas as gpd
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

GRID_FILE = (
    BASE_DIR / "data" / "raw" / "hungaroMet" / "gridpoint_coordinates.txt"
)

NUTS_FILE = (
    BASE_DIR / "data" / "raw" / "geography" / "NUTS_RG_01M_2024_4326.geojson"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" / "gridpoint_county_mapping.csv"
)

def main():
    print("Loading HungaroMet gridpoints....")

    grid = pd.read_csv(GRID_FILE, sep="\t")

    grid = grid.rename(
        columns={
            "Index": "grid_index",
            "Lambda": "longitude",
            "Fi": "latitude"
        }
    )

    print(f"Grid points loaded: {len(grid)}")

    # Convert coordinates to spatial points
    grid_gdf = gpd.GeoDataFrame(
        grid,
        geometry=gpd.points_from_xy(
            grid["longitude"],
            grid["latitude"]
        ),
        crs="EPSG:4326"
    )

    print("Loading NUTS county boundaries....")

    nuts = gpd.read_file(NUTS_FILE)

    # magyarország -> CNTR: HU, LEVL: 3
    counties =nuts[
        (nuts["CNTR_CODE"] == "HU")
        & (nuts["LEVL_CODE"] == 3)
    ].copy()

    # Remove Bp
    counties = counties[
        counties["NUTS_NAME"] != "Budapest"
    ]

    if len(counties) != 19:
        raise ValueError(f"Expected {19} counties, found {len(counties)}")

    # SPATIAL JOIN
    # which county contains which gridpoint
    mapping = gpd.sjoin(
        grid_gdf,
        counties[
            [
                "NUTS_ID",
                "NUTS_NAME",
                "geometry"
            ]
        ],
        how="left",
        predicate="within"
    )

    mapping = mapping.rename(
        columns={
            "NUTS_ID": "nuts_id",
            "NUTS_NAME": "county_name"
        }
    )

    #missing = mapping["county_name"].isna().sum()
    mapping = mapping[
        mapping["county_name"].notna()
    ].copy()

    #print(f"Gridpoints without county: {missing}")
    #print(f"Usable county gridppoints: {len(mapping)}")

    mapping = mapping[
        [
            "grid_index",
            "longitude",
            "latitude",
            "nuts_id",
            "county_name",
        ]
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Mapping saved to {OUTPUT_FILE}")

    """county_count = (
        mapping.groupby("county_name").size().sort_values()
    )"""
    #print(county_count)
    #print(f"Counties in {len(county_count)}")



if __name__ == "__main__":
    main()