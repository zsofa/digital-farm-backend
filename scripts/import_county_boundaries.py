from pathlib import Path
import geopandas as gpd

from ml.data_loader import get_connection


GEOJSON_FILE = Path("data/raw/geography/NUTS_RG_01M_2024_4326.geojson")

COUNTY_MAPPING = {
    "HU120": "Pest",
    "HU211": "Fejér",
    "HU212": "Komárom-Esztergom",
    "HU213": "Veszprém",
    "HU221": "Győr-Moson-Sopron",
    "HU222": "Vas",
    "HU223": "Zala",
    "HU231": "Baranya",
    "HU232": "Somogy",
    "HU233": "Tolna",
    "HU311": "Borsod-Abaúj-Zemplén",
    "HU312": "Heves",
    "HU313": "Nógrád",
    "HU321": "Hajdú-Bihar",
    "HU322": "Jász-Nagykun-Szolnok",
    "HU323": "Szabolcs-Szatmár-Bereg",
    "HU331": "Bács-Kiskun",
    "HU332": "Békés",
    "HU333": "Csongrád-Csanád",
}

def load_counties():

    dataframe = gpd.read_file(GEOJSON_FILE)
    dataframe = dataframe[dataframe["NUTS_ID"].isin(COUNTY_MAPPING.keys())].copy()

    if dataframe.crs is None:
        raise ValueError("GeoJSON CRS missing")


    dataframe = dataframe.to_crs(epsg=4326)
    dataframe["county_name"] = dataframe["NUTS_ID"].map(COUNTY_MAPPING)

    return dataframe

def import_counties(dataframe):
    query = """
        INSERT INTO processed.county_boundary (
            county_name,
            geometry
        )
        VALUES (
            %s,
            ST_Multi(ST_GeomFromText(%s, 4326))
        )
        ON CONFLICT (county_name)
        DO UPDATE SET
            geometry = EXCLUDED.geometry;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            for _, row in dataframe.iterrows():
                cursor.execute(
                    query,
                    (
                        row["county_name"],
                        row.geometry.wkt,
                    ),
                )

        connection.commit()
    finally:
        connection.close()


def main():
    dataframe = load_counties()

    print(f"Counties found: {len(dataframe)}")

    print(
        dataframe[
            ["NUTS_ID", "county_name"]
        ].sort_values("county_name").to_string(index=False)
    )

    if len(dataframe) != 19:
        raise ValueError(
            f"Expected 19 counties, found {len(dataframe)}."
        )

    import_counties(dataframe)

    print("\nCounty boundaries imported.")


if __name__ == "__main__":
    main()
