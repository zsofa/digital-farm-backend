from pathlib import Path

import geopandas as gpd
from owslib.wcs import WebCoverageService
from pyproj import CRS, Transformer
from shapely.ops import transform

BASE_DIR = Path(__file__).resolve().parent.parent

NUTS_FILE = (
    BASE_DIR / "data" / "raw" / "geography" / "NUTS_RG_01M_2024_4326.geojson"
)

OUTPUT_DIR = BASE_DIR / "data" / "raw" / "soilgrids"

PROPERTIES = [
    "phh2o",
    "soc",
    "clay"
]

DEPTHS = [
    "0-5cm",
    "5-15cm",
    "15-30cm"
]

# SoilGrids native projection:
# Interrupted Goode Homolosine, WGS84
SOILGRIDS_PROJ = (
    "+proj=igh "
    "+lat_0=0 "
    "+lon_0=0 "
    "+datum=WGS84 "
    "+units=m "
    "+no_defs"
)

SOILGRIDS_WCS_CRS = "urn:ogc:def:crs:EPSG::152160"
RESOLUTION = 250

def get_hu_bounds():
    print("Getting HU boundaries....")

    nuts = gpd.read_file(NUTS_FILE)

    hungary = nuts[
        (nuts["CNTR_CODE"] == "HU") & (nuts["LEVL_CODE"] == 3)
    ].copy()

    if len(hungary) != 20:
        raise ValueError(f"Expected 20 HU NUTS3 units, not {len(hungary)}")


    hungary_geometry = hungary.geometry.union_all()
    source_crs = CRS.from_epsg(4326)

    target_crs = CRS.from_proj4(SOILGRIDS_PROJ)

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    projected_geometry = transform(transformer.transform, hungary_geometry)

    min_x, min_y, max_x, max_y = projected_geometry.bounds

    padding = 1000

    bounds = (
        min_x - padding,
        min_y - padding,
        max_x + padding,
        max_y + padding
    )

    print("HU SoilGrids bounds: ")
    print(
        f"X: {bounds[0]:.0f} - {bounds[2]:.0f}"
    )
    print(
        f"Y: {bounds[1]:.0f} - {bounds[3]:.0f}"
    )

    return bounds


def connect_to_wcs(property_name):
    url = (
        "https://maps.isric.org/mapserv"
        f"?map=/map/{property_name}.map"
    )

    print(
        f"Connecting to SoilGrids WCS: "
        f"{property_name}"
    )

    return WebCoverageService(
        url,
        version="1.0.0",
        timeout=120,
    )

def download_layer(
    wcs,
    property_name,
    depth,
    bounds,
):
    coverage_id = (
        f"{property_name}_{depth}_mean"
    )

    print(
        f"\nDownloading {coverage_id}..."
    )

    # Verify that the requested coverage
    # actually exists in SoilGrids.
    if coverage_id not in wcs.contents:
        raise ValueError(
            f"Coverage not found: {coverage_id}"
        )

    coverage = wcs.contents[coverage_id]

    print(
        f"Supported formats: "
        f"{coverage.supportedFormats}"
    )

    min_x, min_y, max_x, max_y = bounds

    bbox = (
        min_x,
        min_y,
        max_x,
        max_y,
    )

    response = wcs.getCoverage(
        identifier=coverage_id,
        crs=SOILGRIDS_WCS_CRS,
        bbox=bbox,
        resx=RESOLUTION,
        resy=RESOLUTION,
        format="GEOTIFF_INT16",
    )

    output_file = (
        OUTPUT_DIR
        / f"{coverage_id}.tif"
    )

    with open(
        output_file,
        "wb",
    ) as file:
        file.write(
            response.read()
        )

    size_mb = (
        output_file.stat().st_size
        / 1024
        / 1024
    )

    print(
        f"Saved: {output_file.name} "
        f"({size_mb:.2f} MB)"
    )


def main():
    print("Downloading SoilGrids data for hu")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    bounds = get_hu_bounds()

    for property_name in PROPERTIES:

        wcs = connect_to_wcs(
            property_name
        )

        available_mean_layers = [
            coverage_id
            for coverage_id
            in wcs.contents.keys()
            if "_mean" in coverage_id
        ]

        print(
            f"{property_name}: "
            f"{len(available_mean_layers)} "
            f"mean layers available."
        )

        for depth in DEPTHS:
            download_layer(
                wcs,
                property_name,
                depth,
                bounds,
            )

    print(
        "\nSoilGrids download completed."
    )

    downloaded_files = list(
        OUTPUT_DIR.glob("*.tif")
    )

    print(
        f"Downloaded GeoTIFF files: "
        f"{len(downloaded_files)}"
    )

    if len(downloaded_files) != 9:
        raise ValueError(
            "Expected 9 GeoTIFF files."
        )


if __name__ == "__main__":
    main()
