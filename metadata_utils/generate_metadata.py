import os
import glob
import json
import time
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import argparse
from shapely.geometry import box
from tqdm import tqdm

# Constants
DISTRICT_AREA = 30000000  # 20 sq km in m²
UPAZILA_AREA = 15000000   # 12 sq km in m²
district_radius = np.sqrt(DISTRICT_AREA / np.pi)
upazila_radius = np.sqrt(UPAZILA_AREA / np.pi)

# Load shapefiles once globally
shp_path = 'bgd_adm_bbs_20201113_shp'
districts = gpd.read_file(f'{shp_path}/bgd_admbnda_adm2_bbs_20201113.shp').to_crs(epsg=32646)
upazilas = gpd.read_file(f'{shp_path}/bgd_admbnda_adm3_bbs_20201113.shp').to_crs(epsg=32646)
unions = gpd.read_file(f'{shp_path}/bgd_admbnda_adm4_bbs_20201113.shp').to_crs(epsg=32646)

# Metadata
district_df = pd.read_excel('files/district_coordinates.xlsx')
upazila_df = pd.read_excel('files/upazila_coordinates.xlsx')


def generate_base_metadata(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # Construct key
    df['Union_Upazila'] = df['ADM4_EN'].str.strip() + '_' + df['Upazila'].str.strip()

    # Required fields
    metadata_fields = [
        'Location',
        'Upazila',
        'District',
        'Population Density',
        'Literacy',
        'Region Type'
    ]

    # Build dictionary
    metadata_dict = {
        row['Union_Upazila']: {field: row[field] for field in metadata_fields}
        for _, row in df.iterrows()
    }

    return metadata_dict

# Utility
def create_point_gdf(df, lat_col, lon_col, crs='EPSG:4326'):
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=crs
    ).to_crs(epsg=32646)

district_gdf = create_point_gdf(district_df, 'Latitude', 'Longitude')
upazila_gdf = create_point_gdf(upazila_df, 'Latitude', 'Longitude')

def get_patch_metadata(tif_path, base_metadata):
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        tiff_crs = src.crs

    tiff_bbox = gpd.GeoDataFrame({'geometry': [box(*bounds)]}, crs=tiff_crs).to_crs(epsg=32646)
    tiff_geom = tiff_bbox.geometry.iloc[0]
    tiff_centroid = tiff_geom.centroid

    intersecting_unions = unions[unions.intersects(tiff_geom)]

    intersecting_unions['intersection_area'] = intersecting_unions.intersection(tiff_geom).area
    max_union = intersecting_unions.loc[intersecting_unions['intersection_area'].idxmax()]

    key = f"{max_union['ADM4_EN'].strip()}_{max_union['ADM3_EN'].strip()}"
    base_info = base_metadata.get(key)

    upazila_name = max_union["ADM3_EN"].strip()
    district_name = max_union["ADM2_EN"].strip()

    # Find matching coordinates
    matching_district = district_gdf[district_gdf["ADM2_EN"].str.strip() == district_name].copy()
    matching_upazila = upazila_gdf[upazila_gdf["ADM3_EN"].str.strip() == upazila_name].copy()

    # Compute distances
    district_dist_km = matching_district.distance(tiff_centroid).min() / 1000
    upazila_dist_km = matching_upazila.distance(tiff_centroid).min() / 1000

    inside_district_sadar = bool(district_dist_km <= district_radius / 1000)
    inside_upazila_sadar = bool(not inside_district_sadar and upazila_dist_km <= upazila_radius / 1000)

    return {
        **base_info,
        "distance_to_district_sadar": f'{round(district_dist_km, 2)} km',
        "distance_to_upazila_sadar": f'{round(upazila_dist_km, 2)} km',
        "inside_district_sadar": inside_district_sadar,
        "inside_upazila_sadar": inside_upazila_sadar
    }

def main(folders):
    start_time = time.time()
    base_metadata = generate_base_metadata('region_metadata.xlsx', 'metadata')
    all_captions = {}

    for folder in folders:
        tif_files = glob.glob(os.path.join(folder, "*.tif"))
        print(f"Processing {len(tif_files)} TIFF files in {folder}")
        captions = {}

        for tif_path in tqdm(tif_files, desc=f"Folder: {os.path.basename(folder)}"):
            key = os.path.basename(tif_path).replace('.tif', '.png')
            captions[key] = get_patch_metadata(tif_path, base_metadata)
        
        all_captions.update(captions)


    with open("captions.json", 'w', encoding='utf-8') as f:
        json.dump(all_captions, f, ensure_ascii=False, indent=2)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    folders = ["/teamspace/studios/this_studio/segmenter/Dataset/BingRGB/train/images", "/teamspace/studios/this_studio/segmenter/Dataset/BingRGB/train/images"]
    main(folders)
