import geopandas as gpd
import rasterio
from rasterio.plot import show
from shapely.geometry import box
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from geopy.distance import geodesic
from matplotlib.patches import Patch, Circle
import time 

start_time = time.time()

district_df = pd.read_excel('data/district_coordinates.xlsx')
upazila_df = pd.read_excel('data/upazila_coordinates.xlsx')

district_radius = np.sqrt(20000000 / np.pi)  # 10 km² = 10,000,000 m²
upazila_radius = np.sqrt(12000000 / np.pi)

districts = gpd.read_file('data/bgd_adm_bbs_20201113_SHP/bgd_admbnda_adm2_bbs_20201113.shp')
upazilas = gpd.read_file('data/bgd_adm_bbs_20201113_SHP/bgd_admbnda_adm3_bbs_20201113.shp')
unions = gpd.read_file('data/bgd_adm_bbs_20201113_SHP/bgd_admbnda_adm4_bbs_20201113.shp')

unions = unions.to_crs(epsg=32646)
upazilas = upazilas.to_crs(epsg=32646)
districts = districts.to_crs(epsg=32646)

def visualize_intersecting_unions(intersecting_unions, intersecting_upazilas, intersecting_districts, tiff_gdf, max_union_name, max_upazila_name, tif_img, centroid):
    with open('data/metadata_backbone.json', 'r') as f:
        meta_data_json = json.load(f)
        union_metadata = meta_data_json[max_union_name+"_"+max_upazila_name]


    matched_districts = district_df[district_df['ADM2_EN'].isin(intersecting_unions['ADM2_EN'])]
    matched_upazilas = upazila_df[upazila_df['ADM3_EN'].isin(intersecting_unions['ADM3_EN'])]

    # Convert to GeoDataFrames
    district_gdf = gpd.GeoDataFrame(
        matched_districts,
        geometry=gpd.points_from_xy(matched_districts.Longitude, matched_districts.Latitude),
        crs='EPSG:4326'
    )

    upazila_gdf = gpd.GeoDataFrame(
        matched_upazilas,
        geometry=gpd.points_from_xy(matched_upazilas.Longitude, matched_upazilas.Latitude),
        crs='EPSG:4326'
    )

    tiff_centroid_coords = (centroid.y, centroid.x)

    # Find closest district center
    district_gdf['distance_km'] = district_gdf.geometry.apply(
        lambda pt: geodesic(tiff_centroid_coords, (pt.y, pt.x)).kilometers
    )

    # Find closest upazila center
    upazila_gdf['distance_km'] = upazila_gdf.geometry.apply(
        lambda pt: geodesic(tiff_centroid_coords, (pt.y, pt.x)).kilometers
    )

    upazila_gdf = upazila_gdf.copy().to_crs(epsg=32646)
    district_gdf = district_gdf.copy().to_crs(epsg=32646)


    district_circles = district_gdf.copy()
    district_circles['geometry'] = district_circles.buffer(district_radius)

    upazila_circles = upazila_gdf.copy()
    upazila_circles['geometry'] = upazila_circles.buffer(upazila_radius)

    closest_district = district_gdf.loc[district_gdf['distance_km'].idxmin()]
    closest_upazila = upazila_gdf.loc[upazila_gdf['distance_km'].idxmin()]

    union_metadata += f" This patch is {closest_district['distance_km']:0.2f} km away from the district center and {closest_upazila['distance_km']: 0.2f} km away from the upazila center."

    if closest_district['distance_km'] <= district_radius/1000:
        union_metadata += f" This patch is located inside of the District Sadar."
    elif closest_district['distance_km'] <= upazila_radius/1000:
        union_metadata += f" This patch is located outside of the District Sadar. This patch is located inside of the Upazila Sadar."
    else:
        union_metadata += f" This patch is located outside of the District Sadar. This patch is located outside of the Upazila Sadar."
    
    return union_metadata
 

    # fig, ax = plt.subplots()
    # intersecting_unions.plot(ax=ax, column='ADM4_EN', cmap='Reds', edgecolor='black', linewidth=0.5, alpha=0.6, label='Current patch')
    # intersecting_upazilas.plot(ax=ax, column='ADM3_EN', cmap ='viridis', edgecolor='green', alpha=0.6)
    # intersecting_districts.plot(ax=ax, column='ADM2_EN', cmap='tab20', edgecolor='black', alpha=0.6)
    
   

    # tiff_gdf.boundary.plot(ax=ax, color='red', linewidth=0.5)


    # # district_circles.plot(ax=ax, edgecolor='red', facecolor=None)
    # district_circles.plot(ax=ax, edgecolor='blue', facecolor='none', label='District Sadar(15 SqKm)')
    # upazila_circles.plot(ax=ax, edgecolor='yellow', facecolor='none', label='Upazila Sadar(5 SqKm)')

    # for _, row in intersecting_districts.iterrows():
    #     centroid = row['geometry'].centroid
    #     ax.text(centroid.x, centroid.y, row["ADM2_EN"], fontsize=4, ha='center', bbox=dict(facecolor='white', alpha=0.5, boxstyle='round'))

    # for _, row in intersecting_upazilas.iterrows():
    #     centroid = row['geometry'].centroid
    #     ax.text(centroid.x, centroid.y, row["ADM3_EN"], fontsize=4, ha='center', bbox=dict(facecolor='white', alpha=0.5, boxstyle='round'))

    # # for _, row in intersecting_unions.iterrows():
    # #     centroid = row['geometry'].centroid
    # #     ax.text(centroid.x, centroid.y, row["ADM4_EN"], fontsize=6, ha='center', bbox=dict(facecolor='white', alpha=0.5, boxstyle='round'))

    # plt.title(f"Maximum Intersection Union Name: {max_union_name}")
    
    # plt.axis('off')
    # plt.figtext(0.5, 0, union_metadata, wrap=True, horizontalalignment='center', verticalalignment = 'center', fontsize=10)
    
    # # plt.subplots_adjust(bottom=0.15, top=0.9)
    # # legend_elements = [
    # #     Patch(facecolor='none', edgecolor='blue', label='District Center (15 sq km)'),
    # #     Patch(facecolor='none', edgecolor='yellow', label='Upazila Center (5 sq km)'),
    # #     Patch(facecolor='none', edgecolor='red', label='Current Patch')
    # # ]
    # # ax.legend(handles=legend_elements, loc='upper right')

    # tif_basename = os.path.splitext(os.path.basename(tif_img))[0]
   
    # plt.savefig(f'data/overlays/{tif_basename}.png', dpi=300, bbox_inches='tight')
    # plt.close()

def get_union_name_from_tif(tif_img):
    with rasterio.open(tif_img) as src:
        bounds = src.bounds
        tiff_crs = src.crs

    tiff_bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    tiff_gdf = gpd.GeoDataFrame({'geometry': [tiff_bbox]}, crs=tiff_crs)

    centroid = tiff_gdf.copy().to_crs(epsg=4326).geometry[0].centroid
    tiff_gdf = tiff_gdf.to_crs(epsg=32646)
    tiff_bbox = tiff_gdf.geometry.iloc[0]

    intersecting_unions = unions[unions.intersects(tiff_bbox)].copy()
    intersecting_upazilas = upazilas[upazilas.intersects(tiff_bbox)].copy()
    intersecting_districts = districts[districts.intersects(tiff_bbox)].copy()

    intersecting_unions['intersection'] = intersecting_unions.geometry.intersection(tiff_bbox)
    intersecting_unions['intersection_area_m2'] = intersecting_unions['intersection'].area

    max_intersection_union = intersecting_unions.loc[intersecting_unions['intersection_area_m2'].idxmax()]
    max_union_name, max_upazila_name = max_intersection_union['ADM4_EN'], max_intersection_union['ADM3_EN']

    meta_data = visualize_intersecting_unions(intersecting_unions, intersecting_upazilas, intersecting_districts, tiff_gdf, max_union_name, max_upazila_name, tif_img, centroid)

    return meta_data

folder_list = ["D:/Senior_Project/BingRGB_Dhaka/data/tif_11Class/train/images", "D:/Senior_Project/BingRGB_Dhaka/data/tif_11Class/test/images"]
meta_data_dict = {}

for folder in folder_list:
    tif_files = glob.glob( folder+ "/*.tif")

    for tif_img in tif_files:
        meta_data = get_union_name_from_tif(tif_img)
        key = os.path.basename(tif_img).replace('tif', 'png')
        meta_data_dict[key] = meta_data

with open('data/captions.json', 'w', encoding='utf-8') as f:
    json.dump(meta_data_dict, f, ensure_ascii=False, indent=2)

print(f"Elapsed time: {time.time() - start_time}s")
