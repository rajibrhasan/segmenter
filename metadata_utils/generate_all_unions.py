import geopandas as gpd
from pyproj import Geod
from shapely.geometry import Polygon

def compare_area_methods(gdf, index=0):
    """
    Compares area computed geodesically and using EPSG:3106 projection.
    
    Parameters:
    - gdf: GeoDataFrame in EPSG:4326
    - index: row index of the polygon to calculate area for

    Returns:
    - Dictionary with geodesic and projected areas (m² and km²)
    """
    # --- Geodesic area using pyproj.Geod ---
    geod = Geod(ellps="WGS84")
    polygon = gdf.geometry.iloc[index]
    lon, lat = polygon.exterior.coords.xy
    area_geo, _ = geod.polygon_area_perimeter(lon, lat)
    area_geo = abs(area_geo)

    # --- Projected area using EPSG:3106 ---
    gdf_proj = gdf.to_crs(epsg=3106)
    polygon_proj = gdf_proj.geometry.iloc[index]
    area_proj = polygon_proj.area

    return {
        "geodesic_area_m2": area_geo,
        "geodesic_area_km2": area_geo / 1e6,
        "projected_area_m2": area_proj,
        "projected_area_km2": area_proj / 1e6
    }




# Step 1: Load the shapefile
shapefile_path = "data/bgd_adm_bbs_20201113_shp/bgd_admbnda_adm4_bbs_20201113.shp"
gdf = gpd.read_file(shapefile_path)

# result = compare_area_methods(gdf, index=0)
# print(result)

# print(gdf.columns)
# print(gdf.crs)

gdf_btm = gdf.to_crs(epsg=3106)

# Step 3: Calculate area in square meters
gdf_btm["area_m2"] = gdf_btm.geometry.area

# Step 4 (Optional): Convert to square kilometers
gdf_btm["Area"] = gdf_btm["area_m2"] / 1e6

# Step 5: Show the result
print(gdf_btm[["ADM4_EN", "Area"]].head())

# # Step 2: Select specific columns (replace with your column names)
selected_columns = ['ADM4_EN', 'ADM3_EN', 'ADM2_EN', 'ADM1_EN', 'Area']  # example column names
df_selected = gdf_btm[selected_columns]
print(df_selected.head())
# # Step 3: Export to Excel
output_path = "output1.xlsx"
df_selected.to_excel(output_path, index=False)
print(f"Saved selected columns to {output_path}")
