"""bikestress_route_with_residential.py

Enhanced version that:
1. Computes LTS for existing bike infrastructure (using facility type, speed, AADT)
2. Adds residential streets as fallback routes where no bike infrastructure exists

This gives a more complete picture of the bikeable network in Lexington.
"""

# Import the original script's functionality
import sys
from pathlib import Path
import warnings, math, re, webbrowser
import numpy as np
import pandas as pd
import geopandas as gpd
from pyogrio import read_dataframe, write_dataframe
gpd.options.io_engine = "pyogrio"
import folium
from folium.plugins import LocateControl, MeasureControl, Fullscreen, MousePosition

# Configuration
LEX_STREET_DATA_PATH = Path("lex_street_data.geojson")
GEOJSON_IN = Path("lexbike.geojson")  # Existing bike infrastructure
OUT_ENRICH = Path("lexbike_with_residential.geojson")
OUT_MAP = Path("lexbike_LTS_map_with_residential.html")

# Street classification criteria
BIKEABLE_RDCLASS = [5, 6, 7, 8]  # Road classes that are bikeable without infrastructure
# RDCLASS guide:
# 1: Interstate/Highway (UNBIKEABLE)
# 2: Parkway/Expressway (UNBIKEABLE)
# 3: Arterial (UNBIKEABLE without infrastructure)
# 4: Major Collector (UNBIKEABLE without infrastructure)
# 5: Minor Collector (BIKEABLE - residential)
# 6: Local Street (BIKEABLE - residential)
# 7: Service Road (BIKEABLE)
# 8: Alley (BIKEABLE)

UNBIKEABLE_RDCLASS = [1, 2, 3, 4]  # Road classes that are unbikeable without infrastructure
RESIDENTIAL_MAX_SPEED = 35  # Maximum speed limit for residential streets to include

# AADT settings
AADT_PATH = None  # Path to AADT data if available
AADT_MAX_DIST_M = 50

# Facility type detection (same as original script)
FACILITY_CANDIDATES = ["Type_Facility", "class", "Type_Road", "infra", "infra_type", "facility_t", "fac_type"]
CENTERLINE_CANDIDATES = ["_centerline", "center_line", "center"]
NAME_CANDIDATES = ["Name_Facility", "name", "street", "label", "road", "corridor"]

def pick_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def normalize_fac(val: str) -> str:
    s = (str(val) if val is not None else "").lower().strip()
    if any(k in s for k in ["protected", "separated", "cycletrack", "cycle track", "pbl"]):
        return "protected"
    if any(k in s for k in ["shared-use", "shared use", "multiuse", "multi-use", "path", "trail", "mup", "sup"]):
        return "path"
    if any(k in s for k in ["buffer", "bbl", "buffered"]):
        return "buffered_lane"
    if "no lane" in s:
        return "mixed"
    if any(k in s for k in ["bike lane", "bikelane", " lane", " bl", "(bl)", "painted"]):
        return "lane"
    if any(k in s for k in ["boulevard", "greenway", "neighborhood"]):
        return "boulevard"
    if any(k in s for k in ["mixed", "shared lane", "sharrow", "slm"]):
        return "mixed"
    return "unknown"

def compute_lts(row):
    """
    Compute LTS for bike infrastructure based on facility type, speed, and AADT
    (Same logic as original bikestress_route.py)
    """
    fac = row.get("__fac_cat", "unknown")
    speed = row.get("speed_mph", np.nan)
    aadt = row.get("aadt", np.nan)
    centerline = bool(row.get("__centerline", False))

    speed_known = not np.isnan(speed)
    aadt_known = not np.isnan(aadt)

    # Clear low-stress categories
    if fac in ["protected", "path"]:
        return 1
    if fac == "boulevard" and speed_known and speed <= 25 and (not aadt_known or aadt <= 1500):
        return 1

    if fac == "buffered_lane":
        if speed_known and speed <= 30 and (not aadt_known or aadt <= 8000):
            return 2
        return 3

    if fac == "lane":
        if speed_known and speed <= 30 and (not aadt_known or aadt <= 6000):
            return 2
        if speed_known and speed <= 35 and (not aadt_known or aadt <= 12000):
            return 3
        return 4

    if fac in ["mixed", "unknown"]:
        if speed_known and speed <= 25 and aadt_known and aadt <= 1500 and not centerline:
            return 2
        if speed_known and speed <= 30 and (not aadt_known or aadt <= 3000):
            return 3
        return 4

    return 4

def compute_residential_lts(speed_mph, rdclass, aadt=None):
    """
    Compute LTS for residential streets WITHOUT dedicated bike infrastructure
    Uses similar logic to compute_lts but assumes "mixed" facility type

    RDCLASS guide (Lexington-specific):
    5: Minor Collector (can be residential)
    6: Local Street (residential)
    """
    if pd.isna(speed_mph):
        speed_mph = 25  # Assume 25 mph if unknown (typical residential)

    speed_known = not pd.isna(speed_mph)
    aadt_known = not pd.isna(aadt) and aadt > 0

    # RDCLASS 6 (Local streets) - typically very low stress
    # These are like "mixed" facility with no centerline
    if rdclass == 6:
        # Apply similar logic to mixed/unknown from compute_lts
        if speed_known and speed_mph <= 25 and (not aadt_known or aadt <= 1500):
            return 2  # LTS 2: Low stress
        elif speed_known and speed_mph <= 30 and (not aadt_known or aadt <= 3000):
            return 3  # LTS 3: Moderate stress
        else:
            return 3  # Higher speed or unknown = LTS 3

    # RDCLASS 5 (Minor collectors) - slightly busier, treat as mixed with centerline
    elif rdclass == 5:
        if speed_known and speed_mph <= 25 and (not aadt_known or aadt <= 1500):
            return 2  # Still LTS 2 if very low speed/traffic
        elif speed_known and speed_mph <= 30 and (not aadt_known or aadt <= 3000):
            return 3
        else:
            return 3

    # Default to LTS 3 for other included road types
    return 3

def ensure_4326(gdf):
    return gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)

def project_to_local(gdf):
    return gdf.to_crs(32616)  # UTM 16N for Kentucky

def round_to_common_mph(v):
    """Round speeds to common values"""
    COMMON_MPH = np.array([15, 20, 25, 30, 35, 40, 45, 50, 55])
    if v is None or np.isnan(v):
        return v
    v = float(v)
    if v < 10 or v > 60:
        return v
    return float(COMMON_MPH[np.argmin(np.abs(COMMON_MPH - v))])


print("="*60)
print("STEP 1: LOADING DATA")
print("="*60)

# Load existing bike infrastructure
print(f"Loading existing bike infrastructure from {GEOJSON_IN}...")
bike_infra = read_dataframe(GEOJSON_IN)
bike_infra = ensure_4326(bike_infra)
print(f"  Loaded {len(bike_infra)} bike infrastructure segments")

# Load Lexington street data
print(f"\nLoading street network from {LEX_STREET_DATA_PATH}...")
streets = read_dataframe(LEX_STREET_DATA_PATH)
streets = ensure_4326(streets)
print(f"  Loaded {len(streets)} street segments")

print("\n" + "="*60)
print("STEP 2: COMPUTE LTS FOR BIKE INFRASTRUCTURE")
print("="*60)

# Detect facility type column
facility_col = pick_col(bike_infra.columns, FACILITY_CANDIDATES)
centerline_col = pick_col(bike_infra.columns, CENTERLINE_CANDIDATES)
name_col = pick_col(bike_infra.columns, NAME_CANDIDATES)

print(f"  Detected columns:")
print(f"    Facility: {facility_col}")
print(f"    Centerline: {centerline_col}")
print(f"    Name: {name_col}")

# Parse facility types
bike_infra["__fac_cat"] = bike_infra[facility_col].map(normalize_fac) if facility_col else "unknown"
bike_infra["__centerline"] = bike_infra[centerline_col].astype(str).str.lower().isin(["yes", "true", "1"]) if centerline_col else False

# Match bike segments to street speeds
print(f"\n  Matching bike infrastructure to street speeds...")
bike_infra['speed_mph'] = np.nan
bike_infra['__matched_street'] = False

# Normalize road names for matching
def normalize_road_name(name):
    if pd.isna(name) or name == '':
        return ''
    name = str(name).upper().strip()
    name = name.replace('STREET', 'ST').replace('ROAD', 'RD').replace('AVENUE', 'AVE')
    name = name.replace('DRIVE', 'DR').replace('BOULEVARD', 'BLVD').replace('PARKWAY', 'PKWY')
    name = name.replace('LANE', 'LN').replace('COURT', 'CT').replace('PLACE', 'PL')
    return name

bike_infra['__bike_road_name'] = bike_infra['Name_Network'].apply(normalize_road_name) if 'Name_Network' in bike_infra.columns else ''
streets['__street_road_name'] = streets['ROADNAME'].apply(normalize_road_name)
streets['speed_mph'] = pd.to_numeric(streets['SPEED'], errors='coerce')

# Direct road name match
name_matches = 0
for idx in bike_infra.index:
    bike_name = bike_infra.loc[idx, '__bike_road_name']
    if bike_name and bike_name != '':
        matching_streets = streets[streets['__street_road_name'] == bike_name]
        if len(matching_streets) > 0:
            median_speed = matching_streets['speed_mph'].median()
            if not pd.isna(median_speed):
                bike_infra.loc[idx, 'speed_mph'] = median_speed
                bike_infra.loc[idx, '__matched_street'] = True
                name_matches += 1

print(f"    Matched {name_matches} segments by road name")

# Fallback: Spatial matching for remaining
unmatched_idx = bike_infra[~bike_infra['__matched_street']].index
if len(unmatched_idx) > 0:
    print(f"    Spatial matching for {len(unmatched_idx)} remaining segments...")
    bike_unmatched_m = project_to_local(bike_infra.loc[unmatched_idx])
    streets_m = project_to_local(streets[['geometry', 'speed_mph']])

    spatial_match = gpd.sjoin_nearest(
        bike_unmatched_m,
        streets_m,
        how="left",
        max_distance=30,
        distance_col="__dist_speed"
    )
    spatial_match = spatial_match[~spatial_match.index.duplicated(keep='first')]

    # The joined column might have a suffix, check for it
    speed_col = 'speed_mph' if 'speed_mph' in spatial_match.columns else 'speed_mph_right'
    if speed_col in spatial_match.columns:
        matched_spatial = spatial_match[spatial_match[speed_col].notna()].index
        bike_infra.loc[matched_spatial, 'speed_mph'] = spatial_match.loc[matched_spatial, speed_col]
        print(f"Matched {len(matched_spatial)} segments spatially")
    else:
        print(f"Warning: Could not find speed column in spatial match")

# Round speeds
bike_infra["speed_mph"] = bike_infra["speed_mph"].apply(round_to_common_mph)

# Initialize AADT column (not using it yet, but needed for LTS function)
bike_infra['aadt'] = np.nan

# Compute LTS for bike infrastructure
print(f"\n  Computing LTS for bike infrastructure...")
bike_infra["LTS"] = bike_infra.apply(compute_lts, axis=1).astype(int)
bike_infra['source'] = 'bike_infrastructure'

print(f"\n  Bike infrastructure LTS distribution:")
for lts in sorted(bike_infra['LTS'].unique()):
    count = (bike_infra['LTS'] == lts).sum()
    pct = count / len(bike_infra) * 100
    print(f"    LTS {lts}: {count:,} segments ({pct:.1f}%)")

print(f"\n  Speed coverage: {bike_infra['speed_mph'].notna().sum()} / {len(bike_infra)} segments")

print("\n" + "="*60)
print("STEP 3: CLASSIFY ALL STREETS BY ROAD TYPE")
print("="*60)

# Classify ALL streets by their road characteristics
# Note: Some streets may overlap with bike infrastructure 
print(f"  Classifying all {len(streets):,} streets by road type...")

# Classify ALL streets
bikeable_streets = streets[streets['RDCLASS'].isin(BIKEABLE_RDCLASS)].copy()
unbikeable_streets = streets[streets['RDCLASS'].isin(UNBIKEABLE_RDCLASS)].copy()

print(f"\n  Classification:")
print(f"    Bikeable residential/local streets (RDCLASS {BIKEABLE_RDCLASS}): {len(bikeable_streets):,}")
print(f"    Unbikeable major roads (RDCLASS {UNBIKEABLE_RDCLASS}): {len(unbikeable_streets):,}")

# Filter bikeable streets by speed limit
bikeable_filtered = bikeable_streets[
    (bikeable_streets['speed_mph'].isna()) |
    (bikeable_streets['speed_mph'] <= RESIDENTIAL_MAX_SPEED)
].copy()
print(f"    Bikeable after speed filter (≤{RESIDENTIAL_MAX_SPEED} mph): {len(bikeable_filtered):,}")

# Compute LTS for bikeable streets
bikeable_filtered['LTS'] = bikeable_filtered.apply(
    lambda row: compute_residential_lts(row['speed_mph'], row['RDCLASS']),
    axis=1
)

# Create bikeable street network
bikeable_network = bikeable_filtered.copy()
bikeable_network['Type_Facility'] = bikeable_network['RDCLASS'].map({
    5: 'Residential Collector',
    6: 'Residential Street',
    7: 'Service Road',
    8: 'Alley'
})
bikeable_network['Name_Network'] = bikeable_network['ROADNAME']
bikeable_network['source'] = 'bikeable_streets'
bikeable_network['__fac_cat'] = 'residential'

print(f"\n  Bikeable street LTS distribution:")
for lts in sorted(bikeable_network['LTS'].unique()):
    count = (bikeable_network['LTS'] == lts).sum()
    pct = count / len(bikeable_network) * 100
    print(f"    LTS {lts}: {count:,} segments ({pct:.1f}%)")

# Handle unbikeable streets (LTS 5)
# Filter out segments that have bike infrastructure on them
print(f"\n  Filtering unbikeable streets to exclude those with bike infrastructure...")
print(f"    Initial unbikeable streets: {len(unbikeable_streets):,}")

# Create spatial index to identify which unbikeable streets have bike infrastructure
unbikeable_m = project_to_local(unbikeable_streets)
bike_infra_m = project_to_local(bike_infra)

# Buffer bike infrastructure by 20m to catch overlapping streets
bike_buffer = bike_infra_m.geometry.buffer(20)
bike_buffer_union = bike_buffer.union_all()

# Mark unbikeable streets that intersect with bike infrastructure
unbikeable_m['__has_bike_infra'] = unbikeable_m.geometry.intersects(bike_buffer_union)

# Filter to only unbikeable streets WITHOUT bike infrastructure
unbikeable_filtered = unbikeable_streets[~unbikeable_m['__has_bike_infra']].copy()

print(f"    Filtered out {len(unbikeable_streets) - len(unbikeable_filtered):,} segments with bike infrastructure")
print(f"    Remaining unbikeable streets: {len(unbikeable_filtered):,}")

unbikeable_network = unbikeable_filtered.copy()
unbikeable_network['LTS'] = 5  # LTS 5 = Unbikeable without dedicated infrastructure
unbikeable_network['Type_Facility'] = unbikeable_network['RDCLASS'].map({
    1: 'Interstate/Highway',
    2: 'Parkway/Expressway',
    3: 'Arterial',
    4: 'Major Collector'
})
unbikeable_network['Name_Network'] = unbikeable_network['ROADNAME']
unbikeable_network['source'] = 'unbikeable_without_infrastructure'
unbikeable_network['__fac_cat'] = 'unbikeable'

print(f"\n  Unbikeable streets (final): {len(unbikeable_network):,} segments")
print(f"    These are major roads without dedicated bike infrastructure")

print("\n" + "="*60)
print("STEP 4: COMBINE ALL NETWORKS")
print("="*60)

print(f"  Bike infrastructure: {len(bike_infra)} segments")
print(f"  Bikeable streets: {len(bikeable_network)} segments")
print(f"  Unbikeable streets: {len(unbikeable_network)} segments")

# Combine all networks
combined_network = pd.concat([
    bike_infra,
    bikeable_network,
    unbikeable_network
], ignore_index=True)

print(f"  Total network (complete coverage): {len(combined_network):,} segments")

print("\n" + "="*60)
print("LTS DISTRIBUTION (COMPLETE NETWORK)")
print("="*60)
for lts in sorted(combined_network['LTS'].unique()):
    count = (combined_network['LTS'] == lts).sum()
    pct = count / len(combined_network) * 100

    # Count by source
    infra_count = ((combined_network['LTS'] == lts) &
                   (combined_network['source'] == 'bike_infrastructure')).sum()
    bikeable_count = ((combined_network['LTS'] == lts) &
                      (combined_network['source'] == 'bikeable_streets')).sum()
    unbikeable_count = ((combined_network['LTS'] == lts) &
                        (combined_network['source'] == 'unbikeable_without_infrastructure')).sum()

    print(f"LTS {lts}: {count:,} segments ({pct:.1f}%)")
    if infra_count > 0:
        print(f"  - Bike infrastructure: {infra_count:,}")
    if bikeable_count > 0:
        print(f"  - Bikeable streets: {bikeable_count:,}")
    if unbikeable_count > 0:
        print(f"  - Unbikeable (major roads): {unbikeable_count:,}")
print("="*60)

# Save combined network
print(f"\nSaving combined network to {OUT_ENRICH}...")
write_dataframe(combined_network, OUT_ENRICH, driver="GeoJSON")
print(f"Saved {len(combined_network)} segments")

print("\n" + "="*60)
print("CREATING MAP")
print("="*60)

# Simplify geometries 
print("  Simplifying geometry...")
combined_simplified = combined_network.copy()
combined_simplified['geometry'] = combined_simplified['geometry'].simplify(
    tolerance=0.00001, preserve_topology=True
)

# Create map
print("  Creating map...")
m = folium.Map(
    location=[38.0406, -84.5037],
    zoom_start=12,
    prefer_canvas=True
)

# Add controls
LocateControl(auto_start=False, flyTo=True).add_to(m)
MeasureControl(primary_length_unit="meters").add_to(m)
Fullscreen().add_to(m)
MousePosition().add_to(m)

# Define LTS colors
LTS_COLORS = {
    1: "#00D084",  # Bright green
    2: "#3B82F6",  # Bright blue
    3: "#F59E0B",  # Bright amber/orange
    4: "#EF4444",  # Bright red
    5: "#9CA3AF"   # Gray - Unbikeable
}

LTS_NAMES = {
    1: "LTS 1 (Very Low Stress - Suitable for children)",
    2: "LTS 2 (Low Stress - Most adults comfortable)",
    3: "LTS 3 (Moderate Stress - Confident cyclists)",
    4: "LTS 4 (High Stress - Strong & fearless only)",
    5: "Unbikeable (Major roads without infrastructure)"
}

# Add layers by LTS level
print("  Adding map layers...")
for lts_level in [1, 2, 3, 4, 5]:
    subset = combined_simplified[combined_simplified['LTS'] == lts_level]

    if not subset.empty:
        # Prepare tooltip fields
        tooltip_fields = ['LTS']
        tooltip_aliases = ['LTS Level']

        if 'Type_Facility' in subset.columns:
            tooltip_fields.append('Type_Facility')
            tooltip_aliases.append('Type')

        if 'Name_Network' in subset.columns:
            tooltip_fields.append('Name_Network')
            tooltip_aliases.append('Street')

        if 'speed_mph' in subset.columns:
            tooltip_fields.append('speed_mph')
            tooltip_aliases.append('Speed (mph)')

        tooltip_fields.append('source')
        tooltip_aliases.append('Source')

        folium.GeoJson(
            subset.to_json(),
            name=LTS_NAMES[lts_level],
            style_function=lambda feature, lts=lts_level: {
                "color": LTS_COLORS[lts],
                "weight": 4,
                "opacity": 0.8,
                "lineJoin": "round",
                "lineCap": "round"
            },
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=True,
                sticky=False
            ),
            highlight_function=lambda x: {"weight": 6, "opacity": 1.0},
            overlay=True,
            show=True,
            smooth_factor=1.0
        ).add_to(m)

# Add legend 
lts_legend_html = """
<div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
            background: white; padding: 15px; border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3); font-size: 13px;
            max-width: 300px; font-family: Arial, sans-serif;">
<div style="font-weight: bold; font-size: 15px; margin-bottom: 10px;
            border-bottom: 2px solid #333; padding-bottom: 5px;">
Level of Traffic Stress (LTS)
</div>

<div style="margin: 8px 0; display: flex; align-items: center;">
<span style="display:inline-block; width:20px; height:20px;
             background:#00D084; border-radius:3px; margin-right:8px;"></span>
<div>
    <b>LTS 1:</b> Very Low Stress<br>
    <small style="color: #666;">Suitable for children</small>
</div>
</div>

<div style="margin: 8px 0; display: flex; align-items: center;">
<span style="display:inline-block; width:20px; height:20px;
             background:#3B82F6; border-radius:3px; margin-right:8px;"></span>
<div>
    <b>LTS 2:</b> Low Stress<br>
    <small style="color: #666;">Most adults comfortable</small>
</div>
</div>

<div style="margin: 8px 0; display: flex; align-items: center;">
<span style="display:inline-block; width:20px; height:20px;
             background:#F59E0B; border-radius:3px; margin-right:8px;"></span>
<div>
    <b>LTS 3:</b> Moderate Stress<br>
    <small style="color: #666;">Confident cyclists</small>
</div>
</div>

<div style="margin: 8px 0; display: flex; align-items: center;">
<span style="display:inline-block; width:20px; height:20px;
             background:#EF4444; border-radius:3px; margin-right:8px;"></span>
<div>
    <b>LTS 4:</b> High Stress<br>
    <small style="color: #666;">Strong & fearless only</small>
</div>
</div>

<div style="margin: 8px 0; display: flex; align-items: center;">
<span style="display:inline-block; width:20px; height:20px;
             background:#9CA3AF; border-radius:3px; margin-right:8px;"></span>
<div>
    <b>Unbikeable:</b> Major roads<br>
    <small style="color: #666;">Needs infrastructure</small>
</div>
</div>

<div style="margin-top: 12px; padding-top: 8px; border-top: 1px solid #ddd;
            font-size: 11px; color: #666;">
<b>Complete street coverage!</b><br>
Shows all streets in Lexington classified by bikeability.
</div>
</div>
"""
m.get_root().html.add_child(folium.Element(lts_legend_html))

# Fit map to bounds
if not combined_simplified.empty:
    minx, miny, maxx, maxy = combined_simplified.total_bounds
    m.fit_bounds([[miny, minx], [maxy, maxx]])

# Save map
print(f"  Saving map to {OUT_MAP}...")
m.save(str(OUT_MAP))

print("\n" + "="*60)
print("Complete")
print("="*60)
print(f"Network saved to: {OUT_ENRICH}")
print(f"Map saved to: {OUT_MAP}")
print(f"\nComplete street coverage: {len(combined_network):,} segments")
print(f"  - Bike infrastructure: {(combined_network['source'] == 'bike_infrastructure').sum():,}")
print(f"  - Bikeable streets: {(combined_network['source'] == 'bikeable_streets').sum():,}")
print(f"  - Unbikeable major roads: {(combined_network['source'] == 'unbikeable_without_infrastructure').sum():,}")
print(f"\nBikeable network (LTS 1-4): {(combined_network['LTS'] <= 4).sum():,} segments")
print(f"Low-stress routes (LTS 1-2): {(combined_network['LTS'] <= 2).sum():,} segments")
