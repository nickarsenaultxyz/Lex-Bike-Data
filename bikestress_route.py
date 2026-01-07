# -*- coding: utf-8 -*-
"""bikestress_route.py

Updates coming:
1. Include walking paths through campus. Should be able to pull walking paths from a box using OSMnx
2. Create the routing feature
3. All of this should probably be done with GIT so that I can branch off and not mess with the base version
4. There is definitely some data in here that doesn't match with reality. For example the non-existant funded bike path on Liberty?
5. How do they define mixed type facility?
"""

# lex_lts_osm_aadt.py
# Usage: python lex_lts_osm_aadt.py

from pathlib import Path
import warnings, math, re, webbrowser
import numpy as np
import pandas as pd
import geopandas as gpd
from pyogrio import read_dataframe, write_dataframe
try:
    from pyogrio import list_layers as _list_layers
except Exception:
    _list_layers = None
gpd.options.io_engine = "pyogrio"  # use pyogrio for IO
import osmnx as ox
import shapely
import folium
from folium.plugins import LocateControl, MeasureControl, Fullscreen, MousePosition

# OSMnx 2.x moved project_gdf:
try:
    from osmnx.projection import project_gdf as _project_gdf
except Exception:
    _project_gdf = getattr(ox, "project_gdf", None)

# Loading data
GEOJSON_IN = Path("lexbike.geojson")           # bike-lane file (GeoJSON/GPKG/Shapefile ok)
LEX_STREET_DATA_PATH = Path("lex_street_data.geojson")  # Lexington street data with speed limits
CITY       = "Lexington, Kentucky, USA"        # OSM place query
OUT_ENRICH = Path("lexbike_with_speed_aadt.geojson")
OUT_MAP    = Path("lexbike_LTS_map.html")
CITY       = "Lexington, KY"

# Speed data segment snapping
NEAREST_MAX_DIST_M = 30   # distance for matching bike segments to speed data

# Speed limit overrides (for known incorrect speed data)
# You can override by street road name OR by facility type/name
# Format: {"search_string": speed_in_mph}
# The search string will match against both street road name AND facility name
SPEED_OVERRIDES = {
    # Example by road name: "Main St": 25,
    # Example by facility: "Liberty Road": 35,
    # Add your known corrections here
}

# Google Roads API (optional - for fetching speed limits)
# Get a free API key at: https://developers.google.com/maps/documentation/roads/get-api-key
GOOGLE_ROADS_API_KEY = None  # Set to your API key string to enable

# OPTIONAL: AADT data (KYTC or city open data)
# Supported: point or line layer (GeoJSON/GPKG/SHP). If CSV, include lon/lat cols set below.
AADT_PATH =  None # Path("Traffic_Station_Counts.geojson") e.g., Path("k ytc_aadt_2023.geojson")
CSV_LON   = "lon" # change if you use CSV
CSV_LAT   = "lat"
AADT_MAX_DIST_M = 50  # nearest distance for AADT match

# If bike lane file has facility info, set column aliases here (best guess fallback below)
FACILITY_CANDIDATES   = ["Type_Facility","class","Type_Road","infra","infra_type","facility_t","fac_type"]
CENTERLINE_CANDIDATES = ["_centerline","center_line","center"]
NAME_CANDIDATES       = ["Name_Facility", "name","street","label","road","corridor"]

# Used for rounding to nearest common mph
COMMON_MPH = np.array([15,20,25,30,35,40,45,50,55])


def pick_col(cols, candidates):
    for c in candidates:
        if c in cols: return c
    return None

def mph_from_kph(k): return k * 0.621371

def normalize_fac(val: str) -> str:
    s = (str(val) if val is not None else "").lower().strip()
    if any(k in s for k in ["protected","separated","cycletrack","cycle track","pbl"]): return "protected"
    if any(k in s for k in ["shared-use","shared use","multiuse","multi-use","path","trail","mup","sup"]): return "path"
    if any(k in s for k in ["buffer","bbl","buffered"]): return "buffered_lane"
    if "no lane" in s: return "mixed"
    if any(k in s for k in ["bike lane","bikelane"," lane", " bl", "(bl)","painted"]): return "lane"
    if any(k in s for k in ["boulevard","greenway","neighborhood"]): return "boulevard"
    if any(k in s for k in ["mixed","shared lane","sharrow","slm"]): return "mixed"
    return "unknown"


def compute_lts(row):
    fac = row.get("__fac_cat","unknown")
    speed = row.get("speed_mph", np.nan)
    aadt  = row.get("aadt", np.nan)
    centerline = bool(row.get("__centerline", False))

    speed_known = not np.isnan(speed)
    aadt_known  = not np.isnan(aadt)

    # Clear low-stress categories
    if fac in ["protected","path"]: return 1
    if fac == "boulevard" and speed_known and speed <= 25 and (not aadt_known or aadt <= 1500): return 1

    if fac == "buffered_lane":
        if speed_known and speed <= 30 and (not aadt_known or aadt <= 8000): return 2
        return 3

    if fac == "lane":
        if speed_known and speed <= 30 and (not aadt_known or aadt <= 6000): return 2
        if speed_known and speed <= 35 and (not aadt_known or aadt <= 12000): return 3
        return 4

    if fac in ["mixed","unknown"]:
        if speed_known and speed <= 25 and aadt_known and aadt <= 1500 and not centerline: return 2
        if speed_known and speed <= 30 and (not aadt_known or aadt <= 3000): return 3
        return 4

    return 4

def ensure_4326(gdf):
    return gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)


def sanitize_aadt(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")          # force numeric, non-numeric -> NaN
    s = s.replace([np.inf, -np.inf], np.nan)

    # Typical KY AADT: 0–200k; interstates can be higher, but 300k is a safe hard cap
    hard_cap = 300_000
    s[(s < 0) | (s > hard_cap)] = np.nan

    # Optional: dynamic cap to catch a few extreme outliers in otherwise clean data
    if s.notna().sum() >= 20:                            # only if enough data
        p99 = np.nanpercentile(s, 99)
        dyn_cap = max(100_000, p99 * 1.5)                # keep plausible heavy corridors
        s[s > dyn_cap] = np.nan

    return s

# OSM speed prep (no direct edges["maxspeed"] access)
def project_to_local(gdf):
    return _project_gdf(gdf) if _project_gdf else gdf.to_crs(32616)  # UTM 16N ~ KY

# parse explicit maxspeed if present; else all-NaN
def _parse_maxspeed_to_mph(val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return np.nan
    if isinstance(val, (list, tuple, set)):
        vals = [_parse_maxspeed_to_mph(v) for v in val]
        vals = [v for v in vals if not np.isnan(v)]
        return min(vals) if vals else np.nan
    s = str(val).strip().lower()
    if s in ("signals","walk","none"): return np.nan
    m = __import__("re").findall(r"(\d+(?:\.\d+)?)\s*(km/h|kph|mph)?", s)
    if m:
        mphs = [ (float(num)*0.621371 if unit in ("km/h","kph") else float(num)) for num, unit in m ]
        return min(mphs) if mphs else np.nan
    try:
        return float(s)
    except:
        return np.nan

def norm_highway(v):
    if isinstance(v, (list, tuple)):
        return str(v[0]) if v else ""
    return str(v)

def round_to_common_mph(v):
    if v is None or np.isnan(v): return v
    v = float(v)
    if v < 10 or v > 60: return v
    return float(COMMON_MPH[np.argmin(np.abs(COMMON_MPH - v))])

def get_bearing(geom):
    """Calculate the bearing (direction) of a geometry in degrees (0-360)"""
    if geom is None or geom.is_empty or geom.length == 0:
        return None

    # Handle MultiLineString by using the first part
    if geom.geom_type == 'MultiLineString':
        if len(geom.geoms) == 0:
            return None
        line = geom.geoms[0]  # Use first line segment
    elif geom.geom_type == 'LineString':
        line = geom
    else:
        return None

    coords = list(line.coords)
    if len(coords) < 2:
        return None
    # Use start and end points
    x1, y1 = coords[0]
    x2, y2 = coords[-1]
    # Calculate angle
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    # Normalize to 0-360
    bearing = (angle + 360) % 360
    return bearing

def bearing_difference(bearing1, bearing2):
    """Calculate the smallest difference between two bearings (0-180 degrees)"""
    if bearing1 is None or bearing2 is None:
        return 180  # Maximum difference if we can't calculate
    diff = abs(bearing1 - bearing2)
    # Handle wrapping (e.g., 10° and 350° are only 20° apart)
    if diff > 180:
        diff = 360 - diff
    # Also check if they're parallel but opposite direction
    # (e.g., northbound vs southbound on same street)
    opposite_diff = abs(180 - diff)
    return min(diff, opposite_diff)

def apply_speed_overrides(gdf, osm_name_col, facility_name_col, overrides):
    """
    Apply manual speed limit overrides to segments
    Searches in both OSM street name and facility name columns
    """
    if not overrides:
        return gdf

    count = 0
    for search_string, speed_mph in overrides.items():
        # Search in both OSM street name and facility name
        mask = pd.Series(False, index=gdf.index)

        if osm_name_col and osm_name_col in gdf.columns:
            mask |= gdf[osm_name_col].str.contains(search_string, case=False, na=False)

        if facility_name_col and facility_name_col in gdf.columns:
            mask |= gdf[facility_name_col].str.contains(search_string, case=False, na=False)

        if mask.any():
            gdf.loc[mask, 'speed_mph'] = speed_mph
            count += mask.sum()
            print(f"    Override: '{search_string}' -> {speed_mph} mph ({mask.sum()} segments)")

    if count > 0:
        print(f"  Applied {count} manual speed overrides")
    return gdf

def parse_width(width_str):
    """
    Parse OSM width strings like "3.5 m", "12 ft", or just "12"
    Returns width in feet
    Handles arrays/lists by taking the first value
    """
    # Handle None/NaN
    if width_str is None:
        return np.nan

    # Handle arrays/lists (take first value)
    if isinstance(width_str, (list, tuple, np.ndarray)):
        if len(width_str) == 0:
            return np.nan
        width_str = width_str[0]

    # Check for pandas NA
    try:
        if pd.isna(width_str):
            return np.nan
    except (TypeError, ValueError):
        pass

    s = str(width_str).lower().strip()
    if not s or s == 'nan':
        return np.nan

    # Extract number and optional unit
    m = re.search(r'(\d+\.?\d*)\s*(m|meter|metres?|ft|feet|foot)?', s)
    if m:
        num = float(m.group(1))
        unit = m.group(2) if m.group(2) else 'ft'  # default to feet

        # Convert meters to feet
        if unit and unit.startswith('m'):
            return num * 3.28084
        else:
            return num

    # Try to parse as plain number (assume feet)
    try:
        return float(s)
    except:
        return np.nan


def parse_lanes(lanes_str):
    """
    Parse OSM lanes tag - can be like "2", "4", or "2;2" (forward;backward)
    Returns total number of lanes
    Handles arrays/lists by taking the first value
    """
    # Handle None/NaN
    if lanes_str is None:
        return np.nan

    # Handle arrays/lists (take first value)
    if isinstance(lanes_str, (list, tuple, np.ndarray)):
        if len(lanes_str) == 0:
            return np.nan
        lanes_str = lanes_str[0]

    # Check for pandas NA
    try:
        if pd.isna(lanes_str):
            return np.nan
    except (TypeError, ValueError):
        pass

    s = str(lanes_str).strip()
    if not s or s == 'nan':
        return np.nan

    # Handle "2;2" format (directional lanes)
    if ';' in s:
        parts = s.split(';')
        try:
            return sum(int(p) for p in parts)
        except:
            return np.nan

    # Simple number
    try:
        return int(float(s))  # float first to handle "2.0"
    except:
        return np.nan

def prepare_osm_edges_with_mph(G):
    """
    Enhanced version that extracts speed, lane width, and number of lanes
    """
    # Get OSMnx defaults
    G = ox.add_edge_speeds(G)
    e = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)

    # Normalize highway
    e["highway_norm"] = e["highway"].apply(norm_highway) if "highway" in e.columns else ""

    # Extract speed (existing logic)
    if "maxspeed" in e.columns:
        e["maxspeed_mph_raw"] = e["maxspeed"].apply(_parse_maxspeed_to_mph)
    else:
        e["maxspeed_mph_raw"] = pd.Series(np.nan, index=e.index, dtype="float64")

    if "speed_kph" not in e.columns:
        G2 = ox.add_edge_speeds(G)
        e2 = ox.graph_to_gdfs(G2, nodes=False, fill_edge_geometry=True)
        e["speed_kph"] = e2["speed_kph"]

    e["speed_mph_fallback"] = e["speed_kph"] * 0.621371
    e["speed_mph"] = e["maxspeed_mph_raw"].where(~e["maxspeed_mph_raw"].isna(), e["speed_mph_fallback"])
    e["has_maxspeed"] = e["maxspeed_mph_raw"].notna()

    # Extract lane width
    if "width" in e.columns:
        e["lane_width_ft"] = e["width"].apply(parse_width)
    else:
        e["lane_width_ft"] = np.nan

    # Extract number of lanes
    if "lanes" in e.columns:
        e["num_lanes"] = e["lanes"].apply(parse_lanes)
    else:
        e["num_lanes"] = np.nan

    # Keep only drivable roads
    ROAD_WHITELIST = {
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
        "residential", "unclassified", "living_street", "service"
    }
    e = e[e["highway_norm"].isin(ROAD_WHITELIST)].copy()

    # Return the edges DataFrame!
    return e

def two_pass_nearest(segments_m: gpd.GeoDataFrame, edges_m: gpd.GeoDataFrame, max_dist_m=20):
    """
    Two-pass nearest join with lane width and lanes data
    """
    # Create a copy to avoid modifying original
    result = segments_m.copy()

    # Initialize columns
    result["speed_mph"] = np.nan
    result["highway_norm"] = ""
    result["has_maxspeed"] = False
    result["__dist_speed"] = np.nan
    result["lane_width_ft"] = np.nan
    result["num_lanes"] = np.nan

    # Columns to extract from OSM
    osm_cols = ["geometry", "speed_mph", "highway_norm", "has_maxspeed",
                "lane_width_ft", "num_lanes"]

    # Pass 1: Try to match with edges that have explicit maxspeed
    e1 = edges_m[edges_m["has_maxspeed"]].copy()

    if len(e1) > 0:
        match1 = gpd.sjoin_nearest(
            segments_m,
            e1[osm_cols],
            how="left",
            max_distance=max_dist_m,
            distance_col="__dist_speed"
        )

        # Remove duplicates - keep first (nearest) match only
        match1 = match1[~match1.index.duplicated(keep='first')]

        # Copy matched values to result
        matched_idx = match1[match1["speed_mph"].notna()].index
        for col in ["speed_mph", "highway_norm", "has_maxspeed", "__dist_speed",
                    "lane_width_ft", "num_lanes"]:
            result.loc[matched_idx, col] = match1.loc[matched_idx, col]

    # Pass 2: Fill remaining NaN values with any edge (OSMnx defaults)
    unmatched_idx = result[result["speed_mph"].isna()].index

    if len(unmatched_idx) > 0:
        e2 = edges_m[osm_cols].copy()

        match2 = gpd.sjoin_nearest(
            segments_m.loc[unmatched_idx],
            e2,
            how="left",
            max_distance=max_dist_m,
            distance_col="__dist_speed"
        )

        # Remove duplicates - keep first (nearest) match only
        match2 = match2[~match2.index.duplicated(keep='first')]

        # Copy matched values to result
        matched_idx2 = match2[match2["speed_mph"].notna()].index
        for col in ["speed_mph", "highway_norm", "has_maxspeed", "__dist_speed",
                    "lane_width_ft", "num_lanes"]:
            result.loc[matched_idx2, col] = match2.loc[matched_idx2, col]

    return result

def compute_lts_enhanced(row):
    """
    Enhanced LTS calculation that considers lane width and number of lanes
    """
    fac = row.get("__fac_cat", "unknown")
    speed = row.get("speed_mph", np.nan)
    aadt = row.get("aadt", np.nan)
    centerline = bool(row.get("__centerline", False))
    lane_width = row.get("lane_width_ft", np.nan)
    num_lanes = row.get("num_lanes", np.nan)

    speed_known = not np.isnan(speed)
    aadt_known = not np.isnan(aadt)
    width_known = not np.isnan(lane_width)
    lanes_known = not np.isnan(num_lanes)

    # Clear low-stress categories (protected infrastructure)
    if fac in ["protected", "path"]:
        return 1

    if fac == "boulevard" and speed_known and speed <= 25:
        if not aadt_known or aadt <= 1500:
            return 1

    # Enhanced logic for bike lanes
    if fac == "buffered_lane":
        base_lts = 2

        # Speed penalties
        if speed_known:
            if speed > 35:
                base_lts = 4
            elif speed > 30:
                base_lts = 3

        # Traffic penalties
        if aadt_known:
            if aadt > 12000:
                base_lts = 4
            elif aadt > 8000:
                base_lts = max(base_lts, 3)

        # Lane width consideration
        # Narrow lanes (<11 ft) increase stress
        if width_known and lane_width < 11:
            base_lts = min(4, base_lts + 1)

        # Multi-lane penalty (more lanes = more stress)
        if lanes_known and num_lanes >= 4:
            base_lts = min(4, base_lts + 1)

        return base_lts

    if fac == "lane":
        base_lts = 2

        # Speed penalties
        if speed_known:
            if speed > 35:
                base_lts = 4
            elif speed > 30:
                base_lts = 3

        # Traffic penalties
        if aadt_known:
            if aadt > 12000:
                base_lts = 4
            elif aadt > 6000:
                base_lts = max(base_lts, 3)

        # Lane width consideration
        # Very narrow lanes (<10 ft) are dangerous
        if width_known:
            if lane_width < 10:
                base_lts = min(4, base_lts + 2)  # Severe penalty
            elif lane_width < 11:
                base_lts = min(4, base_lts + 1)  # Moderate penalty

        # Multi-lane penalty
        if lanes_known and num_lanes >= 5:
            base_lts = 4  # 5+ lanes is very stressful
        elif lanes_known and num_lanes >= 4:
            base_lts = min(4, base_lts + 1)

        return base_lts

    if fac in ["mixed", "unknown"]:
        # Shared roadway - most affected by width/lanes
        base_lts = 3

        # Low speed + low traffic can be LTS 2
        if speed_known and speed <= 25 and aadt_known and aadt <= 1500 and not centerline:
            base_lts = 2
        elif speed_known and speed <= 30 and (not aadt_known or aadt <= 3000):
            base_lts = 3
        else:
            base_lts = 4

        # Wide lanes on slow streets can be better
        if width_known and lane_width >= 14 and speed_known and speed <= 25:
            base_lts = max(1, base_lts - 1)  # Wide lanes = more room to share

        # Multiple lanes without bike infrastructure is very stressful
        if lanes_known and num_lanes >= 4:
            base_lts = 4

        return base_lts

    return 4

# Load bike segments
gdf = read_dataframe(GEOJSON_IN)
gdf = ensure_4326(gdf)

# Drop columns with unuseful information
columns_to_drop = ['From_', 'To_', 'Status', 'Status_Notes', 'YearComplete','ProjectCost','FundSource','Maintenance'
                  , 'Greenway', 'Notes','Notes_Temp','created_by','created_date','last_edited_by','last_edited_date']
gdf = gdf.drop(columns_to_drop, axis=1)

# Guess optional columns
facility_col   = pick_col(gdf.columns, FACILITY_CANDIDATES)
centerline_col = pick_col(gdf.columns, CENTERLINE_CANDIDATES)
name_col       = pick_col(gdf.columns, NAME_CANDIDATES)

gdf["__fac_cat"] = gdf[facility_col].map(normalize_fac) if facility_col else "unknown"
gdf["__centerline"] = gdf[centerline_col].astype(str).str.lower().isin(["yes","true","1"]) if centerline_col else False

# Load Lexington street data with speed limits
print(f"Loading Lexington street data...")
lex_streets = read_dataframe(LEX_STREET_DATA_PATH)
lex_streets = ensure_4326(lex_streets)

print(f"  Total street segments: {len(lex_streets)}")

# Prepare Lexington street data
lex_streets['speed_mph'] = pd.to_numeric(lex_streets['SPEED'], errors='coerce')
lex_streets['road_name'] = lex_streets['ROADNAME'].fillna('')
lex_streets['road_class'] = lex_streets['RDCLASS']

# Initialize columns that won't be populated from street data
lex_streets['lane_width_ft'] = np.nan
lex_streets['num_lanes'] = np.nan

# Normalize road names for matching
def normalize_road_name(name):
    """Normalize road names for better matching"""
    if pd.isna(name) or name == '':
        return ''
    name = str(name).upper().strip()
    # Remove common abbreviations differences
    name = name.replace('STREET', 'ST').replace('ROAD', 'RD').replace('AVENUE', 'AVE')
    name = name.replace('DRIVE', 'DR').replace('BOULEVARD', 'BLVD').replace('PARKWAY', 'PKWY')
    name = name.replace('LANE', 'LN').replace('COURT', 'CT').replace('PLACE', 'PL')
    return name

# Extract and normalize bike segment road names from Name_Network
gdf['__bike_road_name'] = gdf['Name_Network'].apply(normalize_road_name) if 'Name_Network' in gdf.columns else ''
lex_streets['__street_road_name'] = lex_streets['ROADNAME'].apply(normalize_road_name)

# Initialize result dataframe
joined_speed = gdf.copy()
joined_speed['speed_mph'] = np.nan
joined_speed['lane_width_ft'] = np.nan
joined_speed['num_lanes'] = np.nan
joined_speed['__dist_speed'] = np.nan
joined_speed['__match_method'] = ''

# Strategy 1: Direct road name match
print("  Step 1: Matching by road name...")
name_matches = 0
for idx in joined_speed.index:
    bike_name = joined_speed.loc[idx, '__bike_road_name']
    if bike_name and bike_name != '':
        # Find street segments with matching name
        matching_streets = lex_streets[lex_streets['__street_road_name'] == bike_name]
        if len(matching_streets) > 0:
            # Take the median speed for this road
            median_speed = matching_streets['speed_mph'].median()
            if not pd.isna(median_speed):
                joined_speed.loc[idx, 'speed_mph'] = median_speed
                joined_speed.loc[idx, '__match_method'] = 'name'
                name_matches += 1

print(f"    Matched {name_matches} segments by road name")

# Strategy 2: Spatial matching for remaining segments
unmatched_idx = joined_speed[joined_speed['speed_mph'].isna()].index
if len(unmatched_idx) > 0:
    print(f"  Step 2: Spatial matching for {len(unmatched_idx)} remaining segments...")

    # Project to local CRS for accurate distance matching
    gdf_unmatched = joined_speed.loc[unmatched_idx].copy()
    gdf_unmatched_m = project_to_local(gdf_unmatched)
    lex_streets_m = project_to_local(lex_streets[['geometry', 'speed_mph', 'road_name', 'lane_width_ft', 'num_lanes', 'road_class']])

    # Calculate bearings for both datasets (in WGS84 for better accuracy)
    gdf_unmatched['__bearing'] = gdf_unmatched.geometry.apply(get_bearing)
    lex_streets['__bearing'] = lex_streets.geometry.apply(get_bearing)

    # Spatial join
    spatial_match = gpd.sjoin_nearest(
        gdf_unmatched_m,
        lex_streets_m,
        how="left",
        max_distance=NEAREST_MAX_DIST_M,
        distance_col="__dist_speed"
    )

    # Get the index of the matched street segments
    if 'index_right' in spatial_match.columns:
        # Merge the speed and other data from the matched street segments
        spatial_match = spatial_match.merge(
            lex_streets[['speed_mph', '__bearing']].rename(columns={'__bearing': '__bearing_street'}),
            left_on='index_right',
            right_index=True,
            how='left',
            suffixes=('', '_matched')
        )

    # Merge bearing information from bike segments
    spatial_match = spatial_match.merge(
        gdf_unmatched[['__bearing']],
        left_index=True,
        right_index=True,
        how='left',
        suffixes=('', '_bike')
    )

    # Calculate bearing difference for each match
    spatial_match['__bearing_diff'] = spatial_match.apply(
        lambda row: bearing_difference(row.get('__bearing'), row.get('__bearing_street')),
        axis=1
    )

    # Filter out perpendicular matches
    BEARING_THRESHOLD_PERPENDICULAR = 80  # degrees
    before_filter = len(spatial_match[spatial_match['speed_mph'].notna()]) if 'speed_mph' in spatial_match.columns else 0
    if before_filter > 0:
        perpendicular_matches = spatial_match['__bearing_diff'] > BEARING_THRESHOLD_PERPENDICULAR
        spatial_match.loc[perpendicular_matches, 'speed_mph'] = np.nan
        after_filter = len(spatial_match[spatial_match['speed_mph'].notna()])
        filtered_count = before_filter - after_filter
        if filtered_count > 0:
            print(f"    Filtered out {filtered_count} perpendicular matches (>{BEARING_THRESHOLD_PERPENDICULAR}°)")

    # Remove duplicates - keep first (nearest) match only
    spatial_match = spatial_match[~spatial_match.index.duplicated(keep='first')]

    # Copy matched values back to joined_speed
    matched_spatial = spatial_match[spatial_match['speed_mph'].notna()].index
    for col in ['speed_mph', 'lane_width_ft', 'num_lanes', '__dist_speed']:
        if col in spatial_match.columns:
            joined_speed.loc[matched_spatial, col] = spatial_match.loc[matched_spatial, col]
    joined_speed.loc[matched_spatial, '__match_method'] = 'spatial'

    print(f"    Matched {len(matched_spatial)} segments spatially")

# Diagnostic: Show matching distance statistics
if "__dist_speed" in joined_speed.columns:
    distances = joined_speed["__dist_speed"].dropna()
    if len(distances) > 0:
        print(f"  Matching distance stats:")
        print(f"    Min: {distances.min():.1f}m, Max: {distances.max():.1f}m")
        print(f"    Mean: {distances.mean():.1f}m, Median: {distances.median():.1f}m")
        far_matches = (distances > 5).sum()
        if far_matches > 0:
            print(f"    Warning: {far_matches} segments matched >5m away (may be inaccurate)")

# Round speeds to common values
joined_speed["speed_mph"] = joined_speed["speed_mph"].apply(round_to_common_mph)

print(f"  Matched {joined_speed['speed_mph'].notna().sum()} / {len(joined_speed)} segments to speeds")
print(f"  Speed distribution:\n{joined_speed['speed_mph'].value_counts().sort_index()}")

# Optional: AADT enrichment
# Always ensure aadt column exists
joined_speed['aadt'] = np.nan

if AADT_PATH and AADT_PATH.exists():
    print(f"Loading AADT data from {AADT_PATH}...")

    # Determine file type and load appropriately
    if str(AADT_PATH).endswith('.csv'):
        aadt_df = pd.read_csv(AADT_PATH)
        aadt = gpd.GeoDataFrame(
            aadt_df,
            geometry=gpd.points_from_xy(aadt_df[CSV_LON], aadt_df[CSV_LAT]),
            crs=4326
        )
    else:
        aadt = read_dataframe(AADT_PATH)
        aadt = ensure_4326(aadt)

    # Find AADT column (common names)
    aadt_col = None
    for candidate in ['AADT', 'aadt', 'ADT', 'traffic', 'volume', 'count']:
        if candidate in aadt.columns:
            aadt_col = candidate
            break

    if aadt_col:
        print(f"  Using AADT column: {aadt_col}")

        # Sanitize AADT values
        aadt['aadt_clean'] = sanitize_aadt(aadt[aadt_col])
        aadt = aadt[aadt['aadt_clean'].notna()].copy()

        print(f"  Valid AADT records: {len(aadt)}")
        print(f"  AADT range: {aadt['aadt_clean'].min():.0f} - {aadt['aadt_clean'].max():.0f}")

        # Project for matching
        aadt_m = project_to_local(aadt[['geometry', 'aadt_clean']])
        joined_speed_m = project_to_local(joined_speed)

        # Spatial join
        print(f"  Matching to AADT (max distance: {AADT_MAX_DIST_M}m)...")
        temp_join = gpd.sjoin_nearest(
            joined_speed_m,
            aadt_m[['geometry', 'aadt_clean']],
            how='left',
            max_distance=AADT_MAX_DIST_M,
            distance_col='__dist_aadt'
        )

        # Remove duplicates and update aadt column
        temp_join = temp_join[~temp_join.index.duplicated(keep='first')]
        joined_speed['aadt'] = temp_join['aadt_clean']

        print(f"  Matched {joined_speed['aadt'].notna().sum()} segments to AADT")
    else:
        print("  Warning: Could not find AADT column in data")
else:
    print("No AADT data provided, skipping...")

print("\nData enrichment complete!")
print(f"Final dataset: {len(joined_speed)} segments")
print(f"  With speed: {joined_speed['speed_mph'].notna().sum()}")
print(f"  With AADT: {joined_speed['aadt'].notna().sum()}")

# AADT Code is still not working (needs to be fixed)

# Create base map 
m = folium.Map(
    location=[38.0406, -84.5037],
    zoom_start=12
)

# controls 
LocateControl(auto_start=False, flyTo=True).add_to(m)
MeasureControl(primary_length_unit="meters").add_to(m)
Fullscreen().add_to(m)
MousePosition().add_to(m)

# Compute LTS (enhanced with lane width and lanes)
print("\nComputing Level of Traffic Stress...")
joined_speed["LTS"] = joined_speed.apply(compute_lts_enhanced, axis=1).astype(int)

# Optional: Compare old vs new LTS to see impact of lane width/lanes
if False:  # Set to True to see comparison
    joined_speed["LTS_original"] = joined_speed.apply(compute_lts, axis=1).astype(int)
    diff = joined_speed[joined_speed["LTS_original"] != joined_speed["LTS"]]
    print(f"\nLTS changes with enhanced calculation: {len(diff)} segments")
    if len(diff) > 0:
        print(diff[["Name_Facility", "LTS_original", "LTS", "lane_width_ft", "num_lanes", "speed_mph"]].head(10))

# -----------------------------
# 6) Display data coverage statistics
# -----------------------------
print("\n" + "="*60)
print("DATA COVERAGE SUMMARY")
print("="*60)
print(f"Total segments: {len(joined_speed)}")
print(f"\nSegments with speed data: {joined_speed['speed_mph'].notna().sum()} ({joined_speed['speed_mph'].notna().sum()/len(joined_speed)*100:.1f} %)")

if "aadt" in joined_speed.columns:
    aadt_count = joined_speed['aadt'].notna().sum()
    print(f"Segments with AADT data: {aadt_count} ({aadt_count/len(joined_speed)*100:.1f} %)")
else:
    print(f"AADT data: Not included")

if "lane_width_ft" in joined_speed.columns:
    width_coverage = joined_speed["lane_width_ft"].notna().sum()
    print(f"Segments with lane width data: {width_coverage} ({width_coverage/len(joined_speed)*100:.1f} %)")

    if width_coverage > 0:
        print(f"  Width range: {joined_speed['lane_width_ft'].min():.1f} - {joined_speed['lane_width_ft'].max():.1f} ft")
        print(f"  Mean width: {joined_speed['lane_width_ft'].mean():.1f} ft")
        print(f"  Median width: {joined_speed['lane_width_ft'].median():.1f} ft")

if "num_lanes" in joined_speed.columns:
    lanes_coverage = joined_speed["num_lanes"].notna().sum()
    print(f"Segments with lane count data: {lanes_coverage} ({lanes_coverage/len(joined_speed)*100:.1f} %)")

    if lanes_coverage > 0:
        print(f"  Lane distribution:")
        for num in sorted(joined_speed["num_lanes"].dropna().unique()):
            count = (joined_speed["num_lanes"] == num).sum()
            print(f"    {int(num)} lanes: {count} segments")

print("\n" + "="*60)
print("LTS DISTRIBUTION")
print("="*60)
for lts in [1, 2, 3, 4]:
    count = (joined_speed['LTS'] == lts).sum()
    miles = joined_speed[joined_speed['LTS'] == lts]['Length_Miles'].sum()
    pct = count / len(joined_speed) * 100
    print(f"LTS {lts}: {count:4d} segments ({pct:5.1f}%) - {miles:.2f} miles")
print("="*60)

# Save enriched data
write_dataframe(joined_speed, OUT_ENRICH, driver="GeoJSON")
print(f"\nSaved enriched GeoJSON -> {OUT_ENRICH}")

#  Create interactive map
print(f"Creating interactive map...")

# Simplify geometries for better map performance
print("  Simplifying geometries for web performance...")
# Simplify to ~1 meter tolerance (improves rendering speed)
joined_speed_simplified = joined_speed.copy()
joined_speed_simplified['geometry'] = joined_speed_simplified['geometry'].simplify(tolerance=0.00001, preserve_topology=True)

# Create base map with simple, reliable tiles
m = folium.Map(
    location=[38.0406, -84.5037],  # Use list instead of tuple
    zoom_start=12,
    prefer_canvas=True  # Use canvas rendering (faster than SVG for many features)
)

# Add controls
LocateControl(auto_start=False, flyTo=True).add_to(m)
MeasureControl(primary_length_unit="meters").add_to(m)
Fullscreen().add_to(m)
MousePosition().add_to(m)

# Define LTS colors with more vibrant, high-contrast colors (for map lines)
LTS_COLORS = {
    1: "#00D084",  # Bright green
    2: "#3B82F6",  # Bright blue
    3: "#F59E0B",  # Bright amber/orange
    4: "#EF4444"   # Bright red
}
LTS_NAMES = {
    1: "LTS 1 (Very Low Stress - Suitable for children)",
    2: "LTS 2 (Low Stress - Most adults comfortable)",
    3: "LTS 3 (Moderate Stress - Confident cyclists)",
    4: "LTS 4 (High Stress - Strong & fearless only)"
}

# Define Facility Colors (arbitrary choice for distinctness) - these are NOT used for line color anymore
# but are kept here if you ever want to switch back to facility coloring.
FACILITY_COLORS = {
    "protected": "#008000",  # Dark Green
    "path": "#FFD700",       # Gold
    "buffered_lane": "#4169E1", # Royal Blue
    "lane": "#FF4500",       # Orange Red
    "boulevard": "#8A2BE2",  # Blue Violet
    "mixed": "#808080",      # Gray
    "unknown": "#000000"     # Black
}


def style_by_lts(feature):
    # Get LTS level from feature properties, default to 4 if not found
    lts = feature["properties"].get("LTS", 4)
    return {
        "color": LTS_COLORS.get(int(lts), "#EF4444"), # Use LTS_COLORS for line color
        "weight": 5,           # Increased from 4 to 5 (thicker lines)
        "opacity": 1.0,        # Increased from 0.8 to 1.0 (fully opaque)
        "lineJoin": "round",   # Smoother line joins
        "lineCap": "round"     # Rounded line ends
    }

# Build tooltip fields list - only include columns that exist and have data
tooltip_fields = []
tooltip_aliases = []

if facility_col and facility_col in joined_speed.columns:
    tooltip_fields.append(facility_col)
    tooltip_aliases.append("Facility Type")

# Add data fields that exist and have values
if "speed_mph" in joined_speed.columns and joined_speed["speed_mph"].notna().any():
    tooltip_fields.append("speed_mph")
    tooltip_aliases.append("Speed (mph)")

if "lane_width_ft" in joined_speed.columns and joined_speed["lane_width_ft"].notna().any():
    tooltip_fields.append("lane_width_ft")
    tooltip_aliases.append("Lane Width (ft)")

if "num_lanes" in joined_speed.columns and joined_speed["num_lanes"].notna().any():
    tooltip_fields.append("num_lanes")
    tooltip_aliases.append("# Lanes")

if "aadt" in joined_speed.columns and joined_speed["aadt"].notna().any():
    tooltip_fields.append("aadt")
    tooltip_aliases.append("AADT")

# Always include LTS
tooltip_fields.append("LTS")
tooltip_aliases.append("LTS Level")

# Optimize rendering by grouping by LTS instead of facility
# This reduces the number of layers and improves performance
for lts_level in [1, 2, 3, 4]:
    subset = joined_speed_simplified[joined_speed_simplified['LTS'] == lts_level]

    if not subset.empty:
        folium.GeoJson(
            subset.to_json(),
            name=LTS_NAMES[lts_level],
            style_function=lambda feature, lts=lts_level: {
                "color": LTS_COLORS[lts],
                "weight": 4,
                "opacity": 0.9,
                "lineJoin": "round",
                "lineCap": "round"
            },
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=True,
                sticky=False  # Tooltip doesn't follow cursor (improves performance)
            ),
            highlight_function=lambda x: {"weight": 6, "opacity": 1.0},
            overlay=True,
            show=True,
            # Performance optimization: simplify tolerance
            smooth_factor=1.0  # Reduces points in lines for better performance
        ).add_to(m)

# Add a LayerControl to toggle LTS layers
folium.LayerControl(collapsed=False).add_to(m)

# LTS Legend HTML (explains the colors on the map)
lts_legend_html = """
<div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
            background: white; padding: 15px; border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3); font-size: 13px;
            max-width: 280px; font-family: Arial, sans-serif;">
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

<div style="margin-top: 12px; padding-top: 8px; border-top: 1px solid #ddd;
            font-size: 11px; color: #666;">
Toggle facility types using the layer control.
</div>
</div>
"""
m.get_root().html.add_child(folium.Element(lts_legend_html))


# Fit map to data bounds
if not joined_speed_simplified.empty:
    minx, miny, maxx, maxy = joined_speed_simplified.total_bounds
    m.fit_bounds([[miny, minx], [maxy, maxx]])

# Save map
m.save(str(OUT_MAP))
print(f"Saved map -> {OUT_MAP}")

print("\n✓ Analysis complete!")

"""Current problems:
1. Speeds on certain roads are still not rounding correctly.
2. AADT can't be right. They are giving incredibly large numbers such as 3.4e74
3. Graphics are not loading very quickly.

The Following Cells are sandbox data analysis. I'm trying to compare the data provide by LFUCG to that from OpenStreetMaps.
"""