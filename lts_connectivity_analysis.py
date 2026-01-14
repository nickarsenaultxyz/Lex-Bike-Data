# -*- coding: utf-8 -*-
"""lts_connectivity_analysis.py

LTS Connectivity Analysis based on Furth & Mekuria (2013)
"Network Connectivity for Low-Stress Bicycling"

This script:
1. Builds a network graph from LTS-rated street segments
2. Filters to low-stress links (LTS <= threshold)
3. Finds connected components ("islands" of low-stress cycling)
4. Visualizes each cluster in a different color
5. Calculates connectivity metrics

Key concepts from the paper:
- Connectivity clusters: Groups of streets reachable without using high-stress links
- Barriers: High-stress roads that separate low-stress clusters
- The goal is to identify where infrastructure improvements could connect islands
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from pyogrio import read_dataframe, write_dataframe
gpd.options.io_engine = "pyogrio"
import networkx as nx
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union
import folium
from folium.plugins import LocateControl, MeasureControl, Fullscreen, MousePosition
import colorsys
import random

# Configuration
INPUT_NETWORK = Path("lexbike_with_residential.geojson")
OUT_CLUSTERS = Path("lexbike_connectivity_clusters.geojson")
OUT_MAP = Path("lexbike_connectivity_map.html")

# LTS threshold for "low stress" connectivity analysis
# LTS 2 = "Most adults comfortable" (the mainstream population per Furth & Mekuria)
LTS_THRESHOLD = 2

# Tolerance for snapping endpoints together (in meters)
SNAP_TOLERANCE = 15  # meters

# Minimum cluster size to display (filter out tiny isolated segments)
MIN_CLUSTER_SEGMENTS = 3


def ensure_4326(gdf):
    """Ensure GeoDataFrame is in WGS84"""
    return gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)


def project_to_local(gdf):
    """Project to local UTM for accurate distance calculations"""
    return gdf.to_crs(32616)  # UTM 16N for Kentucky


def get_endpoints(geom):
    """Extract start and end points from a LineString or MultiLineString"""
    if geom is None or geom.is_empty:
        return None, None

    if geom.geom_type == 'MultiLineString':
        # For MultiLineString, get endpoints of the full extent
        coords = []
        for line in geom.geoms:
            coords.extend(list(line.coords))
        if len(coords) < 2:
            return None, None
        return Point(coords[0]), Point(coords[-1])
    elif geom.geom_type == 'LineString':
        coords = list(geom.coords)
        if len(coords) < 2:
            return None, None
        return Point(coords[0]), Point(coords[-1])
    else:
        return None, None


def snap_to_grid(point, tolerance):
    """Snap a point to a grid to group nearby endpoints"""
    if point is None:
        return None
    return (round(point.x / tolerance) * tolerance,
            round(point.y / tolerance) * tolerance)


def build_network_graph(gdf, snap_tolerance=SNAP_TOLERANCE):
    """
    Build a NetworkX graph from street segments.

    Each street segment becomes an edge connecting its two endpoints.
    Endpoints within snap_tolerance are considered the same node.
    """
    print(f"  Building network graph from {len(gdf)} segments...")

    # Project to local CRS for accurate distance calculations
    gdf_m = project_to_local(gdf)

    # Create graph
    G = nx.Graph()

    # Track segment-to-edge mapping
    segment_edges = {}

    for idx, row in gdf_m.iterrows():
        geom = row.geometry
        start_pt, end_pt = get_endpoints(geom)

        if start_pt is None or end_pt is None:
            continue

        # Snap endpoints to grid
        start_node = snap_to_grid(start_pt, snap_tolerance)
        end_node = snap_to_grid(end_pt, snap_tolerance)

        if start_node is None or end_node is None:
            continue

        # Skip self-loops (very short segments that snap to same point)
        if start_node == end_node:
            continue

        # Add edge with segment index as attribute
        if G.has_edge(start_node, end_node):
            # Edge already exists, append segment index
            G[start_node][end_node]['segments'].append(idx)
        else:
            G.add_edge(start_node, end_node, segments=[idx])

        segment_edges[idx] = (start_node, end_node)

    print(f"    Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    return G, segment_edges


def find_connectivity_clusters(G):
    """
    Find connected components in the graph.
    Returns a list of sets, each containing node tuples.
    """
    components = list(nx.connected_components(G))
    print(f"    Found {len(components)} connected components")
    return components


def assign_cluster_ids(gdf, G, segment_edges, components):
    """
    Assign cluster IDs to each segment based on connected components.
    """
    # Create node-to-cluster mapping
    node_to_cluster = {}
    for cluster_id, nodes in enumerate(components):
        for node in nodes:
            node_to_cluster[node] = cluster_id

    # Assign cluster IDs to segments
    cluster_ids = []
    for idx in gdf.index:
        if idx in segment_edges:
            start_node, end_node = segment_edges[idx]
            cluster_id = node_to_cluster.get(start_node, -1)
            cluster_ids.append(cluster_id)
        else:
            cluster_ids.append(-1)  # Not in graph

    return cluster_ids


def generate_distinct_colors(n):
    """Generate n visually distinct colors using HSV color space"""
    colors = []
    for i in range(n):
        # Use golden ratio to spread hues evenly
        hue = (i * 0.618033988749895) % 1.0
        # High saturation and value for visibility
        saturation = 0.7 + (i % 3) * 0.1  # Vary slightly
        value = 0.8 + (i % 2) * 0.1

        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)

    return colors


def calculate_segment_lengths(gdf):
    """
    Calculate length in miles for each segment from geometry.
    """
    gdf_m = project_to_local(gdf)
    # Length in meters, convert to miles
    return gdf_m.geometry.length / 1609.34


def calculate_cluster_metrics(gdf, cluster_col='cluster_id'):
    """
    Calculate metrics for each cluster.
    """
    # First ensure we have length calculated for all segments
    if 'length_miles_calc' not in gdf.columns:
        gdf = gdf.copy()
        gdf['length_miles_calc'] = calculate_segment_lengths(gdf)

    metrics = []

    for cluster_id in gdf[cluster_col].unique():
        if cluster_id < 0:
            continue

        cluster_segments = gdf[gdf[cluster_col] == cluster_id]

        # Use calculated length
        total_miles = cluster_segments['length_miles_calc'].sum()

        # Calculate bounding box area
        bounds = cluster_segments.total_bounds  # minx, miny, maxx, maxy
        bbox_area_sq_miles = ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) *
                              (69.0 * 69.0))  # Rough conversion for lat/lon

        # Get centroid for cluster location
        centroid = cluster_segments.union_all().centroid

        metrics.append({
            'cluster_id': cluster_id,
            'num_segments': len(cluster_segments),
            'total_miles': total_miles,
            'bbox_area_sq_miles': bbox_area_sq_miles,
            'centroid_lat': centroid.y,
            'centroid_lon': centroid.x
        })

    return pd.DataFrame(metrics)


def identify_barriers(gdf_low_stress, gdf_high_stress, buffer_distance=50):
    """
    Identify high-stress segments that act as barriers between low-stress clusters.
    These are segments where improving them could connect clusters.
    """
    print("  Identifying barrier segments...")

    # Project for accurate distance
    low_m = project_to_local(gdf_low_stress)
    high_m = project_to_local(gdf_high_stress)

    # Buffer low-stress network
    low_buffer = low_m.geometry.buffer(buffer_distance)
    low_union = unary_union(low_buffer)

    # Find high-stress segments that touch the low-stress network
    high_m['touches_low_stress'] = high_m.geometry.intersects(low_union)

    barriers = gdf_high_stress[high_m['touches_low_stress']].copy()
    print(f"    Found {len(barriers)} potential barrier segments")

    return barriers


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

print("=" * 60)
print("LTS CONNECTIVITY ANALYSIS")
print("Based on Furth & Mekuria (2013)")
print("=" * 60)

# Load network data
print(f"\nLoading network from {INPUT_NETWORK}...")
network = read_dataframe(INPUT_NETWORK)
network = ensure_4326(network)
print(f"  Loaded {len(network)} total segments")

# Show LTS distribution
print(f"\n  LTS Distribution:")
for lts in sorted(network['LTS'].unique()):
    count = (network['LTS'] == lts).sum()
    pct = count / len(network) * 100
    print(f"    LTS {lts}: {count:,} segments ({pct:.1f}%)")

# =============================================================================
# CONNECTIVITY ANALYSIS AT LTS THRESHOLD
# =============================================================================

print(f"\n" + "=" * 60)
print(f"CONNECTIVITY ANALYSIS AT LTS ≤ {LTS_THRESHOLD}")
print("=" * 60)

# Filter to low-stress segments
low_stress = network[network['LTS'] <= LTS_THRESHOLD].copy()
high_stress = network[network['LTS'] > LTS_THRESHOLD].copy()

print(f"\n  Low-stress segments (LTS ≤ {LTS_THRESHOLD}): {len(low_stress):,}")
print(f"  High-stress segments (LTS > {LTS_THRESHOLD}): {len(high_stress):,}")

# Build network graph
G, segment_edges = build_network_graph(low_stress)

# Find connected components
components = find_connectivity_clusters(G)

# Sort components by size (largest first)
components = sorted(components, key=len, reverse=True)

# Assign cluster IDs
cluster_ids = assign_cluster_ids(low_stress, G, segment_edges, components)
low_stress['cluster_id'] = cluster_ids

# Calculate lengths for all low-stress segments
print(f"\n  Calculating segment lengths...")
low_stress['length_miles_calc'] = calculate_segment_lengths(low_stress)
total_network_miles = low_stress['length_miles_calc'].sum()
print(f"    Total low-stress network: {total_network_miles:.1f} miles")

# Filter out tiny clusters
cluster_sizes = low_stress['cluster_id'].value_counts()
valid_clusters = cluster_sizes[cluster_sizes >= MIN_CLUSTER_SEGMENTS].index.tolist()
low_stress_filtered = low_stress[low_stress['cluster_id'].isin(valid_clusters)].copy()

# Renumber clusters (0 = largest, 1 = second largest, etc.)
cluster_mapping = {old: new for new, old in enumerate(valid_clusters)}
low_stress_filtered['cluster_id'] = low_stress_filtered['cluster_id'].map(cluster_mapping)

print(f"\n  Connectivity Clusters (min {MIN_CLUSTER_SEGMENTS} segments):")
print(f"    Total clusters: {len(valid_clusters)}")

# Calculate cluster metrics
metrics = calculate_cluster_metrics(low_stress_filtered)
metrics = metrics.sort_values('num_segments', ascending=False)

print(f"\n  Top 10 Clusters by Size:")
print(f"    {'Cluster':<10} {'Segments':<12} {'Miles':<10}")
print(f"    {'-'*32}")
for _, row in metrics.head(10).iterrows():
    print(f"    {int(row['cluster_id']):<10} {int(row['num_segments']):<12} {row['total_miles']:.2f}")

# Summary statistics
total_low_stress_miles = metrics['total_miles'].sum()
largest_cluster_miles = metrics.iloc[0]['total_miles'] if len(metrics) > 0 else 0
largest_cluster_pct = (largest_cluster_miles / total_low_stress_miles * 100) if total_low_stress_miles > 0 else 0

print(f"\n  Summary:")
print(f"    Total low-stress miles: {total_low_stress_miles:.2f}")
print(f"    Largest cluster: {largest_cluster_miles:.2f} miles ({largest_cluster_pct:.1f}% of network)")
print(f"    Number of islands: {len(valid_clusters)}")

# Identify potential barriers
barriers = identify_barriers(low_stress_filtered, high_stress)

# =============================================================================
# SAVE RESULTS
# =============================================================================

print(f"\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Save cluster data
print(f"  Saving clusters to {OUT_CLUSTERS}...")
write_dataframe(low_stress_filtered, OUT_CLUSTERS, driver="GeoJSON")

# =============================================================================
# CREATE INTERACTIVE MAP
# =============================================================================

print(f"\n  Creating interactive map...")

# Generate colors for clusters
num_clusters = len(valid_clusters)
cluster_colors = generate_distinct_colors(num_clusters)

# Create cluster color mapping
cluster_color_map = {i: cluster_colors[i] for i in range(num_clusters)}

# Create map
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

# Simplify geometries for performance
low_stress_simplified = low_stress_filtered.copy()
low_stress_simplified['geometry'] = low_stress_simplified['geometry'].simplify(
    tolerance=0.00001, preserve_topology=True
)

# Add high-stress roads as gray background layer
print("    Adding high-stress background layer...")
high_stress_simplified = high_stress.copy()
high_stress_simplified['geometry'] = high_stress_simplified['geometry'].simplify(
    tolerance=0.00001, preserve_topology=True
)

folium.GeoJson(
    high_stress_simplified.to_json(),
    name=f"High Stress (LTS > {LTS_THRESHOLD})",
    style_function=lambda feature: {
        "color": "#D1D5DB",  # Light gray
        "weight": 2,
        "opacity": 0.5
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['LTS', 'Name_Network'] if 'Name_Network' in high_stress_simplified.columns else ['LTS'],
        aliases=['LTS', 'Street'] if 'Name_Network' in high_stress_simplified.columns else ['LTS'],
        localize=True
    ),
    overlay=True,
    show=True
).add_to(m)

# Add each cluster as a separate layer (top 20 largest)
print("    Adding cluster layers...")
top_clusters = metrics.head(20)['cluster_id'].tolist()

for cluster_id in top_clusters:
    cluster_data = low_stress_simplified[low_stress_simplified['cluster_id'] == cluster_id]

    if cluster_data.empty:
        continue

    color = cluster_color_map.get(cluster_id, "#888888")
    cluster_miles = metrics[metrics['cluster_id'] == cluster_id]['total_miles'].values[0]
    cluster_segments = len(cluster_data)

    tooltip_fields = ['cluster_id', 'LTS']
    tooltip_aliases = ['Cluster', 'LTS']

    if 'Name_Network' in cluster_data.columns:
        tooltip_fields.append('Name_Network')
        tooltip_aliases.append('Street')

    if 'Type_Facility' in cluster_data.columns:
        tooltip_fields.append('Type_Facility')
        tooltip_aliases.append('Type')

    folium.GeoJson(
        cluster_data.to_json(),
        name=f"Cluster {cluster_id} ({cluster_segments} seg, {cluster_miles:.1f} mi)",
        style_function=lambda feature, c=color: {
            "color": c,
            "weight": 4,
            "opacity": 0.9
        },
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True
        ),
        highlight_function=lambda x: {"weight": 6, "opacity": 1.0},
        overlay=True,
        show=True
    ).add_to(m)

# Add remaining clusters as "Other Clusters"
remaining_clusters = [c for c in valid_clusters if cluster_mapping.get(c, c) not in top_clusters]
if remaining_clusters:
    remaining_data = low_stress_simplified[
        low_stress_simplified['cluster_id'].isin([cluster_mapping.get(c, c) for c in remaining_clusters])
    ]

    if not remaining_data.empty:
        folium.GeoJson(
            remaining_data.to_json(),
            name=f"Other Clusters ({len(remaining_clusters)} small)",
            style_function=lambda feature: {
                "color": cluster_color_map.get(feature['properties'].get('cluster_id', 0), "#888888"),
                "weight": 3,
                "opacity": 0.7
            },
            overlay=True,
            show=False  # Hidden by default
        ).add_to(m)

# Add layer control
folium.LayerControl(collapsed=True).add_to(m)

# Add legend
legend_html = f"""
<div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
            background: white; padding: 15px; border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3); font-size: 13px;
            max-width: 320px; font-family: Arial, sans-serif;">
<div style="font-weight: bold; font-size: 15px; margin-bottom: 10px;
            border-bottom: 2px solid #333; padding-bottom: 5px;">
LTS Connectivity Analysis
</div>

<div style="margin-bottom: 10px;">
<b>LTS Threshold:</b> ≤ {LTS_THRESHOLD} (Low Stress)
</div>

<div style="margin-bottom: 10px;">
<b>Summary:</b><br>
• {len(valid_clusters)} connectivity clusters (islands)<br>
• {total_low_stress_miles:.1f} total low-stress miles<br>
• Largest cluster: {largest_cluster_miles:.1f} mi ({largest_cluster_pct:.0f}%)
</div>

<div style="margin-bottom: 10px;">
<b>Legend:</b><br>
<span style="display:inline-block; width:30px; height:4px; background:#D1D5DB; margin-right:8px;"></span>
High stress (barriers)<br>
<span style="display:inline-block; width:30px; height:4px; background:linear-gradient(to right, #FF0000, #00FF00, #0000FF); margin-right:8px;"></span>
Clusters (colored)
</div>

<div style="font-size: 11px; color: #666; border-top: 1px solid #ddd; padding-top: 8px;">
Each color = one connectivity island.<br>
Use layer control to toggle clusters.
</div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Fit to bounds
if not low_stress_filtered.empty:
    minx, miny, maxx, maxy = low_stress_filtered.total_bounds
    m.fit_bounds([[miny, minx], [maxy, maxx]])

# Save map
print(f"  Saving map to {OUT_MAP}...")
m.save(str(OUT_MAP))

# =============================================================================
# ANALYSIS SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print(f"\nKey Findings:")
print(f"  • The low-stress network (LTS ≤ {LTS_THRESHOLD}) is fragmented into {len(valid_clusters)} islands")
print(f"  • The largest island contains only {largest_cluster_pct:.1f}% of low-stress miles")
print(f"  • This means cyclists must use high-stress roads to travel between most areas")

print(f"\nPer Furth & Mekuria (2013):")
print(f"  • LTS 2 represents routes suitable for 'most adults'")
print(f"  • Poor connectivity at LTS 2 indicates the network doesn't serve the mainstream population")
print(f"  • Improvements should focus on connecting these islands")

print(f"\nOutput files:")
print(f"  • Cluster data: {OUT_CLUSTERS}")
print(f"  • Interactive map: {OUT_MAP}")

print(f"\nNext steps for improving connectivity:")
print(f"  1. Identify strategic barrier crossings (arterials, highways)")
print(f"  2. Prioritize connections between large clusters")
print(f"  3. Consider protected infrastructure on high-stress barriers")
