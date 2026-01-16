# Lexington Bikestress with Residential Streets

A Python tool for analyzing bicycle Level of Traffic Stress (LTS) across Lexington, Kentucky's complete street network. This project extends traditional bike infrastructure analysis by incorporating residential streets as viable cycling routes.

## Overview

This project computes LTS ratings for the entire Lexington street network, providing a complete picture of bikeable routes:

- **Bike Infrastructure**: Existing bike lanes, paths, and facilities rated LTS 1-4
- **Residential Streets**: Local roads (RDCLASS 5-8) added as bikeable routes where no dedicated infrastructure exists
- **Major Roads**: Arterials and collectors without bike infrastructure marked as "unbikeable" (LTS 5)

The analysis is based on the methodology from Furth & Mekuria (2013)

## LTS Ratings

| LTS | Description | Suitable For |
|-----|-------------|--------------|
| 1 | Very Low Stress | Children, all ages/abilities |
| 2 | Low Stress | Most adults comfortable |
| 3 | Moderate Stress | Confident cyclists |
| 4 | High Stress | Strong & fearless only |
| 5 | Unbikeable | Major roads without infrastructure |

## Scripts

### `bikestress_route_with_residential.py`

Main analysis script that:
1. Loads existing bike infrastructure from `lexbike.geojson`
2. Loads Lexington street data from `lex_street_data.geojson`
3. Computes LTS for bike infrastructure based on facility type, speed, and AADT
4. Classifies all streets by road type (RDCLASS)
5. Adds residential/local streets as bikeable routes (LTS 2-3)
6. Marks major roads without infrastructure as unbikeable (LTS 5)
7. Generates an interactive Folium map

**Output files:**
- `lexbike_with_residential.geojson` - Combined network with LTS ratings
- `lexbike_LTS_map_with_residential.html` - Interactive web map

### `lts_connectivity_analysis.py`

Network connectivity analysis that:
1. Builds a graph from the LTS-rated street network
2. Filters to low-stress links (LTS <= 2 by default)
3. Identifies connected components ("islands" of low-stress cycling)
4. Visualizes clusters in different colors
5. Calculates connectivity metrics

**Output files:**
- `lexbike_connectivity_clusters.geojson` - Cluster assignments
- `lexbike_connectivity_map.html` - Interactive cluster visualization

## Road Classification (RDCLASS)

| RDCLASS | Type | Treatment |
|---------|------|-----------|
| 1 | Interstate/Highway | Unbikeable (LTS 5) |
| 2 | Parkway/Expressway | Unbikeable (LTS 5) |
| 3 | Arterial | Unbikeable (LTS 5) |
| 4 | Major Collector | Unbikeable (LTS 5) |
| 5 | Minor Collector | Bikeable (LTS 2-3) |
| 6 | Local Street | Bikeable (LTS 2-3) |
| 7 | Service Road | Bikeable (LTS 2-3) |
| 8 | Alley | Bikeable (LTS 2-3) |

## Requirements

```
numpy
pandas
geopandas
pyogrio
folium
networkx
shapely
```

## Input Data

- `lexbike.geojson` - Existing bike infrastructure (bike lanes, paths, trails)
- `lex_street_data.geojson` - Complete Lexington street network with RDCLASS and speed attributes

## Usage

```bash
# Generate LTS network with residential streets
python bikestress_route_with_residential.py

# Run connectivity analysis
python lts_connectivity_analysis.py
```

## LTS Calculation Logic

### Bike Infrastructure
- **Protected/Path**: LTS 1
- **Buffered Lane**: LTS 2 (if speed <= 30 mph), else LTS 3
- **Bike Lane**: LTS 2-4 based on speed and AADT
- **Mixed/Sharrow**: LTS 2-4 based on speed and traffic

### Residential Streets
- **Local Streets (RDCLASS 6)**: LTS 2 if speed <= 25 mph, else LTS 3
- **Minor Collectors (RDCLASS 5)**: LTS 2 if speed <= 25 mph, else LTS 3
- Maximum speed limit for inclusion: 35 mph

## Connectivity Cluster Analysis

The `lts_connectivity_analysis.py` script performs network connectivity analysis based on Furth & Mekuria's methodology.

### Methodology

1. **Graph Construction**: Street segments become edges in a NetworkX graph, with endpoints snapped together within a 15m tolerance
2. **Low-Stress Filtering**: Only segments at or below the LTS threshold (default: LTS 2) are included
3. **Component Detection**: Connected components identify "islands" of continuous low-stress cycling
4. **Cluster Metrics**: Each cluster is measured by segment count and total miles

### Key Concepts

- **Connectivity Clusters**: Groups of streets reachable without using high-stress links
- **Barriers**: High-stress roads (arterials, highways) that separate low-stress clusters
- **LTS 2 Threshold**: Represents routes comfortable for "most adults" - the mainstream cycling population

### Interpreting Results

The analysis reveals network fragmentation:
- Many small, disconnected low-stress islands exist throughout the city
- The largest cluster contains only a fraction of total low-stress miles
- Gaps between clusters represent infrastructure improvement opportunities

### Configuration

Key parameters in the script:
- `LTS_THRESHOLD = 2` - Maximum LTS level for "low stress" connectivity
- `SNAP_TOLERANCE = 15` - Meters for snapping street endpoints together
- `MIN_CLUSTER_SEGMENTS = 3` - Minimum segments to display a cluster

### Improvement Priorities

Based on the analysis, infrastructure improvements should focus on:
1. Identifying strategic barrier crossings (arterials, highways)
2. Prioritizing connections between large clusters
3. Adding protected infrastructure on high-stress barriers

## Key Findings

The connectivity analysis reveals how fragmented the low-stress network is. When filtering to LTS <= 2 (routes comfortable for most adults):
- The network breaks into many disconnected clusters
- Cyclists must use high-stress roads to travel between most areas
- Strategic infrastructure improvements could connect these islands

## References

Mekuria, M. C., Furth, P. G., & Nixon, H. (2012). Low-stress bicycling and network connectivity. Mineta Transportation Institute.

Furth, P. G., Mekuria, M. C., & Nixon, H. (2016). Network connectivity for low-stress bicycling. Transportation Research Record, 2587(1), 41-49.
