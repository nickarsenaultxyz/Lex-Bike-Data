# Lexington Bike Infrastructure Level of Traffic Stress (LTS) Analysis

A geospatial analysis tool that calculates and visualizes the Level of Traffic Stress (LTS) for bike infrastructure in Lexington, Kentucky. This project enriches bike facility data with street characteristics (speed limits, traffic volume) and produces an interactive map showing where cyclists face different levels of traffic stress.

## Overview

The Level of Traffic Stress framework categorizes bike infrastructure into four levels:
- **LTS 1**: Low stress - comfortable for children
- **LTS 2**: Moderate stress - comfortable for most adults
- **LTS 3**: High stress - comfortable for confident cyclists
- **LTS 4**: Very high stress - comfortable only for very confident cyclists

This analysis helps identify gaps in low-stress bike networks and prioritize infrastructure improvements.

## Features

- **Two-stage matching algorithm**: Combines road name matching with spatial proximity matching for comprehensive coverage
- **Bearing-based filtering**: Prevents incorrect matches to perpendicular streets
- **Interactive map**: Folium-based visualization with color-coded LTS levels
- **Data enrichment**: Merges bike facility data with street speed limits and traffic volumes
- **Comprehensive coverage**: 84% match rate using Lexington's complete street dataset

## Data Sources

### Input Files
- **lexbike.geojson** - Bike infrastructure data for Lexington
  - Contains bike lanes, shared lanes, greenways, and proposed facilities
  - Includes facility types, intersection names, and segment lengths

- **lex_street_data.geojson** - Complete Lexington street data
  - Contains speed limits, road classifications, and geometry for all streets
  - Replaces KYTC data which only covered state-maintained roads

- **Traffic_Station_Counts.geojson** (optional) - AADT traffic volume data

### Output Files
- **lexbike_with_speed_aadt.geojson** - Enriched bike data with speed, AADT, and LTS values
- **lexbike_LTS_map.html** - Interactive map visualization

## Installation

### Requirements
- Python 3.8+
- Virtual environment (recommended)

## Usage

### Basic Analysis

```bash
cd lex-bike
python3 bikestress_route.py
```

This will:
1. Load bike infrastructure and street data
2. Match bike segments to street speed data using two strategies:
   - Direct road name matching (using `Name_Network` field)
   - Spatial proximity matching with bearing filtering
3. Calculate Level of Traffic Stress for each segment
4. Generate an enriched GeoJSON file and interactive HTML map

### Output Summary

The script outputs:
- Match statistics (name matches vs spatial matches)
- Speed limit distribution across matched segments
- LTS distribution showing percentage and mileage for each level
- Data coverage summary

Example output:
```
Loading Lexington street data...
  Total street segments: 13775
  Step 1: Matching by road name...
    Matched 136 segments by road name
  Step 2: Spatial matching for 392 remaining segments...
    Matched 307 segments spatially

  Matched 443 / 528 segments to speeds (83.9%)

LTS DISTRIBUTION
LTS 1:  125 segments ( 23.7%) - 45.40 miles
LTS 2:   63 segments ( 11.9%) - 12.84 miles
LTS 3:  207 segments ( 39.2%) - 59.54 miles
LTS 4:  133 segments ( 25.2%) - 58.83 miles
```

## Matching Algorithm

### Strategy 1: Road Name Matching
- Normalizes road names (e.g., "STREET" → "ST", "ROAD" → "RD")
- Matches bike segments to street data using the `Name_Network` field
- Uses median speed when multiple street segments exist for one road
- Achieves ~26% coverage with high accuracy

### Strategy 2: Spatial Matching
- Projects geometries to local CRS (UTM Zone 16N) for accurate distance calculations
- Uses nearest neighbor spatial join within 30m
- Calculates bearing (directional angle) for both bike and street segments
- Filters out perpendicular matches (bearing difference > 80°)
- Achieves ~58% additional coverage

### Key Parameters
- `NEAREST_MAX_DIST_M = 30` - Maximum distance for spatial matching
- `BEARING_THRESHOLD_PERPENDICULAR = 80` - Degrees to filter crossing streets

## LTS Calculation Methodology

The LTS calculation considers:
- **Speed limits** - Higher speeds increase stress
- **Facility type** - Protected lanes vs shared lanes vs no infrastructure
- **Traffic volume (AADT)** - Higher volumes increase stress
- **Number of lanes** - More lanes increase stress
- **Lane width** - Narrow lanes increase stress

See `bikestress_route.py` lines 219-474 for detailed LTS calculation logic.

## Project Structure

```
Lex_Bike_DataAnalysis/
├── lex-bike/
│   ├── bikestress_route.py          # Main analysis script
│   ├── lexbike.geojson               # Input: Bike infrastructure
│   ├── lex_street_data.geojson      # Input: Street speed/class data
│   ├── lexbike_with_speed_aadt.geojson  # Output: Enriched data
│   ├── lexbike_LTS_map.html         # Output: Interactive map
│   └── speed_overrides_example.py   # Example speed override config
├── venv/                             # Python virtual environment
└── README.md                         # This file
```

## Configuration

### Adjusting Match Sensitivity
Edit parameters in `bikestress_route.py`:
- Increase `NEAREST_MAX_DIST_M` to match more distant segments (may reduce accuracy)
- Decrease `BEARING_THRESHOLD_PERPENDICULAR` for stricter directional matching

## Viewing Results

Open `lexbike_LTS_map.html` in a web browser to explore the interactive map:
- Color-coded segments by LTS level (green = LTS 1, red = LTS 4)
- Click segments for details (intersections, speed, facility type, LTS)
- Use measurement tools, full-screen mode, and location controls
- Pan/zoom to explore different areas

## Known Issues & Limitations

- 16% of segments remain unmatched (mostly off-road paths and greenways)
- Some segments may have outdated speed data
- LTS calculation doesn't account for intersection complexity
- Traffic volume (AADT) data is optional and may not cover all segments

## Future Enhancements

Planned improvements noted in the code:
1. Include walking paths through campus using OSMnx
2. Create routing feature for low-stress route planning
3. Version control with Git branching
4. Validate funded/proposed facility data
5. Clarify mixed-type facility definitions

## Contributing

This is a city planning analysis tool for Lexington, KY. Suggestions and improvements are welcome.

## References

- Level of Traffic Stress methodology: Mekuria, M. C., Furth, P. G., & Nixon, H. (2012)
- OpenStreetMap for base network data
- City of Lexington GIS data for street characteristics

## License

This project uses open data from the City of Lexington and OpenStreetMap.
