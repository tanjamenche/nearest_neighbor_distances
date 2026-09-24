# Nearest Neighbor Distance Analysis (Multi-Order) with CSR Simulation

This script calculates nearest neighbor (NN) distances between detected cluster centers in HDF5 files (from single-molecule localization microscopy (SMLM) experiments). 
It can process multiple HDF5 files recursively within a specified folder and its subfolders.


## Overview

1. Loads HDF5 files containing cluster center coordinates (locs dataset with x, y)
2. Optionally filters coordinates using a Region of Interest (ROI)
   - ROI can be global (bounding box, mask, YAML)
   - OR per-file YAML polygons (automatic matching via suffix)
3. Computes nearest neighbor (NN) distances up to N-th neighbor (configurable)
   - e.g. 1st, 2nd, 3rd nearest neighbors
4. Simulates a random distribution (CSR) with the same point density within the ROI
5. Computes NN distances for the simulated data
6. Converts all distances from pixels to nanometers (nm)
7. Saves simulated CSR distribution as HDF5 files and saves NN distances per input file as individual CSV files:
   - One column per NN order (data + simulation)

Supports batch processing across subfolders.
Parts of this script were developed with assistance from OpenAI's ChatGPT and Cursor AI. The resulting code was reviewed, modified, and validated by the author.


## Requirements

* Python **3.11**
* PyYAML **6.0.1**
* pandas **2.1.4**
* h5py **3.9.0**
* numpy **1.26.4**
* scipy **1.11.4**

## Input Data

### HDF5 files

HDF5 files (.hdf5) containing a list of cluster center coordinates in the 'locs' dataset. The script uses the 'x' and 'y' fields of this dataset to calculate nearest neighbor distances.
The input coordinates are assumed to be in pixels. Nearest neighbor distances are converted from pixels to nanometers using PIXEL_SIZE_NM. 
The script was tested on HDF5 files with cluster centers generated from localization data via the clustering algorithm DBSCAN with the 
[Picasso Software](https://github.com/jungmannlab/picasso) version v0.7.3 and with the [PicassoBatchProcess](https://github.com/HeilemannLab/PicassoBatchProcess) Software.  

The script can optionally use a ROI to restrict the analysis to a specific area. If no ROI is specified, all cluster centers in the HDF5 file are used. A global ROI can be defined and applied 
to all input files. Alternatively, individual YAML files with polygonal ROI vertices can be automatically matched to each HDF5 file.

### YAML files (with ROI vertices)

YAML files (.yaml) contain polygon vertex coordinates under the `Vertices` field. The script uses these vertices to determine which cluster centers are located within the polygon
and to simulate a CSR distribution of points within the selected ROI. The script was tested on YAML files (polygonal ROIs) 
generated with the [Picasso Software](https://github.com/jungmannlab/picasso) version v0.7.3 (modul "Render": -> Tools -> Tool Settings -> Shape: Polygon, -> Tools -> Pick).

For per-file YAML ROI matching, the script searches for a YAML file with the following naming convention:

<base_name> + ROI_YAML_SUFFIX + ".yaml"

For example, with:
FILE_SUFFIX = "_ROI_dbscan_centers.hdf5"
ROI_YAML_SUFFIX = "_ROI_picks"

the following files are associated:
protein1_ROI_dbscan_centers.hdf5
protein1_ROI_picks.yaml

The common base name is:
protein1

Because of this, each input file pair (HDF5 file + ROI YAML file) must have a unique base name within the folders being processed.
For example:
```
input_data/ 

├── cell1/ 

│ ├── cell1_protein1_ROI_picks.yaml 

│ └── cell1_protein1_ROI_dbscan_centers.hdf5 

│ 
└── cell2/ 

  ├── cell2_protein1_ROI_picks.yaml 
  
  └── cell2_protein1_ROI_dbscan_centers.hdf5
```
  

## Configuration

Before running the script, edit the following variables in the **CONFIGURATION** section in `nearest_neighbor_distances.py`:
```python
INPUT_DIR = r"C:\nearest_neighbor_distances\example_data\input_data"  
# Root directory containing HDF5 files (searched recursively), input e.g. "C:\nearest_neighbor_distances\example_data\input_data"
FILE_SUFFIX = "_ROI_dbscan_centers.hdf5"  
# Only process files ending with this suffix (use ".hdf5" for all), input e.g. "_ROI_DBSCAN_centers.hdf5"
OUTPUT_FOLDER = r"C:\Software\GitHub\nearest_neighbor_distances\example_data\output_data"  
# Where CSV + simulation files are saved (None = next to input files), input e.g. "C:\nearest_neighbor_distances\example_data\output_data"
N_NEAREST_NEIGHBORS = 4  
# Number of nearest neighbors to compute (e.g. 3 → 1st, 2nd, 3rd NN)
PIXEL_SIZE_NM = 157.0  
# Conversion factor: 1 pixel = X nanometers (used for NN distances)
ROI = None  
# Global ROI definition (used if no per-file YAML is applied)
# Options:
#   None → use all data points
#   (x_min, y_min, x_max, y_max) → rectangular ROI
#   "path/to/file.yaml" → polygon ROI from YAML (same ROI for all files)
ROI_YAML_SUFFIX = "_ROI_picks"
# Enables per-file ROI using YAML polygons, input e.g. "_ROI_picks"
# For each HDF5 file, the script searches for:
#   <base_name> + ROI_YAML_SUFFIX + ".yaml"
# Example:
#   cell1_ROI_dbscan_centers.hdf5 → cell1_ROI_picks.yaml
# Behavior:
#   If matching YAML is found → overrides ROI
#   If not found:
#       - If ROI is None → use full dataset
#       - Else → fallback to global ROI
# Set to None to disable per-file ROI completely
RANDOM_SEED = None  
# Seed for random number generator (simulation reproducibility)
# None → different random simulation each run
# Integer (e.g. 42) → reproducible simulation results
```

## Installation
```PowerShell
conda create --name nearest_neighbor_distances python=3.11
conda activate nearest_neighbor_distances
cd filepath\nearest_neighbor_distances
conda install --file requirements.txt
```

## Usage

1. Place the YAML and HDF5 input files in the specified folder and/or its subfolders.
2. Open `nearest_neighbor_distances.py`.
3. Set variables in CONFIGURATION section.
4. Open environment:
```PowerShell
conda activate nearest_neighbor_distances
```
5. Navigate to the file path, where nearest_neighbor_distances.py is stored:
```PowerShell
cd filepath\nearest_neighbor_distances
```
6. Run:
```PowerShell
python nearest_neighbor_distances.py
```
The script searches recursively through the specified folder and its subfolders.

## Output

For each processed file, the script saves a CSV file containing the experimental and simulated nearest neighbor distances. The script also saves the simulated cluster centers as an HDF5 file 
accompanied by a YAML file of the same filename for compatibility with the Picasso Software.

The default CSV output is named:
<input_filename>_nearest_neighbors.csv

The default simulated HDF5 output is named:
<input_filename>_simulation.hdf5

If OUTPUT_FOLDER is specified, these files are saved in that folder. Otherwise, they
are saved next to the corresponding input HDF5 files.


