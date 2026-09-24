"""
Nearest Neighbor Distance Analysis (Multi-Order) with CSR Simulation

This script calculates nearest neighbor (NN) distances between detected cluster centers in HDF5 files (from single-molecule localization microscopy (SMLM) experiments). 
It can process multiple HDF5 files recursively within a specified folder and its subfolders.

Author: Tanja Menche
Affiliation: Research group of Mike Heilemann, Goethe University Frankfurt am Main, Germany
Version: v1.0.0
Date: 2026-09-22


About this script:
------------------
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


User Inputs (edit the variables below):
---------------------------------------
- INPUT_DIR: The root directory containing the HDF5 files and optionally the YAML files with the polygonal ROI coordinates. The script searches this folder and all of its subfolders recursively.
- FILE_SUFFIX: The suffix that the HDF5 files must end with. For example, '_ROI_dbscan_centers.hdf5'. Use '.hdf5' to process all HDF5 files.
- OUTPUT_FOLDER: The folder where the output CSV files and simulated HDF5 files are saved. If set to None, output files are saved next to the corresponding input files.
- N_NEAREST_NEIGHBORS: The number of nearest neighbors to calculate. For example, 4 calculates the 1st, 2nd, 3rd, and 4th nearest neighbors.
- PIXEL_SIZE_NM: The pixel size in nanometers. This value is used to convert the calculated nearest neighbor distances from pixels to nanometers.
- ROI: The global Region of Interest to use for filtering the data. Set to None to use all data points, provide a bounding box as (x_min, y_min, x_max, y_max), or provide a path to a YAML polygon or image mask.
- ROI_YAML_SUFFIX: The suffix used to identify per-file YAML ROI polygons. For example, '_ROI_picks'. Set to None to disable automatic per-file YAML ROI matching.
- RANDOM_SEED: The random seed used for the CSR simulation. Set to None to generate different random simulations on each run. Specify an integer, such as 42, to make the simulation reproducible.

Information about the input data:
--------------------------------
The HDF5 files contain cluster center coordinates in the 'locs' dataset. The script uses the 'x' and 'y' fields of this dataset to calculate nearest neighbor distances.
The input coordinates are assumed to be in pixels. Nearest neighbor distances are converted from pixels to nanometers using PIXEL_SIZE_NM.
The script can optionally use a ROI to restrict the analysis to a specific area. If no ROI is specified, all cluster centers in the HDF5 file are used. A global ROI can be defined and applied 
to all input files. Alternatively, individual YAML files with polygonal ROI vertices can be automatically matched to each HDF5 file.
Corresponding HDF5 files and YAML files can be generated with the Picasso Software version (https://github.com/jungmannlab/picasso) version v0.7.3 from SMLM data (find more information in the README.md file).

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
The YAML file must contain a 'Vertices' entry defining the polygon coordinates of the ROI. The script uses these vertices to determine which cluster centers are located within the polygon
and to simulate a CSR distribution of points within the selected ROI.


How to use:
-----------
1. Edit the variables in the CONFIGURATION section below to match your data and analysis requirements. Optionally, set RANDOM_SEED to an integer if reproducible simulations are required.
2. Run the script.
   The script will recursively process all matching HDF5 files.
3. For each processed file, the script saves a CSV file containing the experimental and simulated nearest neighbor distances. The script also saves the simulated cluster centers as an HDF5 file 
accompanied by a YAML file of the same filename for compatibility with the Picasso Software.

The default CSV output is named:
<input_filename>_nearest_neighbors.csv

The default simulated HDF5 output is named:
<input_filename>_simulation.hdf5

If OUTPUT_FOLDER is specified, these files are saved in that folder. Otherwise, they
are saved next to the corresponding input HDF5 files.
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Tuple, Union, List
import h5py
import numpy as np
import pandas as pd
import yaml
from scipy import spatial

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Input / Output -----------------------------------------------------------

INPUT_DIR = r"C:\nearest_neighbor_distances\example_data\input_data"  
# Root directory containing HDF5 files (searched recursively), input e.g. "C:\nearest_neighbor_distances\example_data\input_data"
FILE_SUFFIX = "_ROI_dbscan_centers.hdf5"  
# Only process files ending with this suffix (use ".hdf5" for all), input e.g. "_ROI_DBSCAN_centers.hdf5"

OUTPUT_FOLDER = r"C:\nearest_neighbor_distances\example_data\output_data"  
# Where CSV + simulation files are saved (None = next to input files), input e.g. "C:\nearest_neighbor_distances\example_data\output_data"


# --- Nearest Neighbor Analysis -----------------------------------------------

N_NEAREST_NEIGHBORS = 4  
# Number of nearest neighbors to compute (e.g. 3 → 1st, 2nd, 3rd NN)


# --- Units -------------------------------------------------------------------

PIXEL_SIZE_NM = 157.0  
# Conversion factor: 1 pixel = X nanometers (used for NN distances)

# --- ROI (Region of Interest) ------------------------------------------------

ROI = None  
# Global ROI definition (used if no per-file YAML is applied)
#
# Options:
#   None → use all data points
#   (x_min, y_min, x_max, y_max) → rectangular ROI
#   "path/to/file.yaml" → polygon ROI from YAML (same ROI for all files)


ROI_YAML_SUFFIX = "_ROI_picks"
# Enables per-file ROI using YAML polygons, input e.g. "_ROI_picks"
#
# For each HDF5 file, the script searches for:
#   <base_name> + ROI_YAML_SUFFIX + ".yaml"
#
# Example:
#   cell1_ROI_dbscan_centers.hdf5 → cell1_ROI_picks.yaml
#
# Behavior:
#   If matching YAML is found → overrides ROI
#   If not found:
#       - If ROI is None → use full dataset
#       - Else → fallback to global ROI
#
# Set to None to disable per-file ROI completely


# --- Simulation --------------------------------------------------------------

RANDOM_SEED = None  
# Seed for random number generator (simulation reproducibility)
#
# None → different random simulation each run
# Integer (e.g. 42) → reproducible simulation results


# =============================================================================


# Optional: convert YAML ROI coordinates to nm

ROI_COORDS_IN_NM = False  
# If True: YAML ROI coordinates are scaled to nm using PIXEL_SIZE_NM
# If False: YAML ROI coordinates are assumed to already match data units (usually pixels)

# Optional: use utils if available in same directory
try:
    from utils import load_hdf5, save_hdf5, load_info, load_mask
except ImportError:
    # Fallback implementations
    def _load_info_fallback(file_path):
        path_base = str(file_path).rsplit(".", 1)[0]
        filename = path_base + ".yaml"
        try:
            with open(filename, "r") as info_file:
                return list(yaml.load_all(info_file, Loader=yaml.UnsafeLoader))
        except FileNotFoundError:
            return []

    def load_hdf5(file_path):
        with h5py.File(file_path, 'r') as hdf5_file:
            locs_data = hdf5_file['locs'][:]
            info = _load_info_fallback(file_path)
            return locs_data, info

    def save_hdf5(file_path, locs, info):
        with h5py.File(file_path, "w") as locs_file:
            locs_file.create_dataset("locs", data=locs)
        if info:
            yaml_path = str(file_path).replace('.hdf5', '.yaml')
            with open(yaml_path, "w") as f:
                yaml.dump_all(info, f, default_flow_style=False)

    def load_info(file_path):
        return _load_info_fallback(file_path)

    def load_mask(mask_path):
        try:
            import cv2
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            return mask
        except (ImportError, Exception):
            return None


# ROI type: None, bbox tuple, mask path, YAML path, or dict with 'polygon' key
ROI_TYPE = Union[
    None,
    Tuple[float, float, float, float],
    str,
    Path,
    dict,  # {'polygon': [(x,y), ...], 'bounds': (x_min, y_min, x_max, y_max)}
]


def _point_in_polygon(x: np.ndarray, y: np.ndarray, vertices: List[Tuple[float, float]]) -> np.ndarray:
    """Point-in-polygon test. Returns boolean array."""
    try:
        from matplotlib.path import Path
        poly_path = Path(vertices)
        points = np.column_stack([x, y])
        return poly_path.contains_points(points)
    except ImportError:
        # Fallback: ray-casting
        n = len(vertices)
        if n < 3:
            return np.zeros(len(x), dtype=bool)
        inside = np.zeros(len(x), dtype=bool)
        for i in range(n):
            j = (i + 1) % n
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            if abs(yj - yi) < 1e-10:
                continue
            mask = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
            inside ^= mask
        return inside


def _read_roi_polygon_from_yaml(
    yaml_path: Path,
    pixel_size_nm: float = 1.0,
    scale_to_nm: bool = False
) -> Optional[dict]:
    """
    Read polygon vertices from YAML (cluster_density_v1 format).
    YAML vertices are in pixel coordinates.
    scale_to_nm: if True, multiply by pixel_size_nm (for locs in nm). If False, use as-is (locs in pixels).
    """
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        if 'Vertices' not in data:
            return None
        scale = pixel_size_nm if scale_to_nm else 1.0
        vertices = [
            (coord[0] * scale, coord[1] * scale)
            for coord_group in data['Vertices']
            for coord in coord_group
        ]
        if len(vertices) < 3:
            return None
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        bounds = (min(xs), min(ys), max(xs), max(ys))
        return {'polygon': vertices, 'bounds': bounds}
    except Exception:
        return None


def _get_roi_for_hdf5(
    hdf5_path: Path,
    input_dir: Path,
    roi_yaml_suffix: Optional[str],
    hdf5_file_suffix: Optional[str] = None
) -> Optional[dict]:
    """Find matching YAML ROI file for an HDF5 (cluster_density_v1 style)."""
    if not roi_yaml_suffix:
        return None
    stem = hdf5_path.stem
    # Remove HDF5 suffix to get base name (like cluster_density get_base_name)
    if hdf5_file_suffix:
        suffix_no_ext = hdf5_file_suffix.replace(".hdf5", "").replace(".h5", "")
        if stem.endswith(suffix_no_ext):
            stem = stem[: -len(suffix_no_ext)]
    else:
        for suffix in ['_ROI_fl_dbscan_centers_filter', '_filtered_in', '_centers', '_dbscan']:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
    yaml_name = stem + roi_yaml_suffix + ".yaml"
    # Look in same directory as HDF5 first
    yaml_path = hdf5_path.parent / yaml_name
    if yaml_path.exists():
        return _read_roi_polygon_from_yaml(yaml_path, PIXEL_SIZE_NM, scale_to_nm=ROI_COORDS_IN_NM)
    # Search recursively in input_dir
    for p in Path(input_dir).rglob(yaml_name):
        if p.exists():
            return _read_roi_polygon_from_yaml(p, PIXEL_SIZE_NM, scale_to_nm=ROI_COORDS_IN_NM)
    return None


def filter_locs_by_roi(
    locs: np.ndarray,
    roi: Optional[ROI_TYPE]
) -> np.ndarray:
    """
    Filter localization data to points within ROI.

    Parameters
    ----------
    locs : np.ndarray
        Structured array with 'x' and 'y' fields
    roi : None, tuple, dict, or str/Path
        - None: use full data (no filtering)
        - (x_min, y_min, x_max, y_max): bounding box
        - dict with 'polygon': YAML polygon ROI
        - str/Path to .yaml: polygon from YAML
        - str/Path to image: mask (255 = valid)

    Returns
    -------
    np.ndarray : Filtered locs
    """
    x = locs['x']
    y = locs['y']

    if roi is None:
        return locs

    if isinstance(roi, dict) and 'polygon' in roi:
        vertices = roi['polygon']
        valid = _point_in_polygon(x, y, vertices)
        return locs[valid]

    if isinstance(roi, (tuple, list)) and len(roi) == 4:
        x_min, y_min, x_max, y_max = roi
        valid = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        return locs[valid]

    # Path to YAML or mask
    roi_path = Path(roi) if isinstance(roi, (str, Path)) else None
    if roi_path and roi_path.suffix.lower() in ('.yaml', '.yml'):
        poly_dict = _read_roi_polygon_from_yaml(roi_path, PIXEL_SIZE_NM, scale_to_nm=ROI_COORDS_IN_NM)
        if poly_dict:
            valid = _point_in_polygon(x, y, poly_dict['polygon'])
            return locs[valid]
        print(f"Warning: Could not read ROI from YAML {roi}, using full data")
        return locs

    # ROI is mask path
    mask = load_mask(roi)
    if mask is None:
        print(f"Warning: Could not load ROI mask from {roi}, using full data")
        return locs

    x_idx = np.clip(x.astype(int), 0, mask.shape[1] - 1)
    y_idx = np.clip(y.astype(int), 0, mask.shape[0] - 1)
    valid = mask[y_idx, x_idx] == 255
    return locs[valid]


def get_roi_bounds_for_simulation(
    locs: np.ndarray,
    roi: Optional[ROI_TYPE]
) -> Tuple[float, float, float, float]:
    """Get (x_min, y_min, x_max, y_max) for simulation area."""
    if roi is None:
        return (
            float(locs['x'].min()),
            float(locs['y'].min()),
            float(locs['x'].max()),
            float(locs['y'].max())
        )
    if isinstance(roi, dict) and 'bounds' in roi:
        return roi['bounds']
    if isinstance(roi, (tuple, list)) and len(roi) == 4:
        return tuple(roi)
    # For mask: use mask bounding box of valid pixels
    if isinstance(roi, (str, Path)):
        mask = load_mask(roi)
        if mask is not None:
            coords = np.where(mask == 255)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                return (float(x_min), float(y_min), float(x_max), float(y_max))
    return (
        float(locs['x'].min()),
        float(locs['y'].min()),
        float(locs['x'].max()),
        float(locs['y'].max())
    )


def get_roi_area_for_simulation(
    roi: Optional[Union[Tuple[float, float, float, float], str, Path]],
    bounds: Tuple[float, float, float, float]
) -> float:
    """Get area of ROI for density calculation."""
    if isinstance(roi, (tuple, list)) and len(roi) == 4:
        x_min, y_min, x_max, y_max = roi
        return (x_max - x_min) * (y_max - y_min)
    if roi is None:
        x_min, y_min, x_max, y_max = bounds
        return (x_max - x_min) * (y_max - y_min)
    # Mask: count valid pixels
    mask = load_mask(roi)
    if mask is None:
        x_min, y_min, x_max, y_max = bounds
        return (x_max - x_min) * (y_max - y_min)
    return float(np.sum(mask == 255))


def nearest_neighbor_distances(coords: np.ndarray, k_neighbors: int) -> np.ndarray:
    """
    Compute nearest neighbor distances up to k-th neighbor.
    """
    if len(coords) < 2:
        return np.empty((0, k_neighbors), dtype=np.float64)

    k = k_neighbors + 1
    tree = spatial.cKDTree(coords.astype(np.float64))
    d, _ = tree.query(coords, k=k)

    return d[:, 1:]


def simulate_clusters_same_density(
    locs: np.ndarray,
    roi: Optional[ROI_TYPE],
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate cluster centers with same density as input, within ROI.

    Uses Complete Spatial Randomness (CSR) - uniform random distribution.

    Parameters
    ----------
    locs : np.ndarray
        Input locs (filtered by ROI)
    roi : ROI specification (None, bbox, polygon dict, mask path)
    random_seed : int, optional

    Returns
    -------
    np.ndarray : Simulated locs with same structure, new x,y coordinates
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    n_points = len(locs)
    if n_points == 0:
        return np.array([], dtype=locs.dtype)

    bounds = get_roi_bounds_for_simulation(locs, roi)
    x_min, y_min, x_max, y_max = bounds

    # Polygon ROI: rejection sampling within bounding box
    if isinstance(roi, dict) and 'polygon' in roi:
        vertices = roi['polygon']
        x_sim_list, y_sim_list = [], []
        for _ in range(500):  # Safety limit
            if len(x_sim_list) >= n_points:
                break
            batch = max((n_points - len(x_sim_list)) * 4, 100)
            x_cand = rng.uniform(x_min, x_max, batch)
            y_cand = rng.uniform(y_min, y_max, batch)
            inside = _point_in_polygon(x_cand, y_cand, vertices)
            x_sim_list.extend(x_cand[inside].tolist())
            y_sim_list.extend(y_cand[inside].tolist())
        n_got = len(x_sim_list)
        x_sim = np.empty(n_points, dtype=np.float64)
        y_sim = np.empty(n_points, dtype=np.float64)
        x_sim[:n_got] = x_sim_list[:n_points]
        y_sim[:n_got] = y_sim_list[:n_points]
        if n_got < n_points:
            x_sim[n_got:] = rng.uniform(x_min, x_max, n_points - n_got)
            y_sim[n_got:] = rng.uniform(y_min, y_max, n_points - n_got)
    # Rectangular ROI
    elif roi is None or (isinstance(roi, (tuple, list)) and len(roi) == 4):
        x_sim = rng.uniform(x_min, x_max, n_points)
        y_sim = rng.uniform(y_min, y_max, n_points)
    # Mask: sample within valid pixels
    elif isinstance(roi, (str, Path)):
        mask = load_mask(roi)
        if mask is None:
            x_sim = rng.uniform(x_min, x_max, n_points)
            y_sim = rng.uniform(y_min, y_max, n_points)
        else:
            valid_indices = np.argwhere(mask == 255)
            if len(valid_indices) == 0:
                x_sim = rng.uniform(x_min, x_max, n_points)
                y_sim = rng.uniform(y_min, y_max, n_points)
            else:
                chosen = rng.choice(len(valid_indices), size=n_points, replace=True)
                selected = valid_indices[chosen]
                jitter = rng.random(size=(n_points, 2))
                coords = selected + jitter
                y_sim = coords[:, 0]
                x_sim = coords[:, 1]
    else:
        x_sim = rng.uniform(x_min, x_max, n_points)
        y_sim = rng.uniform(y_min, y_max, n_points)

    sim_locs = np.copy(locs)
    sim_locs['x'] = x_sim
    sim_locs['y'] = y_sim
    return sim_locs


def process_single_file(
    file_path: Path,
    roi: Optional[Union[Tuple[float, float, float, float], str, Path]],
    random_seed: Optional[int],
    save_simulation: bool = True,
    output_dir: Optional[Path] = None,
    output_basename: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[Path]]:
    """
    Process a single HDF5 file: load, filter, NN analysis, simulate, save.

    output_dir: if set, save simulation HDF5 here; else save next to input file.

    Returns
    -------
    (nn_distances_data, nn_distances_simulation, sim_file_path)
    """
    locs, info = load_hdf5(str(file_path))
    locs_roi = filter_locs_by_roi(locs, roi)

    if len(locs_roi) == 0:
        print(f"  Warning: No points within ROI for {file_path.name}")
        return np.array([]), np.array([]), None

    coords = np.column_stack([locs_roi['x'], locs_roi['y']])
    nn_distances = nearest_neighbor_distances(coords, N_NEAREST_NEIGHBORS) * PIXEL_SIZE_NM

    # Simulate
    sim_locs = simulate_clusters_same_density(locs_roi, roi, random_seed)
    coords_sim = np.column_stack([sim_locs['x'], sim_locs['y']])
    nn_distances_sim = nearest_neighbor_distances(coords_sim, N_NEAREST_NEIGHBORS) * PIXEL_SIZE_NM

    sim_path = None
    if save_simulation and len(sim_locs) > 0:
        save_dir = Path(output_dir) if output_dir else file_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        base = output_basename if output_basename else file_path.stem
        sim_hdf5_path = save_dir / (base + "_simulation.hdf5")
        save_hdf5(str(sim_hdf5_path), sim_locs, info)
        sim_path = sim_hdf5_path
        print(f"  Saved simulation: {sim_hdf5_path.name}")

    return nn_distances, nn_distances_sim, sim_path


def find_hdf5_files(root_dir: Path, file_suffix: str) -> List[Path]:
    """Recursively find all HDF5 files matching the suffix in subfolders."""
    root_dir = Path(root_dir)
    if not root_dir.exists():
        return []

    files = []
    for path in root_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in ('.hdf5', '.h5'):
            if file_suffix and not path.name.endswith(file_suffix):
                continue
            if "_simulation" in path.name:
                continue  # Skip our own output files
            files.append(path)
    return sorted(files)


def _sanitize_column_name(name: str) -> str:
    """Create a valid CSV column name from file path."""
    s = str(name).replace("\\", "_").replace("/", "_").replace(".", "_")
    # Remove or replace characters that might cause issues
    s = "".join(c if c.isalnum() or c == "_" else "_" for c in s)
    return s[:80]  # Limit length


def run_batch_analysis(
    input_dir: Union[str, Path],
    file_suffix: str,
    roi: Optional[ROI_TYPE] = None,
    roi_yaml_suffix: Optional[str] = None,
    output_folder: Optional[Union[str, Path]] = None,
    random_seed: Optional[int] = None
):
    """
    Run nearest neighbor analysis and simulation on all matching HDF5 files.
    Saves all individual NN distances in separate columns (one per file, one per simulation).

    output_folder: if set, CSV and all simulation HDF5 files are saved here.
    When roi_yaml_suffix is set (e.g. "_ROI_picks"), each HDF5 is matched to a YAML ROI file
    by base name, like cluster_density_v1.py.
    """
    input_dir = Path(input_dir)
    if output_folder is not None:
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = output_dir / "nearest_neighbor_results.csv"
    else:
        output_dir = None
        output_csv = input_dir / "nearest_neighbor_results.csv"

    files = find_hdf5_files(input_dir, file_suffix)
    if not files:
        print(f"No HDF5 files found with suffix '{file_suffix}' in {input_dir}")
        return

    print(f"Found {len(files)} HDF5 file(s) to process")

    for fp in files:
        try:
            try:
                rel_path = fp.relative_to(input_dir)
            except ValueError:
                rel_path = fp

            print(f"\nProcessing: {rel_path}")

            # ROI handling
            file_roi = roi
            if roi_yaml_suffix:
                poly_roi = _get_roi_for_hdf5(fp, input_dir, roi_yaml_suffix, hdf5_file_suffix=file_suffix)
                if poly_roi is not None:
                    file_roi = poly_roi
                    print(f"  Using ROI from YAML polygon ({len(poly_roi['polygon'])} vertices)")
                elif roi is None:
                    print(f"  No matching YAML ROI found, using full data")

            nn_data, nn_sim, _ = process_single_file(
                fp, file_roi, random_seed,
                save_simulation=True,
                output_dir=output_dir,
                output_basename=fp.stem if output_dir else None
            )

            # Skip empty
            if nn_data.size == 0:
                print("  Skipping empty dataset")
                continue

            # Create DataFrame
            data_dict = {}
            for i in range(N_NEAREST_NEIGHBORS):
                data_dict[f"nn_{i+1}_nm"] = nn_data[:, i]
                data_dict[f"sim_nn_{i+1}_nm"] = nn_sim[:, i]

            df = pd.DataFrame(data_dict)

            # Save per file
            save_dir = Path(output_dir) if output_dir else fp.parent
            save_dir.mkdir(parents=True, exist_ok=True)

            csv_path = save_dir / f"{fp.stem}_nearest_neighbors.csv"
            df.to_csv(csv_path, index=False)

            print(f"  Saved CSV: {csv_path.name}")

        except Exception as e:
            print(f"  Error processing {fp}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Nearest neighbor analysis and simulation for cluster centers in HDF5 files"
    )
    parser.add_argument(
        "--input", "-i",
        default=INPUT_DIR,
        help="Root directory to search for HDF5 files"
    )
    parser.add_argument(
        "--suffix", "-s",
        default=FILE_SUFFIX,
        help="File suffix to match (e.g., _filtered_in.hdf5). Use .hdf5 for all."
    )
    parser.add_argument(
        "--roi",
        default=None,
        help="ROI: path to mask image, or 'x_min,y_min,x_max,y_max' for bounding box"
    )
    parser.add_argument(
        "--roi-yaml-suffix",
        default=None,
        help="YAML ROI suffix (e.g. _ROI_picks) - match per-file ROI like cluster_density_v1"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output folder for CSV and simulation HDF5 files. If not set, uses OUTPUT_FOLDER from config"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for simulation"
    )
    args = parser.parse_args()

    roi = ROI if args.roi is None else args.roi
    if isinstance(roi, str) and "," in roi:
        parts = [float(p.strip()) for p in roi.split(",")]
        if len(parts) == 4:
            roi = tuple(parts)

    roi_yaml = args.roi_yaml_suffix if args.roi_yaml_suffix is not None else ROI_YAML_SUFFIX
    output_folder = args.output if args.output is not None else OUTPUT_FOLDER

    run_batch_analysis(
        input_dir=args.input,
        file_suffix=args.suffix,
        roi=roi,
        roi_yaml_suffix=roi_yaml,
        output_folder=output_folder,
        random_seed=args.seed
    )


if __name__ == "__main__":
    # When run without args, use configuration constants
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        run_batch_analysis(
            input_dir=INPUT_DIR,
            file_suffix=FILE_SUFFIX,
            roi=ROI,
            roi_yaml_suffix=ROI_YAML_SUFFIX,
            output_folder=OUTPUT_FOLDER,
            random_seed=RANDOM_SEED
        )
