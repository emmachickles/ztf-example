# ZTF Light Curve Tutorial

A beginner-friendly tutorial for exploring ZTF period-finding catalogs and plotting light curves.

## Quick start

### 1. Create a conda environment

```bash
conda create -n ztf-example python=3.10 -y
conda activate ztf-example
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch JupyterLab

```bash
jupyter lab
```

Then open the notebooks in order:

| Notebook | Topic |
|----------|-------|
| `01_exploring_catalogs.ipynb` | Load and filter the pre-computed Lomb-Scargle / BLS catalogs |
| `02_plotting_lightcurves.ipynb` | Extract and plot ZTF light curves from HDF5 matchfiles |
| `03_period_finding.ipynb` | Compute periodograms and validate periods |

## Data paths

These notebooks expect the following data to be available on the system:

- `/ztf/matchfiles/` — HDF5 matchfiles organized by field number
- `/ztf/catalogs/lomb_scargle/` — `.result` files from the Lomb-Scargle period search
- `/ztf/catalogs/box_least_squares/` — `.result` files from the BLS period search

## Files

- `ztf_tools.py` — Python module with high-level functions used by the notebooks
- `interesting_objects.csv` — Placeholder list of interesting sources (replace with real objects)
- `ZTF_Fields.txt` — ZTF field definitions (used internally by `ztf_tools`)
- `archive/` — Original scripts (`plot_ztf_lc.py`, `utils.py`) preserved for reference
