"""
ztf_tools — High-level helpers for working with ZTF light curves and catalogs.

This module wraps the logic from plot_ztf_lc.py and utils.py into a clean API
suitable for interactive exploration in Jupyter notebooks.

Public functions
----------------
load_catalog        Load all .result files from a catalog directory into a DataFrame.
filter_candidates   Apply period/significance cuts to a catalog DataFrame.
get_lightcurve      Full pipeline: field lookup → HDF5 read → BJD → clip → return.
phase_fold          Phase-fold an array of times at a given period.
weighted_bin        Weighted-average phase binning.
plot_lightcurve     Diagnostic plot (raw + phase-folded) for a multi-filter light curve.
"""

import glob
import os

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
import astropy.units as u

# ── Private: field geometry (from utils.py) ──────────────────────────

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_MATCHFILE_DIR = '/ztf/matchfiles'

_FILTERS = {
    'g': {'suffix': 'zg', 'color': 'green'},
    'r': {'suffix': 'zr', 'color': 'red'},
    'i': {'suffix': 'zi', 'color': '#8B0000'},
}


def _ang_dist(ra1, dec1, ra2, dec2):
    """Angular distance (degrees) between two points, all inputs in radians."""
    adist = (np.sin(dec1) * np.sin(dec2)
             + np.cos(dec1) * np.cos(dec2) * np.cos(ra2 - ra1))
    return np.arccos(np.clip(adist, -1, 1)) * 180.0 / np.pi


def _fit_line(x, x0, y0, x1, y1):
    return (y1 - y0) * (x - x0) / (x1 - x0) + y0


def _ortographic_projection(ra, dec, ra0, dec0):
    x = -np.cos(dec) * np.sin(ra - ra0)
    y = np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)
    return x * 180.0 / np.pi, y * 180.0 / np.pi


# CCD corner coordinates (from utils.py)
_CCD_X = [
    -3.646513, -3.647394, -1.920848, -1.920383, -1.790386, -1.790817,
    -0.064115, -0.064099, 0.062113, 0.062129, 1.788830, 1.788400,
    1.918441, 1.918905, 3.645452, 3.644571, -3.646416, -3.646708,
    -1.919998, -1.919844, -1.789454, -1.789597, -0.062733, -0.062727,
    0.061814, 0.061819, 1.788683, 1.788540, 1.918871, 1.919025,
    3.645736, 3.645443, -3.646562, -3.646270, -1.919698, -1.919852,
    -1.789413, -1.789270, -0.062544, -0.062549, 0.062876, 0.062871,
    1.789598, 1.789741, 1.919874, 1.919720, 3.646292, 3.646584,
    -3.645853, -3.644972, -1.918842, -1.919306, -1.789367, -1.788937,
    -0.062651, -0.062666, 0.063143, 0.063128, 1.789415, 1.789845,
    1.919878, 1.919413, 3.645543, 3.646424,
]
_CCD_Y = [
    -3.727898, -2.001758, -2.004785, -3.731333, -3.729368, -2.002803,
    -2.003812, -3.730512, -3.730976, -2.004276, -2.003269, -3.729834,
    -3.731505, -2.004957, -2.001932, -3.728073, -1.816060, -0.089749,
    -0.090622, -1.817335, -1.816611, -0.089881, -0.090172, -1.817035,
    -1.817472, -0.090609, -0.090319, -1.817048, -1.817584, -0.090871,
    -0.089998, -1.816309, 0.090679, 1.816989, 1.818266, 0.091552,
    0.091155, 1.817884, 1.818309, 0.091446, 0.090876, 1.817739,
    1.817315, 0.090586, 0.091290, 1.818003, 1.816728, 0.090417,
    2.002667, 3.728808, 3.732241, 2.005694, 2.003694, 3.730258,
    3.731401, 2.004701, 2.003834, 3.730533, 3.729391, 2.002826,
    2.004674, 3.731221, 3.727789, 2.001648,
]


def _inside_polygon(xp, yp):
    """Return (ccd, quad) if (xp, yp) falls inside a ZTF CCD, else (None, None)."""
    x, y = _CCD_X, _CCD_Y
    for i in range(16):
        idx = 4 * i
        y_test_1 = _fit_line(xp, x[idx], y[idx], x[idx + 3], y[idx + 3])
        y_test_2 = _fit_line(xp, x[idx + 1], y[idx + 1], x[idx + 2], y[idx + 2])
        if yp < y_test_1 or yp > y_test_2:
            continue
        x_test_1 = _fit_line(yp, y[idx], x[idx], y[idx + 1], x[idx + 1])
        x_test_2 = _fit_line(yp, y[idx + 3], x[idx + 3], y[idx + 2], x[idx + 2])
        if xp < x_test_1 or xp > x_test_2:
            continue
        ccd = i + 1
        y_test = _fit_line(
            xp, 0.5 * (x[idx + 2] + x[idx + 3]), 0.5 * (y[idx + 2] + y[idx + 3]),
            0.5 * (x[idx] + x[idx + 1]), 0.5 * (y[idx] + y[idx + 1]),
        )
        x_test = _fit_line(
            yp, 0.5 * (y[idx] + y[idx + 3]), 0.5 * (x[idx] + x[idx + 3]),
            0.5 * (y[idx + 1] + y[idx + 2]), 0.5 * (x[idx + 1] + x[idx + 2]),
        )
        if yp < y_test:
            quad = 4 if xp < x_test else 3
        else:
            quad = 1 if xp < x_test else 2
        return ccd, quad
    return None, None


def read_fields():
    """Load ZTF field definitions from ZTF_Fields.txt (relative to this module)."""
    path = os.path.join(_MODULE_DIR, 'ZTF_Fields.txt')
    fieldno, ra, dec = np.loadtxt(
        path, unpack=True, usecols=(0, 1, 2), dtype='int,float,float')
    return fieldno, ra, dec


def get_field_id(ra_deg, dec_deg, fieldno, ra_all, dec_all):
    """Find all (field, ccd, quad) tuples covering the given (ra, dec) in degrees."""
    deg = np.pi / 180.0
    ra = ra_deg * deg
    dec = dec_deg * deg
    ra_arr = ra_all * deg
    dec_arr = dec_all * deg

    ADIST_MAX = 5.66
    res = []
    for i in range(len(ra_arr)):
        adist = _ang_dist(ra, dec, ra_arr[i], dec_arr[i])
        if adist >= ADIST_MAX:
            continue
        x, y = _ortographic_projection(ra, dec, ra_arr[i], dec_arr[i])
        ccd, quad = _inside_polygon(x, y)
        if ccd is not None and quad is not None:
            res.append([fieldno[i], ccd, quad])
    return res


# Load fields once at import time
_fieldno, _field_ra, _field_dec = read_fields()


# ── Private: HDF5 matchfile reading ─────────────────────────────────

def _getobj(ps_id, fname):
    """Extract a single source's light curve from a ZTF HDF5 matchfile."""
    f = h5py.File(fname, 'r')
    sources = np.array(f['data']['sources']['gaia_id'])
    idx = np.where(sources == ps_id)

    if len(idx[0]) == 0:
        f.close()
        return None

    flux_ref = 10 ** (-0.4 * f['data']['sources'][idx]['mag_ref'])

    exposures = f['data']['exposures']
    times = exposures['jd']
    pid = exposures['pid']

    n_exp = len(times)
    index = np.linspace(0, n_exp - 1, n_exp) + idx[0] * n_exp
    index = index.flatten().astype(int)

    sourcedata = f['data']['sourcedata']
    data = sourcedata[index]

    flux = data['flux']
    fluxerr = data['flux_err']
    flag = data['flag']

    f.close()
    return times, flux, fluxerr, flag, pid, flux_ref


# ── Private: BJD conversion ─────────────────────────────────────────

def _bjd_convert(times, ra, dec, date_format='mjd', telescope='Palomar'):
    """Convert MJD timestamps to BJD_TCB."""
    t = Time(times, format=date_format, scale='utc')
    t2 = t.tcb
    c = SkyCoord(ra, dec, unit='deg')
    observatory = EarthLocation.of_site(telescope)
    delta = t2.light_travel_time(c, kind='barycentric', location=observatory)
    return t2 + delta


# ── Private: outlier clipping ────────────────────────────────────────

def _clip_iqr(t, y, dy, n_iqr=10):
    """Remove outliers beyond n_iqr * IQR from the median."""
    median = np.median(y)
    q75, q25 = np.percentile(y, [75, 25])
    iqr = q75 - q25
    keep = (y > median - n_iqr * iqr) & (y < median + n_iqr * iqr)
    return t[keep], y[keep], dy[keep]


def _clip_flux_err(t, y, dy, n_sigma=3):
    """Remove points whose flux errors are more than n_sigma from the median error."""
    median_err = np.median(dy)
    std_err = np.std(dy)
    keep = dy < median_err + n_sigma * std_err
    return t[keep], y[keep], dy[keep]


# ── Private: single-filter extraction ────────────────────────────────

def _extract_filter(ps_id, field_ccd_quads, filt_suffix, matchfile_dir):
    """Extract light curve data for one filter across all field/ccd/quad combos."""
    ident_ps = [ps_id]
    times_acc, flux_acc, ferr_acc, flag_acc, ref_fluxes = [], [], [], [], []

    for field, ccd, quad in field_ccd_quads:
        try:
            fname = (f'{matchfile_dir}/{field:04d}/'
                     f'data_{field:04d}_{ccd:02d}_{quad:1d}_{filt_suffix}.h5')
            result = _getobj(ident_ps, fname)
            if result is None:
                continue
            bjd, flux, ferr, flag, pid, flux_ref = result
            good = ~np.isnan(flux)
            if len(flag_acc) == 0:
                flag_acc = flag[good]
                times_acc = bjd[good]
                ferr_acc = ferr[good]
                flux_acc = flux[good]
                ref_fluxes.append(flux_ref)
            else:
                flag_acc = np.hstack((flag_acc, flag[good]))
                times_acc = np.hstack((times_acc, bjd[good]))
                flux_acc = np.hstack((flux_acc, flux[good]))
                ferr_acc = np.hstack((ferr_acc, ferr[good]))
        except Exception:
            pass

    if len(times_acc) == 0:
        return None

    t = np.array(times_acc) - 2400000.5  # JD → MJD
    y = np.array(flux_acc) + np.max(ref_fluxes)
    dy = np.array(ferr_acc)
    fl = np.array(flag_acc)

    ok = fl == 0
    t, y, dy = t[ok], y[ok], dy[ok]
    ok = ~np.isnan(y)
    t, y, dy = t[ok], y[ok], dy[ok]
    if len(t) < 3:
        return None
    t, y, dy = _clip_iqr(t, y, dy)
    if len(t) < 3:
        return None
    t, y, dy = _clip_flux_err(t, y, dy)
    if len(t) < 3:
        return None
    return t, y, dy


# =====================================================================
# Public API
# =====================================================================

# ── Column definitions for each catalog type ────────────────────────

LS_COLUMNS = [
    'ra', 'dec', 'ps_id', 'period', 'significance',
    'significance_half', 'significance_twice',
    'mad', 'variance', 'skew', 'kurtosis',
    'ref_flux_flag', 'ref_flux',
]

BLS_COLUMNS = [
    'ra', 'dec', 'ps_id', 'period', 'bls_power',
    'depth', 'depth_err', 'depth_odd', 'depth_odd_err',
    'depth_even', 'depth_even_err', 'depth_half', 'depth_half_err',
    'depth_phased', 'depth_phased_err',
    'harmonic_amplitude', 'harmonic_delta_logL',
    'n_transits', 'n_points_transit', 'transit_duration',
    'mid_transit_time', 'out_of_eclipse_scatter', 'snr',
    'min_time',
]

# Map number of columns → column names for auto-detection
_COL_SCHEMAS = {
    len(LS_COLUMNS): LS_COLUMNS,
    len(BLS_COLUMNS): BLS_COLUMNS,
}


def load_catalog(catalog_dir, max_files=None, seed=42):
    """Load .result files from a catalog directory into a pandas DataFrame.

    Column names are auto-detected based on the number of columns (13 for
    Lomb-Scargle, 24 for BLS).

    Parameters
    ----------
    catalog_dir : str
        Path to a directory containing .result files (e.g. /ztf/catalogs/lomb_scargle).
    max_files : int, optional
        If set, load a random sample of this many files instead of all files.
        Useful for large catalogs with tens of thousands of files.
    seed : int
        Random seed for reproducible sampling (only used when *max_files* is set).

    Returns
    -------
    pandas.DataFrame
    """
    result_files = sorted(glob.glob(os.path.join(catalog_dir, '*.result')))
    if max_files is not None and max_files < len(result_files):
        rng = np.random.default_rng(seed)
        result_files = list(rng.choice(result_files, size=max_files, replace=False))

    # Auto-detect column schema from the first file
    col_names = None
    for fpath in result_files:
        try:
            first = pd.read_csv(fpath, sep=r'\s+', header=None, nrows=1)
            col_names = _COL_SCHEMAS.get(first.shape[1])
            break
        except Exception:
            continue
    if col_names is None:
        return pd.DataFrame()

    frames = []
    for fpath in result_files:
        try:
            df = pd.read_csv(fpath, sep=r'\s+', header=None, names=col_names)
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame(columns=col_names)
    return pd.concat(frames, ignore_index=True)


def _significance_col(df):
    """Return the name of the significance/power column in a catalog DataFrame."""
    if 'significance' in df.columns:
        return 'significance'
    if 'bls_power' in df.columns:
        return 'bls_power'
    raise ValueError("DataFrame has no 'significance' or 'bls_power' column")


def filter_candidates(df, min_significance=0, min_period=0, max_period=np.inf):
    """Apply period and significance cuts to a catalog DataFrame.

    Works with both LS catalogs (uses ``significance`` column) and BLS
    catalogs (uses ``bls_power`` column).

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`load_catalog`.
    min_significance : float
        Keep rows with significance/power > this value.
    min_period : float
        Keep rows with period > this value (days).
    max_period : float
        Keep rows with period < this value (days).

    Returns
    -------
    pandas.DataFrame
        Filtered copy.
    """
    sig_col = _significance_col(df)
    mask = (
        (df[sig_col] > min_significance)
        & (df['period'] > min_period)
        & (df['period'] < max_period)
    )
    return df.loc[mask].copy()


def _find_ps_id(ra, dec, field_ccd_quads):
    """Look up the Pan-STARRS ID of the nearest source to (ra, dec) in the matchfiles.

    Searches the r-band matchfile first (most likely to exist), then g, then i.
    Returns (ps_id, offset_arcsec) or raises ValueError if no matchfile is found.
    """
    for suffix in ['zr', 'zg', 'zi']:
        for field, ccd, quad in field_ccd_quads:
            fname = (f'{_MATCHFILE_DIR}/{int(field):04d}/'
                     f'data_{int(field):04d}_{int(ccd):02d}_{int(quad):1d}_{suffix}.h5')
            if not os.path.exists(fname):
                continue
            try:
                f = h5py.File(fname, 'r')
                src_ra = np.array(f['data']['sources']['ra'])
                src_dec = np.array(f['data']['sources']['decl'])
                cos_dec = np.cos(np.deg2rad(dec))
                dist = np.sqrt((src_ra - ra)**2 + ((src_dec - dec) * cos_dec)**2)
                nearest = np.argmin(dist)
                ps_id = f['data']['sources'][nearest]['gaia_id']
                offset = dist[nearest] * 3600.0  # degrees → arcsec
                f.close()
                return ps_id, offset
            except Exception:
                continue
    raise ValueError(f'No matchfile found for (ra={ra}, dec={dec})')


def lookup_period(ra=None, dec=None, ps_id=None,
                  catalog_dir='/ztf/catalogs/box_least_squares'):
    """Look up the catalog period for a source.

    You can identify the source by *either* (ra, dec) *or* ps_id (or both).
    Searches the .result files in the field(s) covering the source.

    Parameters
    ----------
    ra, dec : float, optional
        Source coordinates in degrees.  Required if *ps_id* is not given.
    ps_id : float or int, optional
        Pan-STARRS source ID.  If given without (ra, dec), the function
        searches for this ID in every .result file in *catalog_dir* that
        covers the field — this requires ra/dec, so at least coordinates
        must be provided.
    catalog_dir : str
        Path to the catalog directory (default: BLS catalog).

    Returns
    -------
    dict with keys:
        ``'period'`` (float, days), ``'ps_id'`` (float),
        ``'significance'`` (float — LS significance or BLS power),
        ``'offset_arcsec'`` (float — match distance; 0 if matched by ps_id).
    Raises ValueError if no match is found.
    """
    if ra is None or dec is None:
        raise ValueError('ra and dec are required for field lookup')

    field_ccd_quads = get_field_id(ra, dec, _fieldno, _field_ra, _field_dec)
    if not field_ccd_quads:
        raise ValueError(f'No ZTF field covers (ra={ra}, dec={dec})')

    # Auto-detect column schema from the first available file
    col_names = None
    for field, ccd, quad in field_ccd_quads:
        fname = os.path.join(catalog_dir,
                             f'{int(field):04d}_{int(ccd):02d}_{int(quad)}.result')
        if not os.path.exists(fname):
            continue
        try:
            first = pd.read_csv(fname, sep=r'\s+', header=None, nrows=1)
            col_names = _COL_SCHEMAS.get(first.shape[1])
            break
        except Exception:
            continue
    if col_names is None:
        raise ValueError('No .result files found for this position')

    best_match = None
    best_dist = 999.0

    for field, ccd, quad in field_ccd_quads:
        fname = os.path.join(catalog_dir,
                             f'{int(field):04d}_{int(ccd):02d}_{int(quad)}.result')
        if not os.path.exists(fname):
            continue
        try:
            df = pd.read_csv(fname, sep=r'\s+', header=None, names=col_names)
        except Exception:
            continue

        if ps_id is not None:
            hit = df.loc[df['ps_id'] == ps_id]
            if len(hit) > 0:
                row = hit.iloc[0]
                sig_col = _significance_col(df)
                return {
                    'period': row['period'],
                    'ps_id': row['ps_id'],
                    'significance': row[sig_col],
                    'offset_arcsec': 0.0,
                }

        # Positional match
        cos_dec = np.cos(np.deg2rad(dec))
        dist = np.sqrt((df['ra'] - ra)**2 + ((df['dec'] - dec) * cos_dec)**2)
        idx = dist.idxmin()
        if dist[idx] < best_dist:
            best_dist = dist[idx]
            best_match = df.loc[idx]

    if best_match is None or best_dist > 2.0 / 3600.0:  # 2 arcsec limit
        raise ValueError(f'No catalog match within 2" of (ra={ra}, dec={dec})')

    sig_col = _significance_col(pd.DataFrame([best_match]))
    return {
        'period': best_match['period'],
        'ps_id': best_match['ps_id'],
        'significance': best_match[sig_col],
        'offset_arcsec': best_dist * 3600.0,
    }


def get_lightcurve(ra=None, dec=None, ps_id=None, filters='gri'):
    """Extract a ZTF light curve.  The simplest way to get data for a source.

    You can call this with **just (ra, dec)** — the Pan-STARRS ID will be
    looked up automatically from the matchfiles.  Or pass all three if you
    already know the ps_id.

    Parameters
    ----------
    ra, dec : float
        Source coordinates in degrees.  Always required.
    ps_id : float or int, optional
        Pan-STARRS source ID.  If omitted, the nearest source in the
        matchfile is used (matched by position).
    filters : str
        Which ZTF filters to extract (subset of ``'gri'``).

    Returns
    -------
    dict
        Keyed by filter letter ('g', 'r', 'i'). Each value is a dict with
        keys 'time', 'flux', 'flux_err' (numpy arrays).
    """
    if ra is None or dec is None:
        raise ValueError('ra and dec are required')

    field_ccd_quads = get_field_id(ra, dec, _fieldno, _field_ra, _field_dec)
    if not field_ccd_quads:
        return {}

    # Auto-detect ps_id from matchfile if not provided
    if ps_id is None:
        ps_id, offset = _find_ps_id(ra, dec, field_ccd_quads)

    lcs = {}
    for filt_name in filters:
        if filt_name not in _FILTERS:
            continue
        result = _extract_filter(
            ps_id, field_ccd_quads, _FILTERS[filt_name]['suffix'], _MATCHFILE_DIR)
        if result is None:
            continue
        t, y, dy = result
        bjd = _bjd_convert(t, ra, dec)
        lcs[filt_name] = {
            'time': bjd.value,
            'flux': y,
            'flux_err': dy,
        }
    return lcs


def phase_fold(times, period):
    """Return phases in [0, 1) for the given period.

    Parameters
    ----------
    times : array-like
        Timestamps (any unit, as long as consistent with *period*).
    period : float
        Folding period in the same units as *times*.

    Returns
    -------
    numpy.ndarray
        Phases in [0, 1).
    """
    times = np.asarray(times)
    return ((times - times.min()) % period) / period


def weighted_bin(phases, flux, flux_err, n_bins=50):
    """Weighted-average binning of phase-folded data.

    Parameters
    ----------
    phases : array-like
        Phase values in [0, 1).
    flux, flux_err : array-like
        Flux and flux uncertainties.
    n_bins : int
        Number of phase bins.

    Returns
    -------
    (bin_centers, binned_flux, binned_err) : tuple of numpy arrays
    """
    phases = np.asarray(phases)
    flux = np.asarray(flux)
    flux_err = np.asarray(flux_err)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    binned_flux = np.full(n_bins, np.nan)
    binned_err = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (phases >= bin_edges[i]) & (phases < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        w = 1.0 / flux_err[mask] ** 2
        binned_flux[i] = np.sum(flux[mask] * w) / np.sum(w)
        binned_err[i] = np.sqrt(1.0 / np.sum(w))

    ok = np.isfinite(binned_flux)
    return centers[ok], binned_flux[ok], binned_err[ok]


def plot_lightcurve(lc_dict, period=None, n_phase_bins=50, title=None):
    """Make a diagnostic plot of a multi-filter light curve.

    Parameters
    ----------
    lc_dict : dict
        Output of :func:`get_lightcurve`.  Keys are filter letters, values are
        dicts with 'time', 'flux', 'flux_err'.
    period : float, optional
        If provided, adds a phase-folded row with weighted binning.
    n_phase_bins : int
        Number of bins for phase folding.
    title : str, optional
        Figure super-title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    filt_names = sorted(lc_dict.keys())
    n_filt = len(filt_names)
    if n_filt == 0:
        raise ValueError("lc_dict is empty — no filters to plot")

    n_rows = 2 if period is not None else 1
    fig, axes = plt.subplots(n_rows, n_filt, figsize=(7 * n_filt, 4 * n_rows),
                             squeeze=False)

    for col, filt_name in enumerate(filt_names):
        lc = lc_dict[filt_name]
        color = _FILTERS.get(filt_name, {}).get('color', 'black')

        # Top row: raw light curve
        axes[0, col].errorbar(lc['time'], lc['flux'], lc['flux_err'],
                              ls=' ', marker='.', ms=2, color=color, alpha=0.5)
        axes[0, col].set_xlabel('Time (BJD_TCB)')
        axes[0, col].set_ylabel('Flux')
        axes[0, col].set_title(f'{filt_name}-band raw')

        # Bottom row: phase-folded binned
        if period is not None:
            ph = phase_fold(lc['time'], period)
            bin_ph, bin_f, bin_e = weighted_bin(ph, lc['flux'], lc['flux_err'],
                                                n_phase_bins)
            for offset in range(3):
                axes[1, col].errorbar(bin_ph + offset, bin_f, bin_e,
                                      ls='-', marker='.', color=color, alpha=0.8)
            axes[1, col].set_xlabel('Phase')
            axes[1, col].set_ylabel('Flux')
            axes[1, col].set_title(f'{filt_name}-band folded (P = {period:.6f} d)')
            axes[1, col].set_xlim(0, 3)

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig
