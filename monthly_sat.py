

#!/usr/bin/env python3
"""
Seasonal Arctic LST climatology + Mann-Kendall trend analysis
2008–2020 MetOp-A AVHRR DAY observations

Outputs:
--------
1. combined_monthly_day_2008_2020.nc
2. Seasonal climatology / min / max figures
3. Seasonal trend NetCDF files
4. Seasonal p-value NetCDF files

Author: Your Name
"""

import os
import warnings

import numpy as np
import xarray as xr
import rioxarray
import pymannkendall as mk

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms

import cartopy.crs as ccrs

from dask.diagnostics import ProgressBar

warnings.filterwarnings("ignore")

# =============================================================================
# SETTINGS
# =============================================================================

DATA_DIR = "/mnt/data7/nfs4/avh_lst/sdupuis/EUSTACE/All_Arctic"
MASK_PATH = "/mnt/data7/nfs4/avh_lst/sdupuis/auxiliary/watermask50.tif"

OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = range(2008, 2021)

# =============================================================================
# LOAD MONTHLY DATA
# =============================================================================

print("Loading datasets...")

results = {}

for year in YEARS:

    print(f"Processing {year}")

    pattern = (
        f"{DATA_DIR}/{year}/AVMEA/"
        f"LST_AVMEA_All_Arctic__v.11.0__*DAY.nc"
    )

    ds = xr.open_mfdataset(
        pattern,
        engine="netcdf4",
        combine="by_coords",
        parallel=True,
        chunks={"time": 1}
    )

    # Remove invalid temperatures
    clean_lst = ds["LST"].where(ds["LST"] > 110, np.nan)

    # Monthly mean
    monthly_mean = (
        clean_lst
        .resample(time="1MS")
        .mean()
    )

    results[year] = monthly_mean

# =============================================================================
# COMBINE ALL YEARS
# =============================================================================

print("Combining all years...")

combined = xr.concat(
    [results[y] for y in YEARS],
    dim="time"
)

print(combined)

clouds = xr.open_dataset('data/ice_flc_percent_2008_2020_2deg.nc')
example = clouds['ice_flc_percent'].isel(year=1, month=1)
#interp
combined_interp = combined.coarsen(lat=40, lon=40, boundary="trim").mean()

# =============================================================================
# SAVE COMBINED FILE
# =============================================================================

combined_outfile = os.path.join(
    OUTPUT_DIR,
    "combined_monthly_day_2008_2020_coarse.nc"
)

print(f"Saving combined file:\n{combined_outfile}")

encoding = {
    "LST": {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32"
    }
}

#combined_interp.to_netcdf(combined_outfile,format="NETCDF4",encoding=encoding)

# =============================================================================
# SEASONAL CLIMATOLOGY 
# =============================================================================

print("Computing seasonal statistics...")

climatology = combined_interp.groupby("time.season").mean("time")


# =============================================================================
# WATER MASK
# =============================================================================

print("Loading water mask...")

water_mask = rioxarray.open_rasterio(MASK_PATH)

water_mask = water_mask.rename({'y':'lat', 'x':'lon'})
water_mask = water_mask.isel(lat=slice(None, None, -1))
water_interp = water_mask.interp(lat=example.coords['latitude'], lon=example.coords['longitude'], method='nearest')

water_mask_true = combined_interp.isel(time=0).copy(data=np.array(np.squeeze(water_interp)))
ls_mask = water_mask_true.where(water_mask_true>0, np.nan)

# Apply mask
clim_masked = climatology * ls_mask


# Load into memory
print("Loading climatology into memory...")

clim_loaded = clim_masked.load()


# =============================================================================
# POLAR PROJECTION PLOT
# =============================================================================

print("Creating polar projection figure...")

# values are strange

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
import cartopy.crs as ccrs

vmin = float(clim_loaded.min())
vmax = float(clim_loaded.max())

proj = ccrs.NorthPolarStereo()

# --- get seasons dynamically (important!) ---
seasons = list(clim_loaded['season'].values)

# --- create mosaic layout (assumes 4 seasons) ---
fig, axs = plt.subplot_mosaic(
    [['(a)', '(b)'], ['(c)', '(d)']],
    figsize=(12, 10),
    subplot_kw=dict(projection=proj)
)

labels = ['(a)', '(b)', '(c)', '(d)']

# --- circular boundary ---
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

# --- plotting loop ---
mappable = None

for i, (season, lab) in enumerate(zip(seasons, labels)):
    ax = axs[lab]

    da = clim_loaded.sel(season=season)

    mappable = da.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='coolwarm',
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False
    )

    # --- map styling ---
    ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(linewidth=0.3)

    ax.set_title(str(season), fontsize=14)

# --- panel labels ---
for lab, ax in axs.items():
    trans = mtransforms.ScaledTranslation(-10/72, 2/72, fig.dpi_scale_trans)
    ax.text(
        0.05, 0.95, lab,
        transform=ax.transAxes + trans,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='0.9', edgecolor='none')
    )

# --- make space at bottom ---
fig.subplots_adjust(bottom=0.12)

# --- add colorbar axis manually ---
cbar_ax = fig.add_axes([0.2, -0.05, 0.6, 0.02])  # [left, bottom, width, height]

cbar = fig.colorbar(
    mappable,
    cax=cbar_ax,
    orientation='horizontal'
)

cbar.set_label(' Mean LST [K]')

plt.tight_layout()
plt.savefig('meanstate_arctic_day.png', dpi=300)

plt.close()

# =============================================================================
# MANN-KENDALL FUNCTIONS
# =============================================================================

print("Preparing Mann-Kendall functions...")


def mk_test_slope(y):

    if np.count_nonzero(np.isnan(y)) > 9:
        return np.nan

    return mk.original_test(y).slope


def mk_test_p(y):

    if np.count_nonzero(np.isnan(y)) > 9:
        return np.nan

    return mk.original_test(y).p


def mann_kendall(data, dim):

    return xr.apply_ufunc(
        mk_test_slope,
        data,
        input_core_dims=[[dim]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[float],
    )


def mann_kendall_p(data, dim):

    return xr.apply_ufunc(
        mk_test_p,
        data,
        input_core_dims=[[dim]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[float],
    )


# =============================================================================
# SEASON DEFINITIONS
# =============================================================================

season_dict = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "fall": [9, 10, 11],
}

# =============================================================================
# TREND ANALYSIS
# =============================================================================

for season_name, months in season_dict.items():

    print(f"\nComputing trends for {season_name}...")

    season = combined.sel(
        time=combined.time.dt.month.isin(months)
    )

    # Fix DJF year assignment
    if season_name == "winter":

        season["time"] = (
            season["time"] +
            np.timedelta64(10000000000000000, "ns")
        )

    season_data = season.groupby("time.year").mean()

    season_masked = season_data * ls_mask

    season_masked = season_masked.chunk({
        "year": -1,
        "lon": 1000,
        "lat": 100
    })

    # -------------------------------------------------------------------------
    # Trend
    # -------------------------------------------------------------------------

    print(f"Computing slope ({season_name})")

    with ProgressBar():
        trend = mann_kendall(
            season_masked,
            dim="year"
        ).compute()

    trend = trend * ls_mask

    trend_ds = trend.to_dataset(name="trend")

    trend_outfile = os.path.join(
        OUTPUT_DIR,
        f"{season_name}_trend_DAY.nc"
    )

    trend_ds.to_netcdf(
        trend_outfile,
        format="NETCDF4"
    )

    # -------------------------------------------------------------------------
    # P-values
    # -------------------------------------------------------------------------

    print(f"Computing p-values ({season_name})")

    with ProgressBar():
        pval = mann_kendall_p(
            season_masked,
            dim="year"
        ).compute()

    pval = pval * ls_mask

    pval_ds = pval.to_dataset(name="p_val")

    pval_outfile = os.path.join(
        OUTPUT_DIR,
        f"{season_name}_p_val_DAY.nc"
    )

    pval_ds.to_netcdf(
        pval_outfile,
        format="NETCDF4"
    )

print("\nDone.")
