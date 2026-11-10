# Natural vegetation response to recurrent droughts, northern Algeria (2000–2025)

Code for the manuscript "Réponse de la végétation naturelle aux sécheresses récurrentes et prolongées dans le nord de l'Algérie (2000–2025)".

## Contents (`scripts/`)
- Climate: `chirps_read.py`, `chirps_spi.py`, `era5_spei_qgis.py`, `era5_soil_qgis.py`
- Annual regressions: `h1_chirps.py` (external index via env var `SPI_CSV`), `soil_index.py`, `sat_soil_index.py`
- Soil moisture and land-surface temperature: `mechanism_monthly_cells.py`, `mechanism_monthly_cells_run.py`, `smap_fix.py`, `lst_from_hdf.py`, `lst_appeears_qgis.py`
- Land cover: `esri_cover_stability.py`, `cover_counts.py`
- Main analysis and checks: `analyze.py`, `seasonal_models.py`, `episode_contrasts.py`, `within_season_slope.py`, `annual_hier_baseline.py`, `aqua_vs_terra.py`, `area_metric.py`
- Figures: `heatmap.py`, `maps.py`, `timescale_fig.py`, `aqua_figs.py`, `fig7.py`

## Notes
- Python 3 with GDAL (run in the OSGeo4W shell / QGIS for NetCDF and HDF inputs), numpy, pandas.
- No script contains credentials. Download scripts prompt for an Earthdata token or password at run time.
- Input data are third-party products and are not redistributed here (TerraClimate, CHIRPS v3.0, ERA5-Land, MOD13Q1, MYD13Q1, MYD11A2 v061, MCD64A1, GLC_FCS30D, ESRI LULC 10 m, Dynamic World, ESA CCI SM COMBINED v09.2, SMAP SPL3SMP_E v006, SoilGrids).
- Paths inside scripts point to the author's folders and must be edited before use.

Licence: MIT. Archived release: code https://doi.org/10.5281/zenodo.24110454 (v1.0.0); data https://doi.org/10.5281/zenodo.24110516.
