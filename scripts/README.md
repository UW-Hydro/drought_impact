# Scripts
a. stein 8.23.2022

## **`coarsen_thresh_USDM.py`**
Coarsen USDM by a factor of 10.

## **`compute_drought_tracks_config.py`**
Compute drought tracks using a configuration file so that it can be ran through a loop with different parameters easier (see `run_thresh_exp_*`).

## **`compute_drought_tracks_dask_paired.py`**
`compute_drought_tracks.py` using USDM and SPI paired data for USDM, SPI30d, and SPI180d. Uses dask and hard-coded.

## **`compute_drought_tracks_dask_simple_d3.py`**
`compute_drought_tracks.py` for only USDM, SPI30d, and SPI180d at a drought threshold of D3 instead of D1. Uses dask and hard-coded.

## **`compute_drought_tracks_dask_simple.py`**
`compute_drought_tracks.py` for only USDM, SPI30d, and SPI180d. Uses dask and hard-coded.

## **`compute_drought_tracks_dask.py`**
`compute_drought_tracks.py` except with dask parallelism hard-coded.

## **`compute_drought_tracks.py`**
Hard-coded compute drought tracks for USDM, SPI, SPEI, EDDI, PDSI, and GRACE.

## **`compute_nexus_af_df_105w_simple.py`**
Computes area fraction (AF) and disagreement fraction (DF) among USDM and SPI for the data that uses everything west of longitude 105 on CONUS. Thresholds are hard-coded in and need to be adjusted in the script itself to change.

## **`compute_nexus_af_df_105w.py`**
`compute_nexus_af_df_105w_simple.py` except ran for SPEI, EDDI, PDSI, and GRACE as well as SPI and USDM.

## **`compute_nexus_af_df.py`**
Computes area fraction (AF) and disagreement fraction (DF) among USDM, SPI, SPEI, EDDI, PDSI, and GRACE for Washington state.

## **`compute_nexus_lagged_r_105w.py`**
Computes lagged r correlation between timeseries for USDM, SPI, SPEI, EDDI, PDSI, and GRACE for the data that uses everything west of longitude 105 on CONUS.

## **`compute_nexus_lagged_r.py`**
Computes lagged r correlation between timeseries for USDM, SPI, SPEI, EDDI, PDSI, and GRACE for Washington state.

## **`create_track_tree.py`**
Takes drought tracks from a drought track dictionary (dtd) and makes network trees for analysis.

## **`graph_dnet.py`**
An attempt to plot USDM and PDSI linked together by overlapping nodes.

## **`run_thresh_exp_spi30d.py`**
Run threshold experiment for spi30d.

## **`run_thresh_exp_spi180d.py`**
Run threshold experiment for spi180d.

## **`run_thresh_exp_usdm.py`**
Run threshold experiment for USDM