# ndrought README
a. stein 8.18.2023

The ndrought package, examining n dimensions of drought, contains functionality for wrangling and comparing various forms of drought.

## compare & compare_xy
Used for comparing one drought metric to another directly. Includes functions for temporal pairing, categorizing drought metrics in USDM categories, and r correlations. `compare` assumes lat-lon coordinates in the dataset while `compare_xy` assumes x-y coordinates in the dataset.

## drought_network
This module contains the DroughtNetwork and its EventNodes that are used to construct events, threads, and networks.

## plotting
Various plots used throughout the development process. Most commonly used function will likely be `usdm_cmap`, which produces a colormap based on the USDM categories.

## wrangle
The main computational engine of ndrought. This module contains various helper functions for accessing, converting, and manipulating data used in ndrought.