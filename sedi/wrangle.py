""" Support for wrangling data and organizing it.

This module contains various helper functions for accessing, converting, and manipulating data used in SEDI.

Updated: 12.22.2021
Author: a. stein

"""

import xarray as xr
import pandas as pd
import numpy as np
import rioxarray
from shapely.geometry import mapping
import geopandas as gpd


def cunnane_empircal_cdf(data):
    """Creates an empircal cdf based on cunnane positions.

    Paramters
    ---------
    data: array-like
        What to create the cdf off of. Purely values.
    
    Returns
    -------
    values, positions
        The values from data sorted and plottable with
        positions to create a cunnane cdf.
    """

    n = len(data)
    sorted = np.sort(data)
    pos = [(i-0.4)/(n+0.2) for i in range(n)]

    return sorted, pos

def da_summary_stats(da:xr.DataArray, stats, groupby="time.month"):
    """Compute summary statistics for da.

    Parameters
    ----------
    da : xr.DataArray
        Data to compute stats on.
    stats : str or list-like
        What stats to compute from the following options:
            'mean' - computes the mean over groupby
            'median' - computes the median over groupby
            'std' - computes the standard deviation over groupby
            'max' - computes the maximum over groupby
            'min' - computes the minimum over groupby
        This is case-insensitive but must be one of the above options.
        You can specify as just one of them (such as stats='mean'), or
        provide them as a list-like object that is iterable. The order
        provided in stats does not change the order of stats provided
        in the returned object. Any misspelled specifications will
        simply not be computed.
    groupby : xr.DataArray.groupby suitable
        How to group the values in da, defaults as 'time.month'.

    Returns
    -------
    list or xr.DataArray
        List of xr.DataArray objects based on what is specified in stats
        in the following order: [mean, median, std, max, min]. If any
        of the five stats was not specified, then it is not included in the
        return list, but the order follows the same scheme excluding that
        statistic. If only one value was specified in stats, then returned
        DataArray is not wrapped in a list and can be accessed without
        unwrapping.
    """

    # I'm going to want to check through a list, yet I want to
    # make it compatible for only providing the name of one and
    # not be silly by requiring a string wrapper, so let's put it
    # in a list if not already
    if isinstance(stats, str):
        stats = [stats]

    # do a quick swap around to get everything lowercase
    stats_temp = stats
    stats = []
    for stat in stats_temp:
        assert isinstance(stat, str)
        stats.append(stat.lower())
    stats_temp = None

    # group the dataarray by whatever is given
    da_grouped = da.groupby(groupby)
    
    # start collecting statistics that we want to compute
    computed_stats = []
    if 'mean' in stats:
        mean = da_grouped.mean()
        computed_stats.append(mean)
    if 'median' in stats:
        median = da_grouped.median()
        computed_stats.append(median)
    if 'std' in stats:
        std = da_grouped.std()
        computed_stats.append(std)
    if 'max' in stats:
        max = da_grouped.max()
        computed_stats.append(max)
    if 'min' in stats:
        min = da_grouped.min()
        computed_stats.append(min)

    # want to catch if there was a typo and nothing was actually computed
    if len(computed_stats)==0:
        raise Exception('Please revise your stats input to be one of the specified options.')
    # it's a little stilly to need to unwrap just one
    # dataarray if we only computed one thing, so will
    # just unwrap it here
    if len(computed_stats)==1:
        return computed_stats[0]
    else:
        return computed_stats

def clip_xarray(da:xr.DataArray, gdf:gpd.GeoDataFrame, drop=True, all_touched=True):
    """Wrapper for rio.clip.

    https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray.raster_array.RasterArray.clip

    This function will not alter either da or gdf, it creates a copy of da to
    prevent this specifically.

    Parameters
    ----------
    da : xr.DataArray
        Data to be clipped, must contain longitude and latitude as lon and lat.
    gdf : gpd.GeoDataFrame
        Contains region to clip da to. Must have a crs and geometry specified.
    drop : boolean, (optional)
        Whether to remove pixels clipped outside of the gdf geometry or retain
        them as masked. Defaults as True to remove pixels.
    all_touched : boolean, (optional)
        Whether to include pixels that touch the geometry (True) or just those
        who have centroids within the geometry (False). Defaults as True.
    
    Returns
    -------
    xr.DataArray
        da clipped to gdf geometry with the crs from gdf.
    
    """
    da = da.copy()
    if da.rio.crs != gdf.crs:
        raise Exception('da and gdf CRS do not match, please fix matching')

    clipped = da.rio.clip(gdf.geometry.apply(mapping), gdf.crs, drop=drop, all_touched=all_touched)

    # get rid of the copy we made
    da = None

    return clipped

def xarray_clip_combine_years(data_path:str, years, combine_var:str, clip_gdf:gpd.GeoDataFrame):
    """Load, clip, then combine several years of netcdf data.

    This came about as I was looking to combine several years of data that covered all of CONUS,
    but only needed data for Washington and Oregon. I kept using xr.open_mfdataset() but finding
    it struggling with the large amounts of data. So this function instead opens each file 
    individually, clips it to the desired region (using clip_xarray), then combining all
    the data after. Data is unloaded in-between each year to minimize active memory taken up.

    Parameters
    ----------
    data_path : str
        Location of files to be combined. It is expected that the files follow the same naming
        convention, only changing the year at the end so that they can be opened with the string
        f'{data_path}{year}.nc' for each year provided in years. For example, if you wanted to
        combined a bunch of precipitation files called 'precip_1979.nc', 'precip_1980.nc', ... etc.
        then you would provide the path to those files plus 'precip_' at the end of the string.
        It is expected that the netcdf files have spatial dimensions and a crs set that is 
        accessible by rioxarray.
    years : list-like, iterable
        What years to combine the data from. If you wanted to combine all data from 1979 to
        2022 for example, then np.arange(1979, 2023, 1) would satisfy this parameter.
    combine_var : str
       What variable in the netcdf files to combine by. Note this requires all of the files
       you are combining to use the same variable name.
    clip_gdf : gpd.GeoDataFrame
        Contains geometry to clip the netcdf files to and must posses the same crs as the
        netcdf files, otherwise an Exception will be thrown.

    Returns
    -------
    xr.Dataset
        Contains data clipped to clip_gdf for all years provided in a single Dataset.
    """
    all_data_list = []
    t = tqdm(years, total=len(years))
    for year in t:
        t.set_description(f'{year}')
        data_ds = xr.open_dataset(f'{data_path}{year}.nc')
        # double check matching CRS
        if data_ds.rio.crs != clip_gdf.crs:
            raise Exception(f'{data_path}{year}.nc does not match the CRS of clip_gdf')
        
        # clip and set crs for dataarray
        data_da = data_ds[combine_var]
        data_da = data_da.rio.write_crs(data_ds.rio.crs)
        clipped_data_da = clip_xarray(data_da, clip_gdf)

        # add it to our list for later combining
        all_data_list.append(clipped_data_da)

        # clean up some of our variables to free up storage
        data_da = None
        data_ds = None
        clipped_data_da = None

    all_data_ds = xr.combine_by_coords(all_data_list)

    return all_data_ds

