""" Support for wrangling data and organizing it.

This module contains various helper functions for accessing, converting, and manipulating data used in SEDI.

Updated: 12.22.2021
Author: a. stein

"""

import xarray as xr
import pandas as pd
import numpy as np
import rioxarray

def clip_da_by_geometry(data:xr.DataArray, geometry:pd.Series, crs, path='raster.tif', time=None, save_raster=True):
    """Clip DataArray by GeoDataFrame.

    Using a DataArray with lat/lon coordinates, this function 
    creates a raster tif file that can then be manipulated and
    clipped to the area described by the GeoDataFrame.

    Parameters
    ----------
    data: xr.DataArray
        Contains a single variable defined by at least lat-lon
        coordinates. If also defined by time, provide another
        xr.DataArray in `time` to correctly set the variable
        later.
    geometery: polygon shape
        Contains the area to clip out
    crs
        The crs to use, should match geometery given.
    path: str (optional)
        Where to save the raster file created in the process.
        If none given, saves in current directory as
        'raster.tif'. Make certain to specify as a `.tif`.
        If you do not want this file to remain in your
        directory, set save_raster to False.
    time: xr.DataArray (optional)
        Describes the time coordinate of 
    save_raster: boolean (optional)
        Whether to retain the raster file created in the process.
        Defaults as True to retain the file. Setting it to
        False will delete the file created without impacted
        the returned DataArray.

    Returns
    -------
    xr.DataArray
        Clipped data to geometry. In creating a raster, 
    
    """
    try:
        os.remove(path)
    except:
        pass
    data.rio.to_raster(path)
    raster = rioxarray.open_rasterio(path, masked=True)

    raster.rio.write_crs(crs, inplace=True)
    clipped_raster = raster.rio.clip(geometry, crs)

    # fix naming
    clipped_raster = clipped_raster.rename({'y':'lat', 'x':'lon'})
    if isinstance(time, xr.DataArray):
        clipped_raster['band'] = time.values
        clipped_raster = clipped_raster.rename({'band':time.name})

    if not save_raster:
        os.remove(path)
        
    return clipped_raster

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

