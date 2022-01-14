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