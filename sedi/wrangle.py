""" Support for wrangling data and organizing it.

This module contains various helper functions for accessing, converting, and manipulating data used in SEDI.

Updated: 12.22.2021
Author: a. stein

"""

import xarray as xr
import pandas as pd
import rioxarray

def clip_da_by_geometry(data:xr.DataArray, geometry:pd.Series, crs, path='raster.tif', time=None):
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
        'raster.tif'. Make certain to specify as a `.tif`
    time: xr.DataArray (optional)
        Describes the time coordinate of 

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
        
    return clipped_raster
    