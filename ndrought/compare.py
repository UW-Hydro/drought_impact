import xarray as xr

def spi_to_usdmcat(spi_da:xr.DataArray):
    """Categorizes SPI based on USDM categories.

    Uses the mapping scheme presented by USDM (https://droughtmonitor.unl.edu/About/AbouttheData/DroughtClassification.aspx)
    Where Neutral is -1, D0 is 0, D1 is 1, D2, is 2, D3 is 3, and D4 is 4.

    Parameters
    ----------
    spi_da : xr.DataArray
        Contains SPI values.
    
    Returns
    -------
    xr.DataArray
        DataArray formatted the same as spi_da but using USDM categories.

    """

    # make sure we don't overwrite the original
    spi_da_copy = spi_da.copy()
    # can only do boolean indexing on the underlying array
    spi_da_vals = spi_da.values
    # get a copy to make sure reassignment isn't compounding
    spi_da_origin = spi_da_vals.copy()

    # assign neutral
    spi_da_vals[spi_da_origin > -0.5] = -1
    # assign D0
    spi_da_vals[(spi_da_origin <= -0.5)&(spi_da_origin > -0.8)] = 0
    # assign D1
    spi_da_vals[(spi_da_origin <= -0.8)&(spi_da_origin > -1.3)] = 1
    # assign D2
    spi_da_vals[(spi_da_origin <= -1.3)&(spi_da_origin > -1.6)] = 2
    # assign D3
    spi_da_vals[(spi_da_origin <= -1.6)&(spi_da_origin > -2.0)] = 3
    # assign D4
    spi_da_vals[(spi_da_origin <= -2.0)] = 4

    # put them back into the dataarray
    spi_da_copy.loc[:,:] = spi_da_vals

    return spi_da_copy