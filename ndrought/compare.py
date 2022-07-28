import xarray as xr
import pandas as pd
import numpy as np

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

def spi_to_usdmcat_multtime(spi_ds:xr.Dataset):
    """Categorizes SPI based on USDM categories for multiple times.
    
    See spi_to_usdmcat for further documentation.
    
    Parameters
    ----------
    spi_ds : xr.Dataset
        SPI at multiple time values as the coordinate 'day'.
    
    Returns
    -------
    xr.Dataset
        SPI categorized by spi_to_usdmcat.
    """
    
    return spi_to_usdmcat(xr.concat([spi_ds.sel(day=day) for day in spi_ds['day'].values], dim='day'))

def pair_to_usdm_date(usdm_dates:pd.DatetimeIndex, other_dates:pd.DatetimeIndex, other_name:str, realign=False):
    """Pairs dates from one metric to USDM dates for comparison.

    This function finds which date is closest to the cutoff date for the USDM 
    (Tuesday) and pairs it with that date as the most recent data that could 
    have been considered in the USDM. Note that this was developed for SPI 
    that has a greater frequency than USDM and should be double checked before 
    using a measure with a lower frequency.

    WARNING: The current catch for too-many dates provided is experimental.

    NOTE: The USDM considers data up UNTIL the Tuesday morning before maps are
    released on Thursday. A more robust analysis should determine if there is a
    time cuttoff that would exclude maps on Tuesday as well as integrate any map
    made since the previous Wednesday that could also influence the USDM map.
    Adding this feature has not yet been done but should before drawing in-depth
    conclusions as currently only the most recent map is considered.

    Parameters
    ----------
    usdm_dates: DateTimeIndex
        Array of dates from USDM measurements.
    other_dates: DateTimeIndex
        Array of dates to pair to USDM dates.
    other_name: str
        Name of the other dates.
    realign: boolean, (optional)
        Whether to automatically clip dates to ensure proper pairing,
        defaults as False.

    Returns
    -------
    pd.DataFrame
        DataFrame where each row pairs an other_date to a usdm_date.
    """

    # check if times are too far out of alignment
    if usdm_dates[-1] - pd.Timedelta(days=7) > other_dates[-1]:
        if realign:
            usdm_dates = usdm_dates[usdm_dates <= other_dates[-1] + pd.Timedelta(days=7)]
        else:
            raise Exception('USDM dates extends more than a week beyond other dates, resulting in an inability to pair. Please adjust USDM dates accordingly or set realign=True to (experimentally) automatically correct.')
    if other_dates[0] + pd.Timedelta(days=7) < usdm_dates[-1]:
        if realign:
            other_dates = other_dates[other_dates >= usdm_dates[0] - pd.Timedelta(days=7)]
        else:
            raise Exception('Other dates extends more than a week prior to USDM dates, resulting in an inability to pair. Please adjust other dates accordingly or set realign=True to (experimentally) automatically correct')

    # put this into a DataFrame
    pair_dates = pd.DataFrame(pd.Series(other_dates, name=other_name))
    # add the column for USDM dates
    pair_dates['USDM Date'] = np.nan * np.zeros(len(pair_dates[other_name]))

    # now we need to iterate through and find which other dates are the closest
    # to the USDM date and pair them
    i = 0
    for date in usdm_dates:
            while i < len(other_dates) and other_dates[i] <= date:
                i += 1
            pair_dates['USDM Date'].iloc[i-1] = date

    # now drop the dates that did not get chosen
    pair_dates = pair_dates.dropna('index')
    # reset the index
    pair_dates = pair_dates.reset_index()
    # and make sure to drop pandas trying to preserve the old index
    pair_dates = pair_dates.drop(columns='index')

    return pair_dates

def dm_to_usdmcat(da:xr.DataArray):
    """Categorizes drought measure based on USDM categories.

    Uses the mapping scheme presented by USDM (https://droughtmonitor.unl.edu/About/AbouttheData/DroughtClassification.aspx)
    Where Neutral is -1, D0 is 0, D1 is 1, D2, is 2, D3 is 3, and D4 is 4.

    Parameters
    ----------
    da : xr.DataArray
        Contains SPI values.
    
    Returns
    -------
    xr.DataArray
        DataArray formatted the same as da but using USDM categories.

    """

    # make sure we don't overwrite the original
    da_copy = da.copy()
    # can only do boolean indexing on the underlying array
    da_vals = da.values
    da_vals_nonnan = da_vals[np.isnan(da_vals) == False]
    # calculate percentiles
    (p30, p20, p10, p5, p2) = np.percentile(da_vals_nonnan.ravel(), [30, 20, 10, 5, 2])
    # get a copy to make sure reassignment isn't compounding
    da_origin = da_vals.copy()

    # assign neutral
    da_vals[da_origin > p30] = -1
    # assign D0
    da_vals[(da_origin <= p30)&(da_origin > p20)] = 0
    # assign D1
    da_vals[(da_origin <= p20)&(da_origin > p10)] = 1
    # assign D2
    da_vals[(da_origin <= p10)&(da_origin > p5)] = 2
    # assign D3
    da_vals[(da_origin <= p5)&(da_origin > p2)] = 3
    # assign D4
    da_vals[(da_origin <= p2)] = 4

    # put them back into the dataarray
    da_copy.loc[:,:] = da_vals

    return da_copy

def dm_to_usdmcat_multtime(ds:xr.Dataset):
    """Categorizes drought measure based on USDM categories for multiple times.
    
    See dm_to_usdmcat for further documentation.
    
    Parameters
    ----------
    spi_ds : xr.Dataset
        SPI at multiple time values as the coordinate 'day'.
    
    Returns
    -------
    xr.Dataset
        Drought measure categorized by dm_to_usdmcat.
    """
    if 'day' in ds.coords:
        return dm_to_usdmcat(xr.concat([ds.sel(day=day) for day in ds['day'].values], dim='day'))
    elif 'time' in ds.coords:
        return dm_to_usdmcat(xr.concat([ds.sel(time=t) for t in ds['time'].values], dim='time'))
    else:
        raise Exception('Time dimension not day or time')

  