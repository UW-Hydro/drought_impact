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

    The USDM cuts off receiving data on Tuesday mornings before eventually
    publishing their maps on Thusdays. This function finds which date is
    closest to the cutoff date for the USDM (that Tuesday) and pairs it
    with that date as the most recent data that could have been considered
    in the USDM. Note that this was developed for SPI that has a greater
    frequency than USDM and should be double checked before using a measure
    with a lower frequency.

    WARNING: There is not currently a catch if there are too many dates provided,
    (such as when the other date range exceeds the USDM date range)

    Parameters
    ----------
    usdm_dates: DateTimeIndex
        Array of dates from USDM measurements.
    other_dates: DateTimeIndex
        Array of dates to pair to USDM dates.
    other_name: str
        Name of the other dates.

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

    # setup and grab the day of the week
    pair_dates = pd.DataFrame(pd.Series(other_dates, name=other_name))
    pair_dates['DoW'] = other_dates.dayofweek
    
    # figure out how many days till the next Tuesday for USDM cutoff
    dtt_dict = {0:1, 1:0, 2:6, 3:5, 4:4, 5:3, 6:2}
    pair_dates['DtTue'] = [dtt_dict[day] for day in pair_dates['DoW'].values]

    # figure out if that is the closest to the USDM cutoff Tuesday to be used
    pair_dates['Pull'] = np.zeros(len(pair_dates))

    dtt_all = pair_dates['DtTue'].values
    for i, dtt_i in enumerate(dtt_all):
        if i < len(pair_dates)-1:
            if dtt_i < dtt_all[i+1]:
                pair_dates['Pull'].iloc[i] = 1

    # now select out the dates we want
    pair_dates = pair_dates[pair_dates['Pull'] == 1]
    # and match them with their USDM dates
    pair_dates['USDM Date'] = usdm_dates
    # reset the index
    pair_dates = pair_dates.reset_index()
    # and make sure to drop pandas trying to preserve the old index
    pair_dates = pair_dates.drop(columns='index')

    return pair_dates[[other_name, 'USDM Date']]

