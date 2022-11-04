import xarray as xr
import pandas as pd
import numpy as np
import multiprocessing as mp

import warnings
warnings.filterwarnings("ignore")

from tqdm.autonotebook import tqdm

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

def dm_to_usdmcat(da:xr.DataArray, percentiles=None):
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
    if percentiles is None:
        (p30, p20, p10, p5, p2) = np.percentile(da_vals_nonnan.ravel(), [30, 20, 10, 5, 2])
    else:
        if len(percentiles) != 5:
            raise Exception('percentiles should specify thresholds for neutral, D0, D1, D2, D3, and D4')
        (p30, p20, p10, p5, p2) = percentiles
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

def dm_to_usdmcat_multtime(ds:xr.Dataset, percentiles=None):
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
        return dm_to_usdmcat(xr.concat([ds.sel(day=day) for day in ds['day'].values], dim='day'), percentiles=percentiles)
    elif 'time' in ds.coords:
        return dm_to_usdmcat(xr.concat([ds.sel(time=t) for t in ds['time'].values], dim='time'), percentiles=percentiles)
    else:
        raise Exception('Time dimension not day or time')

def pair_dates(dates_a:pd.DatetimeIndex, dates_b:pd.DatetimeIndex, dates_a_name:str, dates_b_name:str, method='last-b', realign=False):
    """Pairs dates between two metrics for comparison.
    
    Note that this was developed for SPI and USDM and should be double checked.

    WARNING: The current catch for too-many dates provided is experimental.

    Parameters
    ----------
    dates_a: DateTimeIndex
    dates_b: DateTimeIndex
    dates_a_name: str
        Name of dates_a.
    dates_b_name: str
        Name of dates_b.
    method: str
        How to pair dates between dates_a and dates_b. The following are
        currently supported:
        - last-a: match dates_b to the last dates_a available. Use if there
            is reason to believe that measure A informs measure B
        - last-b: match dates_a to the last dates_b available. Use if there
            is reason to believe that measure B informs measure A
        - nearest: dates are paired by their nearest neighbors, dropping any
            dates that are not chosen in the process. Use if there is no
            reason to believe A nor B inform each other and do not want
            an aggregate process

        To come:
        - cluster-a: match all dates_b to their nearest dates_a, which
            can result in multiple dates being assigned to a date in dates_a
            and an aggregation scheme should be used in the data. Use if
            believe B informs A and wanting to allow for multi-pairing.
        - cluster-b: match all dates_a to their nearest dates_b, which
            can result in multiple dates being assigned to a date in dates_b
            and an aggregation scheme should be used in the data. Use if
            believe A informs B and wanting to allow for multi-pairing.
        - cluster-nearest: each date is assigned to the nearest other date,
            being the only method to guarantee no dates are dropped in pairing.
            Use if there is no strong belief that one measure dominantly
            influences the other and wanting to allow for multi-pairing.
    realign: boolean, (optional)
        Whether to automatically clip dates to ensure proper pairing,
        defaults as False.

    Returns
    -------
    pd.DataFrame
        DataFrame where each row pairs an dates_b to a dates_a.
    """

    # check if times are too far out of alignment
    if dates_a[-1] - pd.Timedelta(days=7) > dates_b[-1]:
        if realign:
            dates_a = dates_a[dates_a <= dates_b[-1] + pd.Timedelta(days=7)]
        else:
            raise Exception('dates_a extends more than a week beyond dates_b, resulting in an inability to pair. Please adjust dates_a accordingly or set realign=True to (experimentally) automatically correct.')
    if dates_b[-1] - pd.Timedelta(days=7) > dates_a[-1]:
        if realign:
            dates_b = dates_b[dates_b <= dates_a[-1] + pd.Timedelta(days=7)]
        else:
            raise Exception('dates_b extends more than a week beyond dates_a, resulting in an inability to pair. Please adjust dates_b accordingly or set realign=True to (experimentally) automatically correct.')
    if dates_b[0] + pd.Timedelta(days=7) < dates_a[-1]:
        if realign:
            dates_b = dates_b[dates_b >= dates_a[0] - pd.Timedelta(days=7)]
        else:
            raise Exception('dates_b extends more than a week prior to dates_a, resulting in an inability to pair. Please adjust dates_b accordingly or set realign=True to (experimentally) automatically correct')
    if dates_a[0] + pd.Timedelta(days=7) < dates_b[-1]:
        if realign:
            dates_a = dates_a[dates_a >= dates_b[0] - pd.Timedelta(days=7)]
        else:
            raise Exception('dates_a extends more than a week prior to dates_b, resulting in an inability to pair. Please adjust dates_a accordingly or set realign=True to (experimentally) automatically correct')

    # now we need to iterate through and find which other dates are the closest
    # and pair them
    if method == 'last-b':
        pair_dates = pd.DataFrame(pd.Series(dates_b, name=dates_b_name))
        # add the column for dates_a dates
        pair_dates[dates_a_name] = np.nan * np.zeros(len(pair_dates[dates_b_name]))

        i = 0
        for date in dates_a:
            if date >= dates_b[i]:
                while i < len(dates_b)-1 and dates_b[i+1] <= date:
                    i += 1
                if not isinstance(pair_dates[dates_a_name].iloc[i], pd.Timestamp):
                    pair_dates[dates_a_name].iloc[i] = date
    elif method == 'last-a':
        pair_dates = pd.DataFrame(pd.Series(dates_a, name=dates_a_name))
        # add the column for dates_b dates
        pair_dates[dates_b_name] = np.nan * np.zeros(len(pair_dates[dates_a_name]))

        i = 0
        for date in dates_b:
            if date >= dates_a[i]:
                while i < len(dates_a)-1 and dates_a[i+1] <= date:
                    i += 1
                if not isinstance(pair_dates[dates_b_name].iloc[i], pd.Timestamp):
                    pair_dates[dates_b_name].iloc[i] = date
    elif method == 'nearest':
        i = 0
        j = 0
        pairs = []

        # go through and match each date in A with the nearest date in B
        while i < len(dates_a) and j < len(dates_b):
            current_difference = np.abs(dates_a[i] - dates_b[j])

            # if we've gotten to the end of the list, match with
            # it and reset our search
            if j+1 == len(dates_b):
                pairs.append((dates_a[i], dates_b[j]))
                i += 1
                j = 0
            # if there is an exact match or the found date is closer
            # than the following, consider it a match
            elif current_difference == pd.Timedelta('0 day') or current_difference < np.abs(dates_a[i] - dates_b[j+1]):
                pairs.append((dates_a[i], dates_b[j]))
                i += 1
            # keep looking
            else:
                j += 1

        # now need to trim duplicate pairings

        paired_b = np.array(pairs)[:, 1]
        remove_pairs = []

        for date in dates_b:
            found_pairs = np.where(date == paired_b)[0]
            # check if found more than one pairing
            if len(found_pairs > 1):
                # gather differences
                deltas = []
                for duplicate in found_pairs:
                    pairing = pairs[duplicate]
                    deltas.append(np.abs(pairing[0] - pairing[1]))
                # find closest
                minimum_index = np.argmin(np.array(deltas))
                minimum_pairing = pairs[found_pairs[minimum_index]]
            # collect which pairs to remove (removing now would mess
            # with the ordering of things)
            for duplicate in found_pairs:
                pairing = pairs[duplicate]
                if pairing != minimum_pairing:
                    remove_pairs.append(pairing)
        # finally remove unfavorable duplicates
        for pair in remove_pairs:
            pairs.remove(pair)
            
        pair_dates = pd.DataFrame(data=np.array(pairs), columns=[dates_a_name, dates_b_name])
            
    #elif method == 'cluster-a':
    #elif method == 'cluster-b':
    #elif method == 'cluster-nearest':
    else:
        raise Exception(f'{method} is not a supported method. Please visit the documentation for supported methods.')

    # now drop the dates that did not get chosen
    pair_dates = pair_dates.dropna('index')
    # reset the index
    pair_dates = pair_dates.reset_index()
    # and make sure to drop pandas trying to preserve the old index
    pair_dates = pair_dates.drop(columns='index')

    return pair_dates

def nan_corrcoef(x, y):
    """Computes correlation between x and y that can have nans.

    Parameters
    ----------
    x, y: array-like
    
    Returns
    -------
    r 
    """

    n = len(x)
    if n != len(y):
        raise Exception('x and y are not the same size')

    numer = n*np.sum(x*y) - (np.sum(x)*np.sum(y))
    denom = n*np.sum(x**2) - np.sum(x)**2
    denom *= n*np.sum(y**2) - np.sum(y)**2
    denom = np.sqrt(denom)

    return numer/denom

def lag_nan_corrcoef(x:xr.DataArray, y:xr.DataArray, lag:int, lag_step=1):
    """ Lagged correlation coefficients using nan_corrcoef.

    Parameters
    ----------
    x, y: xr.DataArray
    lag: int
        How many time steps to lag by
    lag_step, (optional): int
        Interval to lag time steps by, defaults 1.

    Returns
    -------
    lag, r
        Array of lags used and their corresponding r values.        
    """

    n = len(x)
    if n != len(y):
        raise Exception('x and y are not the same size')
    if lag > n:
        raise Exception('lag cannot exceed size of x or y')
    
    r_array = []

    # check if empty pixel
    if np.all(np.isnan(x)) or np.all(np.isnan(y)):
        r_array = np.nan*np.arange(-1*lag, lag+1, 1)
    else:
        for lag_i in np.arange(-lag, 0, lag_step):
            r_array.append(nan_corrcoef(x[:lag_i], y[-1*lag_i:]).values)
        r_array.append(nan_corrcoef(x, y).values)
        for lag_i in np.arange(1, lag+1, lag_step):
            r_array.append(nan_corrcoef(x[lag_i:], y[:-1*lag_i]).values)

    return np.arange(-1*lag, lag+1, 1), np.hstack(r_array)

# generalize compute_r !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def compute_r_pixel(args):
   
        # unpack our wrapped arguments
        pixel_x, pixel_y, lag = args

        __, r_pixel = lag_nan_corrcoef(pixel_x, pixel_y, lag=lag)

        return r_pixel

def compute_r_multi_mp(ds:xr.Dataset, pool:mp.Pool, x_vars=list, y_vars=list, lag=20, floor=-1):

    result_ds = xr.Dataset(
        coords=dict(
            lat=(["lat"], ds.lat.values),
            lon=(["lon"], ds.lon.values),
            lag=(["lag"], np.arange(-1*lag, lag+1))
        )
    )

    m = len(result_ds.lat)
    n = len(result_ds.lon)

    for x_var in x_vars:
        for y_var in y_vars:
            args = []
            t = tqdm(total=m*n, desc=f'Prepping {x_var} {y_var}')
            for lat in ds.lat.values:
                for lon in ds.lon.values:

                    pixel_ds = ds.sel(lat=lat, lon=lon).copy()

                    pixel_x = pixel_ds[x_var]
                    pixel_x[pixel_x < floor] = np.nan

                    pixel_y = pixel_ds[y_var]
                    pixel_y[pixel_y < floor] = np.nan

                    args.append((pixel_x, pixel_y, lag))

                    pixel_ds = None
                    pixel_x = None
                    pixel_y = None

                    t.update()

            results = pool.map(compute_r_pixel, tqdm(args, desc=f"Computing {x_var} {y_var} R"))
            print(f"Reshaping and storing {x_var} {y_var}")
            results_reshaped = np.array([results[(i-1)*n:i*n] for i in np.arange(1,m+1)])

            result_ds[f'{x_var}_{y_var}_r'] = (["lat", "lon", "lag"], results_reshaped)

    return result_ds    


def create_paired_ds(da_a:list, da_b:list, dates_a:pd.DataFrame, dates_b:pd.DataFrame, rescale=False):
    lat_a = da_a[0].lat
    lat_b = da_b[0].lat
    lon_a = da_a[0].lon
    lon_b = da_b[0].lon

    if len(lat_a) != len(lat_b) or len(lon_a) != len(lon_b):
        if rescale:
            if len(lat_a) > len(lat_b) or len(lon_a) > len(lon_b):
                da_a_rescaled = [da.rio.reproject_match(da_b[0]).rename({'x':'lon', 'y':'lat'}) for da in da_a]
                da_a = da_a_rescaled
            else:
                da_b_rescaled = [da.rio.reproject_match(da_a[0]).rename({'x':'lon', 'y':'lat'}) for da in da_b]
                da_b = da_b_rescaled
        else:
            raise Exception('Coordinates are different scales. Either fix or set rescale to True to enable auto upscaling.')
    elif np.any(lat_a != lat_b) or np.any(lon_a != lon_b):
        raise Exception('Coordinates do not match despite same size.')
        

    index = dates_a.index.values
    if np.any(index != dates_b.index.values):
        raise Exception('Dates are not paired to the same index.')


    paired_ds = xr.Dataset(
        coords=dict(
            index=(['index'], index),
            lat=(['lat'], da_a[0].lat.values),
            lon=(['lon'], da_a[0].lon.values),
        )
    )
    paired_ds['index'] = index
    
    for da in da_a:
        paired_ds[da.name] = xr.DataArray(
            da.values,
            dims = ['index', 'lat', 'lon'],
            coords=dict(
                index=index,
                lat=paired_ds.lat.values,
                lon=paired_ds.lon.values,
            )
    )
    
    
    paired_ds[f'{dates_a.name.upper()} Date'] = xr.DataArray(
        dates_a.values,
        dims=['index'],
        coords=dict(index=index)
    )

    for da in da_b:
        paired_ds[da.name] = xr.DataArray(
            da.values,
            dims=['index', 'lat', 'lon'],
            coords=dict(
                index=index,
                lat=paired_ds.lat.values,
                lon=paired_ds.lon.values,
            )
        )
    paired_ds[f'{dates_b.name.upper()} Date'] = xr.DataArray(
        dates_b.values,
        dims=['index'],
        coords=dict(index=index)
    )

    return paired_ds

