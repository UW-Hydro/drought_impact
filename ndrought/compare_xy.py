import xarray as xr
import pandas as pd
import numpy as np
import multiprocessing as mp

import warnings
warnings.filterwarnings("ignore")

from tqdm.autonotebook import tqdm

intervals = ['14d', '30d', '90d', '180d', '270d', '1y', '2y', '5y', ]
grace_vars = ['gws', 'rtzsm', 'sfsm']

def sel_time_and_set_crs(ds, var, dates, crs='EPSG:5070'):
    
    if var in 'usdm':
        da_list = [ds['USDM'].sel(time=dates).rio.write_crs(crs, inplace=True)]
    elif var == 'pdsi':
        da_list = [ds[var].sel(time=dates).rio.write_crs(crs, inplace=True)]
    elif var == 'grace':
        da_list = [ds[grace_var].sel(time=dates).rio.write_crs(crs, inplace=True) for grace_var in grace_vars]
    else:
        da_list = [ds[f'{var}_{interval}'].sel(time=dates).rio.write_crs(crs, inplace=True) for interval in intervals]
    
    return da_list

def create_paired_ds(da_a:list, da_b:list, dates_a:pd.DataFrame, dates_b:pd.DataFrame, rescale=False):
    y_a = da_a[0].y
    y_b = da_b[0].y
    x_a = da_a[0].x
    x_b = da_b[0].x

    if len(y_a) != len(y_b) or len(x_a) != len(x_b):
        if rescale:
            if len(y_a) > len(y_b) or len(x_a) > len(x_b):
                da_a_rescaled = [da.rio.reproject_match(da_b[0]) for da in da_a]
                da_a = da_a_rescaled
            else:
                da_b_rescaled = [da.rio.reproject_match(da_a[0]) for da in da_b]
                da_b = da_b_rescaled
        else:
            raise Exception('Coordinates are different scales. Either fix or set rescale to True to enable auto upscaling.')
    elif np.any(y_a != y_b) or np.any(x_a != x_b):
        raise Exception('Coordinates do not match despite same size.')
        

    index = dates_a.index.values
    if np.any(index != dates_b.index.values):
        raise Exception('Dates are not paired to the same index.')


    paired_ds = xr.Dataset(
        coords=dict(
            index=(['index'], index),
            y=(['y'], da_a[0].y.values),
            x=(['x'], da_a[0].x.values),
        )
    )
    paired_ds['index'] = index
    
    for da in da_a:
        paired_ds[da.name] = xr.DataArray(
            da.values,
            dims = ['index', 'y', 'x'],
            coords=dict(
                index=index,
                y=paired_ds.y.values,
                x=paired_ds.x.values,
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
            dims=['index', 'y', 'x'],
            coords=dict(
                index=index,
                y=paired_ds.y.values,
                x=paired_ds.x.values,
            )
        )
    paired_ds[f'{dates_b.name.upper()} Date'] = xr.DataArray(
        dates_b.values,
        dims=['index'],
        coords=dict(index=index)
    )

    return paired_ds

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

def compute_r_pixel(args):
   
        # unpack our wrapped arguments
        pixel_x, pixel_y, lag = args

        __, r_pixel = lag_nan_corrcoef(pixel_x, pixel_y, lag=lag)

        return r_pixel

def compute_r_multi_mp(ds:xr.Dataset, pool:mp.Pool, x_vars=list, y_vars=list, lag=20, floor=-1):

    result_ds = xr.Dataset(
        coords=dict(
            y=(["y"], ds.y.values),
            x=(["x"], ds.x.values),
            lag=(["lag"], np.arange(-1*lag, lag+1))
        )
    )

    m = len(result_ds.y)
    n = len(result_ds.x)

    for x_var in x_vars:
        for y_var in y_vars:
            args = []
            t = tqdm(total=m*n, desc=f'Prepping {x_var} {y_var}')
            for y in ds.y.values:
                for x in ds.x.values:

                    pixel_ds = ds.sel(y=y, x=x).copy()

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

            result_ds[f'{x_var}_{y_var}_r'] = (["y", "x", "lag"], results_reshaped)

    return result_ds 

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

