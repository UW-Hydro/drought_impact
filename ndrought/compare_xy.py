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

