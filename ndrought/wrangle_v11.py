""" Support for wrangling data and organizing it.

This module contains various helper functions for accessing, converting, and manipulating data used in nDrought.

Updated: 3.31.2023
Author: a. stein

"""

import xarray as xr
import pandas as pd
import numpy as np
import rioxarray
from shapely.geometry import mapping
import geopandas as gpd
import matplotlib as plt
from tqdm.autonotebook import tqdm

import pickle

import skimage

from skimage.color import rgb2gray
from skimage.measure import regionprops_table

import pyproj



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

def threshold_filtered_pair(x:np.ndarray, y:np.ndarray, minx=None, maxx=None, miny=None, maxy=None):
    """Filter for x and y based on a threshold to either, assuming 1:1 mapping.

    Parameters 
    ----------
    x, y : np.ndarray
        Array of values that assumes x and y are mapped 1:1 and ordered, such that 
        the first element of x maps to the first element of y, and so forth.
    minx, miny : float, (optional)
        Minimum inclusive threshold for x and y, respectively.
    maxx,y maxy : float, (optional)
        Maximum inclusive threshold for x and y, respectively.

    Returns
    -------
    x, y : np.ndarray
        x and y filtered based on the provided thresholds, respectively.
    """
    # for each of these, if a threshold isn't provided then
    # we need to have an array that contains all of the indices
    if minx:
        x_above_floor = np.where(x>=minx)[0]
    else:
        x_above_floor = np.arange(0, len(x))

    if maxx:
        x_below_ceiling = np.where(x<=maxx)[0]
    else:
        x_below_ceiling = np.arange(0, len(x))

    if miny:
        y_above_floor = np.where(y>=miny)[0]
    else:
        y_above_floor = np.arange(0, len(y))
    
    if maxy:
        y_below_ceiling = np.where(y<=maxy)[0]
    else:
        y_below_ceiling = np.arange(0, len(y))

    # find common indices to select out
    x_filter_idx = np.intersect1d(x_above_floor, x_below_ceiling)
    y_filter_idx = np.intersect1d(y_above_floor, y_below_ceiling)
    filter_idx = np.intersect1d(x_filter_idx, y_filter_idx)

    return x[filter_idx], y[filter_idx]

def cat_area_frac(da:xr.DataArray, cat_val:int):
    """Fraction of non-nan area within a given drought category.

    This function is applied at every index increment, collapsing
    the data from spatio-temporal to just temporal.

    Parameters
    ----------
    da : xr.DataArray
        Contains data categorized according to USDM drought categories.
        Expecting `index` to be the temporal dimension.
    cat_val : int
        Category value to compute fraction over. For example, neutral or
        wet corresponds to -1, D0 to 0, D1 to 1, D2 to 2, D3 to 3, and
        D4 to 4.

    Returns
    -------
    list
        Fractions of area in drought category ordered in the same order
        as the index dimension.

    """

    tot_cells = (np.isnan(da.isel(index=0).values) == False).sum()
    cat_cells = [(da.sel(index=i).values == cat_val).sum() for i in da.index]
    percents = cat_cells/tot_cells

    return percents

def compile_cat_area_fracs(da:xr.DataArray, var_prefix=None):
    """Applies cat_area_frac to categories neutral to D4.

    Parameters
    ----------
    da : xr.DataArray
        Data categorized to the USDM drought categories, where
        -1 is netural or wet, 0 is D0, 1 is D1, 2 is D2, 3 is D3,
        and 4 is D4. Expecting `index` to be the temporal dimension.
    var_prefix : str, (optional)
        Append a prefix to the variables in the xarray Dataset that
        will be returned, should you aim to concat them into a larger
        dataset.
    
    Returns
    -------
    xr.Dataset
        Dataset containing fraction of non-nan area in each USDM
        drought category, using the `index` dimension from the
        provided variable `da`.
        
    """

    neutral_wet = cat_area_frac(da, -1)
    d0 = cat_area_frac(da, 0)
    d1 = cat_area_frac(da, 1)
    d2 = cat_area_frac(da, 2)
    d3 = cat_area_frac(da, 3)
    d4 = cat_area_frac(da, 4)

    index = da.index.values

    ds = xr.Dataset(
        coords=dict(
            index=index
        ),
        data_vars=dict(
            neutral_wet=(["index"], neutral_wet),
            D0=(["index"], d0),
            D1=(["index"], d1),
            D2=(["index"], d2),
            D3=(["index"], d3),
            D4=(["index"], d4),
        ),
        attrs=dict(
            {
                'description':'Fraction of total non-nan area that is in that USDM drought category for the given index.'
            }
        )
    )

    if var_prefix:
        ds = ds.rename({
            "neutral_wet":f"{var_prefix}_neutral_wet",
            "D0":f"{var_prefix}_D0",
            "D1":f"{var_prefix}_D1",
            "D2":f"{var_prefix}_D2",
            "D3":f"{var_prefix}_D3",
            "D4":f"{var_prefix}_D4"
        })

    return ds

def apply_by_geometries(da:xr.DataArray, geometries:gpd.GeoSeries, func, **func_kwargs):
    """Apply a function to the data based on the GeoSeries.

    Parameters
    ----------
    da : xr.DataArray
        Data to apply function to.
    geometries : gpd.GeoSeries
        Contains geometries to clip the data to.
    func : function
        Function to apply to da.
    **func_kwargs, (optional)
        Keyword arguments for func.
    
    Returns
    -------
    list
        Data after function has been applied to each geometric region, 
        in the same order as geometries.
        
    """

    applied = []
    
    for geo in tqdm(geometries, total=len(geometries)):
        minx, miny, maxx, maxy = geo.bounds
        # clipping to box first reduces total data working with for clip
        clipping = (da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)).rio.clip([geo])
        post_op = func(clipping, **func_kwargs)
        applied.append(post_op)

    return applied

def cat_norm_time_freq(da:xr.DataArray, cat_val:int):
    """Normalized time frequency within a given drought category.

    This function is applied individually to each spatial cell,
    collapsing the data from spatio-temporal to just spatial.
    Normalization is relative to the total number of temporal
    indices.

    Parameters
    ----------
    da : xr.DataArray
        Contains data categorized according to USDM drought categories.
        Expecting `index` to be the temporal dimension, and `lat` and 
        `lon` as the latitude (y) and longitude (x) spatial dimensions.
    cat_val : int
        Category value to compute normalized time frequency over. For 
        example, neutral or wet corresponds to -1, D0 to 0, D1 to 1, 
        D2 to 2, D3 to 3, and D4 to 4.

    Returns
    -------
    numpy masked array
        Normalized time frequencies in drought category with the shape
        (`lat`, `lon`), masked by the np.nan values of the first temporal
        index in `da`.
    """

    cat_count = np.array([[np.sum(da.sel(lat=lat, lon=lon).values == cat_val) for lon in da['lon']] for lat in da['lat']])
    cat_tf = cat_count/len(da['index'].values)
    cat_tf_masked = np.ma.masked_where(np.isnan(da.isel(index=0).values), cat_tf)

    return cat_tf_masked

def compile_norm_time_freqs(da:xr.DataArray, var_prefix=None):
    """Applies cat_norm_time_freq to categories neutral to D4.

    Parameters
    ----------
    da : xr.DataArray
        Data categorized to the USDM drought categories, where
        -1 is netural or wet, 0 is D0, 1 is D1, 2 is D2, 3 is D3,
        and 4 is D4. Expecting `index` to be the temporal dimension,
        and `lat` and `lon` as the latitude (y) and longitude (x)
        spatial dimensions.
    var_prefix : str, (optional)
        Append a prefix to the variables in the xarray Dataset that
        will be returned, should you aim to concat them into a larger
        dataset.

    Returns
    -------
    xr.Dataset
        Dataset containing normalized time frequency for the drought
        categories of each cell across `lat` and `lon` from the
        provided variable `da`.

    """

    neutral_wet = cat_norm_time_freq(da, -1)
    d0 = cat_norm_time_freq(da, 0)
    d1 = cat_norm_time_freq(da, 1)
    d2 = cat_norm_time_freq(da, 2)
    d3 = cat_norm_time_freq(da, 3)
    d4 = cat_norm_time_freq(da, 4)

    lon = da.lon.values
    lat = da.lat.values

    ds = xr.Dataset(
        coords=dict(
            lat=lat,
            lon=lon
        ),
        data_vars=dict(
            neutral_wet=(["lat", "lon"], neutral_wet),
            D0=(["lat", "lon"], d0),
            D1=(["lat", "lon"], d1),
            D2=(["lat", "lon"], d2),
            D3=(["lat", "lon"], d3),
            D4=(["lat", "lon"], d4),
        ),
        attrs=dict(
            {
                'description':'Temporal frequency each lat-lon cell is in each USDM drought category normalized by the total number of time indices.'
            }
        )
    )

    if var_prefix:
        ds = ds.rename({
            "neutral_wet":f"{var_prefix}_neutral_wet",
            "D0":f"{var_prefix}_D0",
            "D1":f"{var_prefix}_D1",
            "D2":f"{var_prefix}_D2",
            "D3":f"{var_prefix}_D3",
            "D4":f"{var_prefix}_D4"
        })

    return ds

def get_usdmcat_com(data:np.ndarray):
    """Compute center of mass for USDM category data.

    Parameters
    ----------
    data: np.ndarray
        Contains values ranging from -1 to 4 according
        to USDM category gridding, with empty cells
        being set to np.nan
    
    Returns
    -------
    x, y : float
        Center of mass coordinates in x and y computed
        by ndimage.measurements.center_of_mass
    
    """
    
    # shift everything to positive nonzero
    data += 2
    # replace nans with zero mass
    data[np.isnan(data)] = 0

    # compute center of mass
    com_y, com_x = ndimage.measurements.center_of_mass(data)

    return com_x, com_y

def transform_index_to_coords(idx_data:np.ndarray, coord_ref:np.ndarray):
    """Transforms coordinate indices to coordinates.

    This function is useful especially if you have partial indices, like
    those that might be computed from calculating center of mass. Note
    that this is designed for 1D arrays, so you would make 2 separate
    function calls to transform longitude data and latitude data.

    Parameters
    ----------
    idx_data : np.ndarray
        Index based data.
    coord_ref : np.ndarray
        Reference coordinate array that covers the full spread to project
        the idx_data onto.

    Returns
    -------
    np.ndarray    
    """

    return ((coord_ref[-1] - coord_ref[0])*idx_data/len(coord_ref))+coord_ref[0]

def compute_usdmcat_com_coords(da:xr.DataArray):
    """Computes center of mass based on UDSM categories as weights.

    This function combines get_usdm_com and transform_index_to_coords
    into one function for simplicity.

    Parameters
    ----------
    da : xr.DataArray
        Contains data categorized from -1 to 4 according to gridded USDM
        scheme. Expects `lat` and `lon` dimensions for longitude and latitude,
        along with `index` dimension as the temporal component.

    Returns
    -------
    x, y : np.ndarray
        X and Y coordinates as longitude and latitude values, respectively, for
        the center of mass at each corresponding time interval in `index`.
    
    """

    com_x_list = []
    com_y_list = []

    for idx in da.index:
        com_x, com_y = get_usdmcat_com(da.sel(index=idx).values)
        com_x_list.append(com_x) 
        com_y_list.append(com_y)


    com_x_coords = transform_index_to_coords(np.array(com_x_list), da.lon.values)
    com_y_coords = transform_index_to_coords(np.array(com_y_list), da.lat.values)

    return com_x_coords, com_y_coords

def identify_drought_blob(vals:np.ndarray, threshold=1):
    """Using sci-kit image, identify drought event blobs.

    Parameters
    ----------
    vals: np.ndarray
        Spatial values for drought data categorized
        according to the USDM scheme for a single
        time step.

    Returns
    -------
    pd.DataFrame
        Drought blobs using connectivity 2 from
        skimage.measure.label. Blobs are binary
        definitions of drought, where the measure
        exceeds D1. Each blob is provided with
        it's area, bbox, convex_area, and coordinates
        of all cells contained within the blob.    
    """

    # first we're going to make this binary
    # by setting data in a drought to 1 and
    # not in a drought to 0, including nan

    vals[(vals < threshold) | np.isnan(vals)] = 0
    vals[vals > 0] = 1

    # now we are going to convert to RGBL
    (h, w) = vals.shape
    t = (h, w, 3)
    A = np.zeros(t, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            # since we already made it binary, this
            # will make 1 vals be white and 0 vals
            # be black in our RGB array
            color_val = 255*vals[i,j]
            A[i, j] = [color_val, color_val, color_val]

    # connectivity 2 will consider diagonals as connected
    blobs = skimage.measure.label(rgb2gray(A) > 0, connectivity=2)

    properties =['area','bbox','convex_area','coords']
    df = pd.DataFrame(regionprops_table(blobs, properties=properties))
    df['drought_id'] = np.nan*np.zeros(len(df))

    return df


def connect_blobs_over_time(df_1:pd.DataFrame, df_2:pd.DataFrame):
    """Identify blobs shared between time frames.

    Parameters
    ---------
    df_1 : pd.DataFrame
        Blob dataframe at first time index.
    df_2 : pd.DataFrame
        Blob dataframe at second time index.

    Returns
    -------
    list
        Indices to each dataframe denoting which
        blobs are shared, where each tuple in the
        list is connection. The first index of
        each tuple corresponds to df_1, while the
        second index correponds to df_2
    """

    blob_pairs = []

    for idx_1, df_1_coords in enumerate(df_1.coords.values):
        df_1_coords_set = set(tuple(coord) for coord in df_1_coords)
        for idx_2, df_2_coords in enumerate(df_2.coords.values):
            df_2_coords_set = set(tuple(coord) for coord in df_2_coords)
            if len(df_1_coords_set.intersection(df_2_coords_set)) > 0:
                blob_pairs.append((idx_1, idx_2))

    return blob_pairs

def propagate_drought_id(df_1=None, df_2=None, connections=[], new_blob_num=1):

    if len(connections) > 0:

        # need to keep track of splits among multiple
        # blobs (since they are 1-to-many and we are
        # iterating through linearly)
        split_origins = dict()

        for i in np.arange(len(df_2)):
            drought_id = ""

            # ALL CONNECTIONS
            # first we need to figure out if we are connected
            connects_origins = list()
            for connect in connections:
                # this means that our current index
                # connects to a previous time's index
                if connect[1] == i:
                    # we already know it's going to index i
                    # we need to figure out where it's coming from
                    connects_origins.append(connect[0])

            # SPLITS        
            # now we need to check if this is part of a split
            split_connections = dict()
            for origin in connects_origins:
                split_counter = 0
                for connect in connections:
                    # we want to count how many times the origin is
                    # connected to something ... if it ends up being
                    # more than once then it's a split
                    if connect[0] == origin:
                        split_counter += 1
                # meaning we found a split
                if split_counter > 1:
                    split_connections[origin] = split_counter
                    # if this is a new split we found, we
                    # should make sure to save a note of it
                    if origin not in split_origins.keys():
                        split_origins[origin] = 1
                
            # so this would be if the split was found        
            if len(split_connections) > 0:
                for split_origin in split_connections.keys():
                    split_origin_id = df_1['drought_id'].values[split_origin]
                    current_split_num = split_origins[split_origin]

                    drought_id = f'{split_origin_id}-{current_split_num}'
                    
                    # iterate for the next blob it splits into
                    split_origins[split_origin] += 1

            # MERGES
            # we have a merge if more than 1 blob
            # goes into this one
            if len(connects_origins) > 1:
                merged_blob_ids = df_1.iloc[connects_origins].sort_values('area', ascending=False)['drought_id'].values
                # double check if we already had a split and began
                # writing our code for this blob, if not we need to
                # set it up
                if len(drought_id) == 0:
                    drought_id = merged_blob_ids[0]
                for id in merged_blob_ids[1:]:
                    drought_id = f'{drought_id}.({id})'
                    
            # NO SPLIT NO MERGE        
            if len(connects_origins) == 1 and len(split_connections) == 0:
                drought_id = df_1.iloc[connects_origins[0]]['drought_id']
                

            # CONNECTIONS EXIST, BUT NEW BLOB
            if len(connects_origins) == 0:
                drought_id = f'{new_blob_num}'
                new_blob_num += 1    

            df_2.loc[i, 'drought_id'] = drought_id                   

    else:
        # there were no connections, all id's start from scratch
        for i in np.arange(len(df_2)):
            df_2.loc[i, 'drought_id'] = f'{new_blob_num}'
            new_blob_num += 1

    return df_2, new_blob_num

def encode_drought_events(data:np.ndarray):
    """Detect and encode drought events.

    Parameters
    ----------
    data: np.ndarray
        Expecting first index to be temporal while second
        and third are spatial.

    Returns
    -------
    pd.DataFrame
        A multi-indexed dataframe with time as the first level
        and drought_id as the second level. 'area', 'convex_area',
        and 'coords' are also outputted in this dataframe computed 
        from sci-kit image. 
    
    """
    blob_dfs = []

    for i in tqdm(np.arange(data.shape[0]), desc='Identifying Blobs'):
        blob_dfs.append(identify_drought_blob(data[i,:,:]))
    

    #return blob_dfs

    new_blob_num = 1
    init_df, new_blob_num = propagate_drought_id(df_2=blob_dfs[0])
    init_df['time'] = 0
    encoded_blob_dfs = [init_df]    
    for i in tqdm(np.arange(len(blob_dfs)-1), desc='Encoding Blobs'):
        df_1 = encoded_blob_dfs[i]
        df_2 = blob_dfs[i+1]

        blob_pairs = connect_blobs_over_time(df_1, df_2)
        df_2_encoded, new_blob_num = propagate_drought_id(df_1, df_2, blob_pairs, new_blob_num)
        df_2_encoded['time'] = i+1
        encoded_blob_dfs.append(df_2_encoded)

    all_blobs_df = pd.concat([df[['time', 'drought_id', 'area', 'convex_area', 'coords']] for df in encoded_blob_dfs], ignore_index=True)
    all_blobs_df = all_blobs_df.set_index(['time', 'drought_id'])
    all_blobs_df['drought_id'] = all_blobs_df.index.get_level_values(1)

    return all_blobs_df

def check_event_id_trace(event_id:str, drought_id:str):
    """Check if a drought_id contains a specific event.

    Parameters
    ----------
    event_id : str
        What might be considered the root of the drought_id, 
        the event that you are looking to follow the
        evolution of through merges and splits.
    drought_id : str
        drought_id to check from propagate_drought_id.

    Returns
    -------
    boolean
        If the drought_id contains the event_id.

    """

    event_found = False
    sub_events = drought_id.split('.')
    i = 0
    while i < len(sub_events) and not event_found:
        check_sub_event = sub_events[i].split('-')[0]
        if check_sub_event[0] == '(':
            check_sub_event = check_sub_event[1:]
        if check_sub_event[-1] == ')':
            check_sub_event = check_sub_event[:-1]
        event_found = check_sub_event == event_id
        i += 1

    return event_found

def plot_drought_evolution(df:pd.DataFrame, event_id='', plot_var='area', ax=None, plot_legend=True, cmap=plt.cm.get_cmap('hsv')):
    """Plots the evolution of droughts over time from blob detection.

    Parameters
    ----------
    df : pd.DataFrame
        Expected to be the output of encode_drought_events
    event_id : str (optional)
        The initial drought_id of the drought wishing to follow.
        Defaults as '' to plot everything.
    plot_var : str (optional)
        Variable from df to plot, defaults as 'area' to plot pixel
        area computed from blob detection.
    ax (optional)
        matplotlib axes object to plot on. one will be created
        if not given.
    plot_legend : boolean
        Whether to plot the legend (which can sometimes be quite long).
        Defaults as True to plot the legend
    """

    assert(isinstance(event_id, str))

    if ax is None:
        __, ax = plt.subplots()

    related_events_idx = [i for i, val in enumerate(df['drought_id']) if event_id == '' or check_event_id_trace(event_id, val)]
    thread_df = df.iloc[related_events_idx]

    # need to grab the last value since we aren't going to get a df to plot
    # from times when there is no drought
    times = np.arange(df.index.get_level_values(0)[-1]+1)
    template = np.zeros(len(times))

    unique_drought_id = sorted(set(thread_df['drought_id'].values))
    #print(unique_drought_id)

    droughts = []
    for id in unique_drought_id:
        event_df = thread_df[thread_df.drought_id == id]
        event_array = template.copy()
        event_times = event_df.index.get_level_values(0)

        for time in event_times:
            event_array[time] = event_df.loc[time][plot_var]
        
        droughts.append(event_array)

    color_array = np.linspace(0, 1, len(droughts))
    colors = cmap(color_array)

    ax.stackplot(times, *droughts, labels=unique_drought_id, colors=colors)
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{plot_var} in drought event')

    if plot_legend:
        ax.legend()

    return ax

def convert_pickle_to_dtd(path):
    """convert drought tracks pickled to drought track dict"""
    with open(path, 'rb') as f:
        unpickler = pickle.Unpickler(f)
        dt = unpickler.load()

    dtd = dict()
    vars = ['x', 'y', 'u', 'v', 't', 'c', 'a', 's', 'sf', 'id']

    for tracks, var in zip(dt, vars):
        var_tracks = []
        for track in tracks:
            var_tracks.append(np.array(track))
        dtd[var] = var_tracks

    return dtd

def compute_track_lifetime(t_track, to_days):
    track_lifetime = []

    for times in t_track:
        track_lifetime.append((np.max(times) - np.min(times))*to_days)

    return track_lifetime

def compute_distance(u,v):
    return np.sqrt(np.power(u, 2)+np.power(v, 2))

def compute_track_distance(u_track, v_track):
    distance = []
    for u, v in zip(u_track, v_track):
        distance.append(compute_distance(u,v).sum())
    return distance

def compute_track_displacement(x_track, y_track, u_track, v_track):
    displacement = []

    for x, y, u, v in zip(x_track, y_track, u_track, v_track):
        del_x = u[-1]+x[-1]-(x[0])
        del_y = v[-1]+y[-1]-(y[0])

        displacement.append(compute_distance(del_x, del_y))
    
    return displacement

def compute_track_similarity(a_track):
    return [a.sum()/len(a) for a in a_track]

def compute_track_summary_characterization(dtrack_dict, to_days, bins=[0, 30, 60, 90, 180, 365, 730, 1825, 7*1163]):
    x_track = dtrack_dict['x']
    y_track = dtrack_dict['y']
    u_track = dtrack_dict['u']
    v_track = dtrack_dict['v']
    t_track = dtrack_dict['t']
    a_track = dtrack_dict['a']

    lifetimes = compute_track_lifetime(t_track, to_days)
    distances = compute_track_distance(u_track, v_track)
    displacements = compute_track_displacement(x_track, y_track, u_track, v_track)
    similarity = compute_track_similarity(a_track)

    summary_df = pd.DataFrame(index=np.arange(len(lifetimes)))
    summary_df['lifetime'] = lifetimes
    summary_df['distance'] = distances
    summary_df['displacement'] = displacements
    summary_df['average velocity'] = summary_df['distance']/summary_df['lifetime']
    summary_df['similarity'] = similarity

    #try:
    summary_df['xy_size'] = dtrack_dict['s']
    summary_df['uv_size'] = dtrack_dict['sf']
    #except Exception as e:
    #    print(e)

    #return summary_df

    return summary_df, summary_df.groupby(pd.cut(summary_df.lifetime, bins=bins))#.agg(['mean', 'median', 'max', 'min', 'std', 'count'])
    # for some reason agg is only working on a single column now

def transform_points(x, y, inproj=pyproj.CRS('epsg:5070'), outproj=pyproj.CRS('epsg:4326')):
    """Transforms points using pyproj
    """
    return pyproj.transform(inproj,outproj,x,y)



def create_txy_points(x_tracks, y_tracks, t_tracks):
    txy_tracks = dict()
    i = 0
    for x_track, y_track, t_track in zip(x_tracks, y_tracks, t_tracks):
        txy_track = []
        for x, y, t in zip(x_track, y_track, t_track):
            txy_track.append((t,x,y))
        txy_tracks[i] = (txy_track)
        i += 1
    return txy_tracks

def find_like_terminations(txy_points):
    terminations = dict()
    for key in txy_points.keys():
        terminations[key] = txy_points[key][-1]
    
    like_terminations = []
    for key_a in terminations.keys():
        like_term = [key_a]
        term_a = terminations[key_a]
        for key_b in terminations.keys():
            if key_a != key_b and term_a == terminations[key_b]:
                like_term.append(key_b)

        like_term = list(set(like_term))
        if len(like_term) > 1 and not like_term in like_terminations:
            like_terminations.append(like_term)
    return like_terminations

def create_area_filter(s_tracks, like_terminations):

    sums = dict()
    i = 0
    for s_track in s_tracks:
        sums[i] = np.sum(s_track)
        i += 1

    select_tracks = []

    for like_terms in like_terminations:
        mat_size = len(like_terms)
        s_mat = np.zeros((mat_size, mat_size))

        for i, key_a in enumerate(like_terms):
            sum_a = sums[key_a]
            for j, key_b in enumerate(like_terms):
                sum_b = sums[key_b]
                s_mat[i, j] = sum_a-sum_b

        s_mat >= 0

        found_i = -1
        found_track = -1

        i=0
        while found_i == -1 and i < s_mat.shape[0]:
            if np.all(s_mat[i, :] >= 0):
                found_i = i
                found_track = like_terms[i]
            i += 1

        select_tracks.append(found_track)
    
    termination_filter = []

    for term in np.hstack(like_terminations):
        if not term in select_tracks:
            termination_filter.append(term)

    return termination_filter

def to_y(y, y_meta):
    y_min, y_max, y_spacing = y_meta
    return ((y_min-y_max)/y_spacing)*(y)+y_max

def to_x(x, x_meta):
    x_min, x_max, x_spacing = x_meta
    return ((x_max-x_min)/x_spacing)*(x)+x_min

def to_xy(coord, coord_meta):
    y_min, y_max, y_spacing, x_min, x_max, x_spacing = coord_meta

    y_meta = (y_min, y_max, y_spacing)
    x_meta = (x_min, x_max, x_spacing)

    y, x = coord
    return (to_x(x, x_meta), to_y(y, y_meta))

def collect_drought_track(args):
    x_list = []
    y_list = []
    u_list = []
    v_list = []
    t_list = []
    color_list = []
    alpha_list = []
    s_list = []
    sf_list = []

    (origin, net_adj_dict, net_centroids, s_thresh, ratio_thresh, cmap) = args

    q = Queue()
    q.put(origin.id)
    thread_ids = [origin.id]

    while not q.empty():
        
        current_id = q.get()

        for future_id in net_adj_dict[current_id]:
            if not future_id in thread_ids:
                
                        
                x, y, t, s = net_centroids[current_id]
                
                if s > s_thresh:
                    u, v, __, s_f = net_centroids[future_id]
                    if s_f > ratio_thresh*s:
                        
                        q.put(future_id)
                        thread_ids.append(future_id)

                        x_list.append(x)
                        y_list.append(y)
                        u_list.append(u-x)
                        v_list.append(v-y)
                        t_list.append(t)

                        alpha_list.append(np.min((s_f/s, s/s_f)))
                        s_list.append(s)
                        sf_list.append(s_f)

    if len(t_list) > 0:
        t_min = np.min(t_list)
        t_max = np.max(t_list)
        color_list = [cmap(np.round((t-t_min)/(t_max-t_min), 4))[:-1] for t in t_list]

    return x_list, y_list, u_list, v_list, t_list, color_list, alpha_list, s_list, sf_list

def extract_drought_tracks(net, coord_meta, client, log_dir, cmap=plt.cm.get_cmap('viridis'), s_thresh=0, ratio_thresh=0):

    net_centroids = {node.id:(*to_xy(node.coords.mean(axis=0), coord_meta), node.time, len(node.coords)) for node in net.nodes}

    x_tracks = []
    y_tracks = []
    u_tracks = []
    v_tracks = []
    t_tracks = []
    color_tracks = []
    alpha_tracks = []
    s_tracks = []
    sf_tracks = []

    valid_origins = []
    for origin in tqdm(net.origins, desc='Collecting Valid Origins'):
        # the ones that are one-off events I don't want to plot
        if len(origin.future) > 0:           
            valid_origins.append(origin)

    print(f'{len(valid_origins)} Valid Origins Found')

    values = [dask.delayed(collect_drought_track)((o, net.adj_dict, net_centroids, s_thresh, ratio_thresh, cmap)) for o in valid_origins]
    persisted_values = dask.persist(*values)
    for pv in tqdm(persisted_values):
        try:
            wait(pv)
        except Exception:
            pass
    
    print(f'Extracting Tracks from {net.name}')
    futures = client.compute(persisted_values)

    results = []
    for o, future in zip(valid_origins, futures):
        try:
            results.append(future.result())
        except Exception as e:
            with open(f'{log_dir}/origin_error_{o.id}.log', 'w') as file:
                file.write(f'{e}')
                file.close()   
     
    if len(results) != len(valid_origins):
        print(f'>>>>>>>>>> LOST {len(valid_origins) - len(results)} TRACKS <<<<<<<<<<<')
        
    for result in tqdm(results, desc='Reshaping and Packaging'):
        x_list, y_list, u_list, v_list, t_list, color_list, alpha_list, s_list, sf_list = result

        x_tracks.append(x_list)
        y_tracks.append(y_list)
        u_tracks.append(u_list)
        v_tracks.append(v_list)
        t_tracks.append(t_list)
        color_tracks.append(color_list)
        alpha_tracks.append(alpha_list)
        s_tracks.append(s_list)
        sf_tracks.append(sf_list)


    return x_tracks, y_tracks, u_tracks, v_tracks, t_tracks, color_tracks, alpha_tracks, s_tracks, sf_tracks

def filter_track(tracks, track_filter):
    filtered_tracks = []    
    for i, track in enumerate(tracks):
        if not i in track_filter:
            filtered_tracks.append(track)
    return filtered_tracks

def prune_tracks(dtd):
    x_tracks = dtd['x']
    y_tracks = dtd['y']
    t_tracks = dtd['t']
    s_tracks = dtd['s']

    txy_points = create_txy_points(x_tracks, y_tracks, t_tracks)
    like_terminations = find_like_terminations(txy_points)
    area_filter = create_area_filter(s_tracks, like_terminations)

    filtered_dtd = dict()
    for key in dtd.keys():
        filtered_dtd[key] = filter_track(dtd[key], area_filter)

    return filtered_dtd

def dm_to_usdmcat(da:xr.DataArray, percentiles = [30, 20, 10, 5, 2]):
    """Categorizes drought measure based on USDM categories.

    Uses the mapping scheme presented by USDM (https://droughtmonitor.unl.edu/About/AbouttheData/DroughtClassification.aspx)
    Where Neutral is -1, D0 is 0, D1 is 1, D2, is 2, D3 is 3, and D4 is 4.

    Parameters
    ----------
    da : xr.DataArray
        Contains data values.
    percentiles: array-like
        Inculsive upperbounds for D0, D1, D2, D3, and D4 in that order, where anything greater than the upperbound for D0 is
        considered Neutral. It is assumed that 0 is the lower
        bound for D4.
    
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
    (p30, p20, p10, p5, p2) = np.percentile(da_vals_nonnan.ravel(), percentiles)
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

def dm_to_usdmcat_multtime(ds:xr.Dataset, percentiles=[30, 20, 10, 5, 2]):
    """Categorizes drought measure based on USDM categories for multiple times.
    
    See dm_to_usdmcat for further documentation.
    
    Parameters
    ----------
    ds : xr.Dataset
        Data at multiple time values as the coordinate 'time'.
    
    Returns
    -------
    xr.Dataset
        Drought measure categorized by dm_to_usdmcat.
    """
    
    return dm_to_usdmcat(xr.concat([ds.sel(time=t) for t in ds['time'].values], dim='time'), percentiles)