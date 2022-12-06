import xarray as xr
import pandas as pd
import numpy as np

from tqdm.autonotebook import tqdm

import sys
sys.path.append('../')
import ndrought.compare as compare

import multiprocessing as mp
import os

# load in data
print('Loading Data')
dm_path = '/pool0/home/steinadi/data/drought/drought_impact/data/drought_measures'

usdm = xr.open_dataset(f'{dm_path}/usdm/USDM_WA_20000104_20220412.nc').load()

intervals = ['14d', '30d', '90d', '180d', '270d', '1y', '2y', '5y', ]
spi = xr.open_dataset(f'{dm_path}/spi/WA/spi_usdmcat_WA.nc').load()
spei = xr.open_dataset(f'{dm_path}/spei/WA/spei_usdmcat_WA.nc').load()
eddi = xr.open_dataset(f'{dm_path}/eddi/WA/eddi_usdmcat_WA.nc').load()
pdsi = xr.open_dataset(f'{dm_path}/pdsi/WA/pdsi_usdmcat_WA.nc').load()

grace = xr.open_dataset(f'{dm_path}/grace/WA/grace_usdmcat_WA.nc').load()
grace_vars = ['gws', 'rtzsm', 'sfsm']

# pair dates
print ('Pairing Dates ...')
usdm_dates = pd.to_datetime(usdm.date.values)
spi_dates = pd.to_datetime(spi.day.values)
spei_dates = pd.to_datetime(spei.day.values)
eddi_dates = pd.to_datetime(eddi.day.values)
pdsi_dates = pd.to_datetime(pdsi.day.values)
grace_dates = pd.to_datetime(grace.time.values)

all_dates = [usdm_dates, spi_dates, spei_dates, eddi_dates, pdsi_dates, grace_dates]
dm_vars = ['usdm', 'spi', 'spei', 'eddi', 'pdsi', 'grace']

usdm_pairings = dict()
for date, var in zip(all_dates, dm_vars):
    if var == 'usdm':
        usdm_pairings[var] = pd.DataFrame(usdm_dates, columns=[var])
    else:
        method = 'last-b'

        usdm_pairings[var] = compare.pair_dates(usdm_dates, date, 'usdm', var, realign=True, method=method)
print('... USDM dates paired')

spi_pairings = dict()
for date, var in zip(all_dates, dm_vars):
    if var == 'spi':
        spi_pairings[var] = pd.DataFrame(spi_dates,  columns=[var])
    else:
        if var == 'grace':
            method = 'last-b'
        else:
            method = 'last-a'
        
        spi_pairings[var] = compare.pair_dates(spi_dates, date, 'spi', var, realign=True, method=method)
print('... SPI dates paired')

spei_pairings = dict()
for date, var in zip(all_dates, dm_vars):
    if var == 'spei':
        spei_pairings[var] = pd.DataFrame(spei_dates, columns=[var])
    else:
        if var in ['grace', 'spi']:
            method = 'last-b'
        else:
            method = 'last-a'

        spei_pairings[var] = compare.pair_dates(spei_dates, date, 'spei', var, realign=True, method=method)
print('... SPEI dates paired')


eddi_pairings = dict()
for date, var in zip(all_dates, dm_vars):
    if var == 'eddi':
        eddi_pairings[var] = pd.DataFrame(eddi_dates, columns=[var])
    else:
        if var == 'pdsi':
            method = 'nearest'
        elif var == 'usdm':
            method = 'last-a'
        else:
            method = 'last-b'

        eddi_pairings[var] = compare.pair_dates(eddi_dates, date, 'eddi', var, realign=True, method=method)
print('... EDDI dates paired')

pdsi_pairings = dict()
for date, var in zip(all_dates, dm_vars):
    if var == 'pdsi':
        pdsi_pairings[var] = pd.DataFrame(pdsi_dates, columns=[var])
    else:
        if var == 'eddi':
            method = 'nearest'
        elif var == 'usdm':
            method = 'last-a'
        else:
            method = 'last-b'
        
        pdsi_pairings[var] = compare.pair_dates(pdsi_dates, date, 'pdsi', var, realign=True, method=method)
print('... PDSI dates paired')


grace_pairings = dict()
for date, var in zip(all_dates, dm_vars):
    if var == 'grace':
        grace_pairings[var] = pd.DataFrame(grace_dates, columns=[var])
    else:
        method = 'last-a'

        grace_pairings[var] = compare.pair_dates(grace_dates, date, 'grace', var, realign=True, method=method)
print('... GRACE dates paired')
print()


# create paired ds
# (I'm not going to worry about doubling up on combinations because it can serve to cross-validate)

# need to create some easier ways to access the data so I can just iterate by overarching variable
all_dm_ds = {
    'usdm':usdm,
    'spi':spi,
    'spei':spei,
    'eddi':eddi,
    'pdsi':pdsi,
    'grace':grace
}

all_date_pairings = {
    'usdm':usdm_pairings,
    'spi':spi_pairings,
    'spei':spei_pairings,
    'eddi':eddi_pairings,
    'pdsi':pdsi_pairings,
    'grace':grace_pairings,
}


def sel_time_and_set_crs(ds, var, dates, crs='EPSG:4326'):
    
    if var == 'usdm':
        da_list = [ds['USDM'].sel(date=dates).rio.write_crs(crs, inplace=True)]
    elif var == 'pdsi':
        da_list = [ds[var].sel(day=dates).rio.write_crs(crs, inplace=True)]
    elif var == 'grace':
        da_list = [ds[grace_var].sel(time=dates).rio.write_crs(crs, inplace=True) for grace_var in grace_vars]
    else:
        da_list = [ds[f'{var}_{interval}'].sel(day=dates).rio.write_crs(crs, inplace=True) for interval in intervals]
    
    return da_list
    
paired_ds_dict = dict()

t = tqdm(total=len(dm_vars)**2, desc='Pairing Data')

# now this is going to all pair data with themselves, which will be another good check
for var_a in dm_vars:
    if var_a not in paired_ds_dict.keys():
        paired_ds_dict[var_a] = dict()

    date_pairing = all_date_pairings[var_a]
    
    for var_b in (dm_vars):
        t.set_description(f'Pairing {var_a} & {var_b}')
        
        #if var_a is var_b:
        #    # for some reason, when we get the data paired with
        #    # itself the data gets weird and I have to do this fix
        #    dates = pd.to_datetime(list(set(date_pairing[var_b].values.ravel())))
        #    dates_a = pd.DataFrame(data=dates, columns=[var_a])
        #    dates_b = dates_a
        #else:            
        dates_a = date_pairing[var_b][var_a]
        dates_b = date_pairing[var_b][var_b]


        da_a = sel_time_and_set_crs(all_dm_ds[var_a], var_a, dates_a.values)
        da_b = sel_time_and_set_crs(all_dm_ds[var_b], var_b, dates_b.values)

        paired_ds = compare.create_paired_ds(da_a, da_b, dates_a, dates_b, rescale=True)
        paired_ds_dict[var_a][var_b] = paired_ds

        da_a = None
        da_b = None
        paired_ds = None

        t.update()

# now to start doing some statistics finally
dm_vars_expanded = {
    'usdm':['USDM'],
    'spi':[f'spi_{interval}' for interval in intervals],
    'spei':[f'spei_{interval}' for interval in intervals],
    'eddi':[f'eddi_{interval}' for interval in intervals],
    'pdsi':['pdsi'],
    'grace':grace_vars
}

pool = mp.Pool(processes=40)

for var_a in dm_vars:
    for var_b in dm_vars:
        path = f'{dm_path}/ndrought_products/paired_r/{var_a}_{var_b}_r.nc'
        if not os.path.exists(path):
            paired_ds = paired_ds_dict[var_a][var_b]
            paired_ds_r = compare.compute_r_multi_mp(paired_ds, pool, dm_vars_expanded[var_a], dm_vars_expanded[var_b])       
            paired_ds_r.to_netcdf(path)
            paired_ds_r = None



