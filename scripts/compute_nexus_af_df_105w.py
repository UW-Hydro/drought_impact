import xarray as xr
import pandas as pd
import numpy as np

from tqdm.tqdm import tqdm

import sys
sys.path.append('../')
import ndrought.compare_xy as compare
import ndrought.drought_network as dnet

import multiprocessing as mp
import os

import pickle
sys.setrecursionlimit(int(1e4))


# load in data
print('Loading Data')
dm_path = '/pool0/home/steinadi/data/drought/drought_impact/data/drought_measures'

usdm = xr.open_dataset(f'{dm_path}/usdm/USDM_CONUS_105W_20000104_20220412.nc')
print('... USDM loaded')
intervals = ['14d', '30d', '90d', '180d', '270d', '1y', '2y', '5y', ]
spi = xr.open_dataset(f'{dm_path}/spi/CONUS_105W/spi_usdmcat_CONUS_105W.nc')
print('... SPI loaded')
spei = xr.open_dataset(f'{dm_path}/spei/CONUS_105W/spei_usdmcat_CONUS_105W.nc')
print('... SPEI loaded')
eddi = xr.open_dataset(f'{dm_path}/eddi/CONUS_105W/eddi_usdmcat_CONUS_105W.nc')
print('... EDDI loaded')
pdsi = xr.open_dataset(f'{dm_path}/pdsi/CONUS_105W/pdsi.nc')
print('... PDSI loaded')

grace = xr.open_dataset(f'{dm_path}/grace/CONUS_105W/grace_usdmcat_CONUS_105W.nc')
print('... GRACE loaded')

# pair dates
print ('Pairing Dates ...')
usdm_dates = pd.to_datetime(usdm.time.values)
spi_dates = pd.to_datetime(spi.time.values)
spei_dates = pd.to_datetime(spei.time.values)
eddi_dates = pd.to_datetime(eddi.time.values)
pdsi_dates = pd.to_datetime(pdsi.time.values)
grace_dates = pd.to_datetime(grace.time.values)

all_dates = [usdm_dates, spi_dates, spei_dates, eddi_dates, pdsi_dates, grace_dates]
dm_vars = ['usdm', 'spi', 'spei', 'eddi', 'pdsi', 'grace']
all_dates_dict = {var:dates for var, dates in zip(dm_vars, all_dates)}

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

all_date_pairings = {
    'usdm':usdm_pairings,
    'spi':spi_pairings,
    'spei':spei_pairings,
    'eddi':eddi_pairings,
    'pdsi':pdsi_pairings,
    'grace':grace_pairings,
}

# now to start doing some statistics finally
dm_vars_expanded = {
    'usdm':['USDM'],
    'spi':[f'spi_{interval}' for interval in intervals],
    'spei':[f'spei_{interval}' for interval in intervals],
    'eddi':[f'eddi_{interval}' for interval in intervals],
    'pdsi':['pdsi'],
    'grace':grace_vars
}

#pool = mp.Pool(processes=40)

alternate_pairs = []
t = tqdm(total=len(dm_vars)**2, desc='Computing')

dnet_path = f'{dm_path}/ndrought_products/CONUS_105W/paired_dnet'

for var_a in dm_vars:
    for var_b in dm_vars:

        t.set_description(f'Computing {var_a} & {var_b}')

        if not (var_a, var_b) in alternate_pairs:

            in_path = f'{dm_path}/ndrought_products/CONUS_105W/paired_r/{var_a}_{var_b}_paired.nc'

            paired_ds = xr.open_dataset(in_path).load()

            var_a_date_idx = []
            for date in all_date_pairings[var_a][var_b][var_a].values:
                var_a_date_idx.append(np.where(all_dates_dict[var_a] == date)[0][0])
            var_b_date_idx = []
            for date in all_date_pairings[var_a][var_b][var_b].values:
                var_b_date_idx.append(np.where(all_dates_dict[var_b] == date)[0][0])
            matched_dates_dict = {var_a_date: var_b_date for var_a_date, var_b_date in zip(var_a_date_idx, var_b_date_idx)}

            for var_a_expanded in dm_vars_expanded[var_a]:
                if not os.path.exists(f'{dnet_path}/{var_a_expanded}_net_{var_b}_match.pickle'):
                    var_a_net = dnet.DroughtNetwork(data=paired_ds[var_a_expanded].values, name=f'{var_a_expanded} ({var_b} Match)')
                    var_a_net.pickle(f'{dnet_path}/{var_a_expanded}_net_{var_b}_match.pickle')
                    var_a_net = None

                var_a_net = dnet.DroughtNetwork.unpickle(f'{dnet_path}/{var_a_expanded}_net_{var_b}_match.pickle')

                for var_b_expanded in dm_vars_expanded[var_b]:
                    if not os.path.exists(f'{dnet_path}/{var_b_expanded}_net_{var_a}_match.pickle'):
                        var_b_net = dnet.DroughtNetwork(data=paired_ds[var_b_expanded].values, name=f'{var_b_expanded} ({var_b} Match)')
                        var_b_net.pickle(f'{dnet_path}/{var_b_expanded}_net_{var_a}_match.pickle')
                        var_b_net = None

                    var_b_net = dnet.DroughtNetwork.unpickle(f'{dnet_path}/{var_b_expanded}_net_{var_a}_match.pickle')

                    path = f'{dm_path}/ndrought_products/CONUS_105W/event_comp/{var_a_expanded}_{var_b_expanded}_comp.pickle'
                    if not os.path.exists(path):

                        try:

                            overlap_events = var_a_net.find_overlapping_nodes_events(var_b_net, matched_dates_dict)

                            af = dnet.compute_alignment_fraction(overlap_events)
                            df_a, df_b = dnet.compute_disagreement_fraction(var_a_net, var_b_net, overlap_events)

                            to_pickle = {
                                'matched_dates':all_date_pairings[var_a][var_b], 
                                'af':af, 
                                f'df_{var_a_expanded}':df_a, 
                                f'df_{var_b_expanded}':df_b
                            }

                            f = open(path, 'wb')
                            pickle.dump(to_pickle, f, pickle.HIGHEST_PROTOCOL)
                            f.close

                            af = None
                            df_a = None
                            df_b = None
                            overlap_events = None
                        except:
                            print(f'Error encountered in {var_a_expanded} & {var_b_expanded}')

                    var_b_net = None
                var_a_net = None
            paired_ds = None

            alternate_pairs.append((var_b, var_a))
        
        t.update()



