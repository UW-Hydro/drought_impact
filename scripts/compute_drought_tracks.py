import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import os
import sys
sys.path.append('../')
import ndrought.drought_network as dnet
import multiprocessing as mp


from queue import Queue
import gc
import pickle
sys.setrecursionlimit(int(1e4))


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
    alpha_list = []

    (origin, net_adj_dict, net_centroids, s_thresh, cmap) = args

    q = Queue()
    q.put(origin.id)
    thread_ids = [origin.id]

    while not q.empty():
        
        current_id = q.get()

        for future_id in net_adj_dict[current_id]:
            if not future_id in thread_ids:
                q.put(future_id)
                thread_ids.append(future_id)
                        
                x, y, t, s = net_centroids[current_id]
                
                if s > s_thresh:
                    u, v, __, s_f = net_centroids[future_id]

                    x_list.append(x)
                    y_list.append(y)
                    u_list.append(u-x)
                    v_list.append(v-y)
                    t_list.append(t)

                    alpha_list.append(np.min((s_f/s, s/s_f)))

    if len(t_list) > 0:
        t_min = np.min(t_list)
        t_max = np.max(t_list)
        color_list = [cmap(np.round((t-t_min)/(t_max-t_min), 4))[:-1] for t in t_list]

    return x_list, y_list, u_list, v_list, t_list, color_list, alpha_list

def extract_drought_tracks(net, coord_meta, cmap=plt.cm.get_cmap('viridis'), s_thresh=0):

    net_centroids = {node.id:(*to_xy(node.coords.mean(axis=0), coord_meta), node.time, len(node.coords)) for node in net.nodes}

    x_tracks = []
    y_tracks = []
    u_tracks = []
    v_tracks = []
    t_tracks = []
    color_tracks = []
    alpha_tracks = []

    valid_origins = []
    for origin in tqdm(net.origins[350:], desc='Collecting Valid Origins'):
        # the ones that are one-off events I don't want to plot
        if len(origin.future) > 0:           
            valid_origins.append(origin)

    print(f'{len(valid_origins)} Valid Origins Found')

    results = []

    for o in tqdm(valid_origins, desc=f'Extracting Tracks from {net.name}'):
        try:
            results.append(collect_drought_track(o, net.adj_dict, net_centroids, s_thresh, cmap))
        except:
            pass

    if len(results) != len(valid_origins):
        print(f'>>>>>>>>>> LOST {len(valid_origins) - len(results)} TRACKS <<<<<<<<<<<')
        
    for result in tqdm(results, desc='Reshaping and Packaging'):
        x_list, y_list, u_list, v_list, t_list, color_list, alpha_list = result

        x_tracks.append(x_list)
        y_tracks.append(y_list)
        u_tracks.append(u_list)
        v_tracks.append(v_list)
        t_tracks.append(t_list)
        color_tracks.append(color_list)
        alpha_tracks.append(alpha_list)


    return x_tracks, y_tracks, u_tracks, v_tracks, t_tracks, color_tracks, alpha_tracks

if __name__ == "__main__":

    # load in data
    print('Loading Data')
    dm_path = '/pool0/home/steinadi/data/drought/drought_impact/data/drought_measures'

    usdm = xr.open_dataset(f'{dm_path}/usdm/USDM_CONUS_105W_20000104_20220412.nc').load()
    print('... USDM loaded')

    intervals = ['14d', '30d', '90d', '180d', '270d', '1y', '2y', '5y', ]
    spi = xr.open_dataset(f'{dm_path}/spi/CONUS_105W/spi_usdmcat_CONUS_105W.nc').load()
    print('... SPI loaded')
    spei = xr.open_dataset(f'{dm_path}/spei/CONUS_105W/spei_usdmcat_CONUS_105W.nc').load()
    print('... SPEI loaded')
    eddi = xr.open_dataset(f'{dm_path}/eddi/CONUS_105W/eddi_usdmcat_CONUS_105W.nc').load()
    print('... EDDI loaded')
    pdsi = xr.open_dataset(f'{dm_path}/pdsi/CONUS_105W/pdsi.nc').load()
    print('... PDSI loaded')

    grace = xr.open_dataset(f'{dm_path}/grace/CONUS_105W/grace_usdmcat_CONUS_105W.nc').load()
    grace_vars = ['gws', 'rtzsm', 'sfsm']
    print('... GRACE loaded')

    # make some shorthand dictionaries
    dm_vars_expanded = {
        'usdm':['USDM'],
        'spi':[f'spi_{interval}' for interval in intervals],
        'spei':[f'spei_{interval}' for interval in intervals],
        'eddi':[f'eddi_{interval}' for interval in intervals],
        'pdsi':['pdsi'],
        'grace':grace_vars
    }

    all_dm_ds = {
        'usdm':usdm,
        'spi':spi,
        'spei':spei,
        'eddi':eddi,
        'pdsi':pdsi,
        'grace':grace
    }

    # compute drought networks if not already made
    for var in dm_vars_expanded.keys():
        for var_exp in dm_vars_expanded[var]:
            dnet_path = f'{dm_path}/ndrought_products/CONUS_105W/individual_dnet/{var_exp}_net.pickle'

            track_path = f'{dm_path}/ndrought_products/CONUS_105W/drought_tracks/{var_exp}_tracks.pickle'        

            if os.path.exists(dnet_path):
                var_dnet = dnet.DroughtNetwork.unpickle(dnet_path)            
            else:
                var_dnet = dnet.DroughtNetwork(all_dm_ds[var][var_exp].values, name=f'{var_exp.upper()} Drought Network')
                var_dnet.pickle(dnet_path)            

            # tossing in an override
            override = False
            if not os.path.exists(track_path) or override:

                try:
                    os.remove(track_path)
                except:
                    pass

                x_coords = all_dm_ds[var][var_exp].x.values
                y_coords = all_dm_ds[var][var_exp].y.values

                coord_meta = (
                    np.min(y_coords), np.max(y_coords), len(y_coords),
                    np.min(x_coords), np.max(x_coords), len(x_coords)
                )

                var_tracks = extract_drought_tracks(
                    net=var_dnet,
                    coord_meta=coord_meta,
                )

                f = open(track_path, 'wb')
                pickle.dump(var_tracks, f, pickle.HIGHEST_PROTOCOL)
                f.close()

                x_coords = None
                y_coords = None
                coord_meta = None
                var_tracks = None
                f = None

            var_dnet = None
            del all_dm_ds[var]

            gc.collect()

