import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import os
import sys
sys.path.append('../')
import ndrought.drought_network_v12 as dnet
import multiprocessing as mp


from queue import Queue
import gc
import pickle
sys.setrecursionlimit(int(1e4))

from dask.distributed import Client, LocalCluster, wait
import dask

import yaml

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

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

if __name__ == "__main__":

    cluster = LocalCluster(n_workers=4, threads_per_worker=1)

    client = Client(cluster)
    print(client.dashboard_link)

    #Load in config
    config = read_yaml(sys.argv[1])
    data_dir = config['DIR']['DATA']
    dnet_dir = config['DIR']['DNET']
    track_dir = config['DIR']['TRACK']
    log_dir = config['DIR']['LOG']

    metric_thresh = np.float64(config['THRESH']['METRIC'])
    area_thresh = np.float64(config['THRESH']['AREA'])
    ratio_thresh = np.float64(config['THRESH']['RATIO'])

    # load in data
    print('Lazy Data Load')
    dm_path = '/pool0/home/steinadi/data/drought/drought_impact/data/drought_measures'

    data = xr.open_dataarray(data_dir)
    print('... Data ready')

    # compute drought networks if not already made     

    if os.path.exists(dnet_dir):
        var_dnet = dnet.DroughtNetwork.unpickle(dnet_dir)            
    else:
        var_dnet = dnet.DroughtNetwork(data.values, name=f'{data.name.upper()} Drought Network', area_threshold=area_thresh, threshold=metric_thresh)
        var_dnet.pickle(dnet_dir)            

    # tossing in an override
    override = False
    if not os.path.exists(track_dir) or override:

        try:
            os.remove(track_dir)
        except:
            pass

        print(f'Loading {data.name} ...')
        data.load()
        print(f'... Done')

        x_coords = data.x.values
        y_coords = data.y.values

        coord_meta = (
            np.min(y_coords), np.max(y_coords), len(y_coords),
            np.min(x_coords), np.max(x_coords), len(x_coords)
        )

        var_tracks = extract_drought_tracks(
            net=var_dnet,
            coord_meta=coord_meta,
            client=client,
            log_dir=log_dir,
            ratio_thresh=ratio_thresh
        )

        f = open(track_dir, 'wb')
        pickle.dump(var_tracks, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        x_coords = None
        y_coords = None
        coord_meta = None
        var_tracks = None
        f = None

    var_dnet = None
    del data

    gc.collect()

