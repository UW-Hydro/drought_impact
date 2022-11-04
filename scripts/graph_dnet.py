import numpy as np
import xarray as xr
import pandas as pd
import networkx as nx
from pyvis.network import Network

import sys
sys.path.append('../')
import ndrought.wrangle as wrangle
import ndrought.compare as compare
import ndrought.plotting as ndplot
import ndrought.drought_network as dnet

# load in data
dm_path = '/pool0/home/steinadi/data/drought/drought_impact/data/drought_measures'
usdm = xr.open_dataset(f'{dm_path}/usdm/USDM_WA_20000104_20220412.nc')
usdm_net = dnet.DroughtNetwork.unpickle(f'{dm_path}/usdm/usdm_WA_net.pickle')
pdsi = xr.open_dataset(f'{dm_path}/pdsi/WA/pdsi_usdmcat_WA.nc')
pdsi_net = dnet.DroughtNetwork.unpickle(f'{dm_path}/pdsi/WA/pdsi_WA_net.pickle')

matched_dates = compare.pair_to_usdm_date(pd.to_datetime(usdm.date.values), pd.to_datetime(pdsi.day.values), 'PDSI', realign=True)

pdsi_date_idx = []

for date in matched_dates['PDSI'].values:
    pdsi_date_idx.append(np.where(pdsi.day.values == date)[0][0])
usdm_date_idx = []

for date in matched_dates['USDM Date'].values:
    usdm_date_idx.append(np.where(usdm.date.values == date)[0][0])
    
matched_dates_dict = dict()
for usdm_date, pdsi_date in zip(usdm_date_idx, pdsi_date_idx):
    matched_dates_dict[usdm_date] = pdsi_date
    
overlap_events = usdm_net.find_overlapping_nodes_events(pdsi_net, matched_dates_dict)

# filter
usdm_adj_filtered = usdm_net.filter_adj_dict_by_area(20)
pdsi_adj_filtered = pdsi_net.filter_adj_dict_by_area(20)
G_usdm_filtered = ndplot.weight_by_area_ratio(usdm_net, usdm_adj_filtered)
G_psdi_filtered = ndplot.weight_by_area_ratio(pdsi_net, pdsi_adj_filtered)
G_usdm_filtered = ndplot.attach_dm_node_label(G_usdm_filtered, 'usdm')
G_pdsi_filtered = ndplot.attach_dm_node_label(G_psdi_filtered, 'pdsi')

G_usdm_pdsi_filtered = nx.union(G_usdm_filtered, G_pdsi_filtered)
G_usdm_pdsi_filtered = ndplot.connect_overlap_nodes(G_usdm_pdsi_filtered, overlap_events, 'usdm', 'pdsi')
g = Network()
g.from_nx(G_usdm_filtered.copy())
g.show('example.html')