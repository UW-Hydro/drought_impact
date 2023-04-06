import xarray as xr
import numpy as np

from tqdm.autonotebook import tqdm

usdm_da = xr.open_dataset('/pool0/home/steinjao/data/drought/drought_impact/data/drought_measures/usdm/USDM_CONUS_105W_20000104_20220412.nc')['USDM'].load()

usdm_da_windowed = usdm_da.coarsen({'x':10, 'y':10}, boundary='trim').construct(
    x=("x_coarse", "x_window"),
    y=("y_coarse", "y_window")
)

def mode(a):
    uniq = np.unique(a, return_counts=True)
    return uniq[0][np.argmax(uniq[1])]

def threshold(a, thresh):
    b = a.copy()
    b[b<thresh] = 0
    b[b>=thresh] = 1
    return b

ny = usdm_da_windowed['y_coarse']
nx = usdm_da_windowed['x_coarse']
times = usdm_da_windowed['time']

prog = tqdm(total=(len(ny)*len(nx)))

all_d1_subs = []
all_d2_subs = []
all_d3_subs = []
all_d4_subs = []
for t in times:
    for yy in ny:
        for xx in nx:
            window = usdm_da_windowed.sel(time='2015-01-06', y_coarse=yy, x_coarse=xx).values
            all_d1_subs.append(mode(threshold(window, 1)))
            all_d2_subs.append(mode(threshold(window, 2)))
            all_d3_subs.append(mode(threshold(window, 3)))
            all_d4_subs.append(mode(threshold(window, 4)))
            window = None
            prog.update()

d1_reshaped = np.reshape(np.array(all_d1_subs), (len(times), len(ny), len(nx)))
d2_reshaped = np.reshape(np.array(all_d2_subs), (len(times), len(ny), len(nx)))
d3_reshaped = np.reshape(np.array(all_d3_subs), (len(times), len(ny), len(nx)))
d4_reshaped = np.reshape(np.array(all_d4_subs), (len(times), len(ny), len(nx)))

coarse_thresh_usdm = xr.Dataset(
    data_vars=dict(
        d1_thresh=(["time", "y", "x"], d1_reshaped),
        d2_thresh=(["time", "y", "x"], d2_reshaped),
        d3_thresh=(["time", "y", "x"], d3_reshaped),
        d4_thresh=(["time", "y", "x"], d4_reshaped)
    ),
    coords = dict(
        time=usdm_da_windowed.time.values,
        y=usdm_da_windowed.y_coarse.values,
        x=usdm_da_windowed.x_coarse.values
    ),
    attrs=usdm_da_windowed.attrs
)

coarse_thresh_usdm.to_netcdf('/pool0/home/steinadi/data/drought/drought_impact/data/drought_measures/usdm/USDM_CONUS_105W_20000104_20220412_25km.nc')