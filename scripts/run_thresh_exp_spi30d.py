import os
import numpy as np

import multiprocessing
import subprocess

def run_script(script, config):
    command = f"python {script} {config}"
    try:
        subprocess.call(command, shell=True)
    except Exception as e:
        print(e)

if __name__ == "__main__":

    metric_thresh = 1
    #area_thresh = np.arange(0, 3000+200, 200)
    #ratio_thresh = np.arange(0, 1+0.1, 0.1)

    area_thresh = [600, 800, 1000, 1200, 1800]
    ratio_thresh = [0.3, 0.8, 0.4, 0.3, 0.5]

    # want to go from most strict to least strict thresholds
    area_thresh = np.flip(area_thresh)
    ratio_thresh = np.flip(ratio_thresh)

    thresh_exp_dir = '/pool0/home/steinadi/data/drought/drought_impact/data/thresh_experiments'

    num_workers = 16
    pool = multiprocessing.Pool(processes=num_workers)

    processes = []

    for a_thresh in area_thresh:
        for r_thresh in ratio_thresh:
            key = f'{a_thresh}a_0{int(r_thresh*10)}r_{metric_thresh}m'
            config = f'{thresh_exp_dir}/spi30d/config/config_{key}.yml'

            if not os.path.exists(f'{thresh_exp_dir}/spi30d/track/track_{key}.pickle'):
                pool.apply_async(run_script, args=('compute_drought_tracks_config.py', config))

    # Close the pool and wait for the processes or threads to finish
    pool.close()
    pool.join()

