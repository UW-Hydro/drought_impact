import pickle
import numpy as np

import sys
sys.path.append('../')
import drought.drought_impact.ndrought.wrangle as dnw
sys.setrecursionlimit(100000)

from tqdm.autonotebook import tqdm

class TrackNode():

    def __init__(self, track, file, ti):
        """

        Parameters
        ----------
        file: str
            Which file the tracks come from
        ti: int
            Which track index within a dtd (drought track dictionary)
        """
        self.track = track
        self.track_set = set(track)
        self.file = file
        self.ti = ti
        file_args = file.split('_')
        self.at = np.float64(file_args[1][:-1])
        self.rt = np.float64(file_args[2][:-1])/10
        self.next = []
        self.last = []
        
    def check_intersect(self, other):
        return len(self.track_set.intersection(other.track_set)) > 0
    
if __name__ == '__main__':

    cwd = '/pool0/home/steinadi/data/drought/drought_impact/data/thresh_experiments'
    track_dir = f'{cwd}/spi30d/track'

    all_dtd = {}
    all_summ = {}
    all_summ_grp = {}

    metric_thresh = 1
    area_thresh = np.arange(400, 3000+200, 200)
    ratio_thresh = np.arange(0, 1+0.1, 0.1)

    files = []
    for a_thresh in area_thresh:
            for r_thresh in ratio_thresh:
                key = f'{a_thresh}a_0{int(r_thresh*10)}r_{metric_thresh}m'
                files.append(f'track_{key}.pickle')

    for file in tqdm(files, desc="Creating dtd's"):
        dtd = dnw.convert_pickle_to_dtd(f'{track_dir}/{file}')
        for key in dtd.keys():
            var_tracks = dtd[key]
            trimmed_tracks = []
            for track in var_tracks:
                if len(track) > 0:
                    trimmed_tracks.append(track)
            dtd[key] = trimmed_tracks
        dtd = dnw.prune_tracks(dtd)

        summ, summ_grp = dnw.compute_track_summary_characterization(dtd, 5)

        all_dtd[file] = dtd
        all_summ[file] = summ
        all_summ_grp[file] = summ_grp

    all_centroids = {}
    for file in files:
        dtd = all_dtd[file]
        dtd_centroids = []
        for x_track, y_track in zip(dtd['x'], dtd['y']):
            track_centroids = []
            for x, y in zip(x_track, y_track):
                track_centroids.append((x,y))
            dtd_centroids.append(track_centroids)
        all_centroids[file] = dtd_centroids

    all_tracknodes = {}

    for file in files:
        file_centroids = all_centroids[file]
        file_tracknodes = []
        for i, track in enumerate(file_centroids):
            file_tracknodes.append(TrackNode(track, file, i))
        all_tracknodes[file] = file_tracknodes

    t = tqdm(total=len(files)**2)

    for file_a in files:
        a_nodes = all_tracknodes[file_a]
        for file_b in files:
            if file_a != file_b:
                b_nodes = all_tracknodes[file_b]
                for node_a in a_nodes:
                    for node_b in b_nodes:
                        if node_a.check_intersect(node_b):
                            del_rt = node_b.rt - node_a.rt
                            del_at = node_b.at - node_a.at
                            # want connections to be only 1 change away,
                            # so either rt changes or at changes, 
                            # not both
                            if del_at == 200 and del_rt == 0:
                                # then a goes to b
                                #print(node_a, node_b, del_rt, del_at)
                                node_a.next.append(node_b)
                                node_b.last.append(node_a)
                            elif del_at == 0:
                                if del_rt == 0.1:
                                    #print(node_a, node_b, del_rt, del_at)
                                    node_a.next.append(node_b)
                                    node_b.last.append(node_a)
                                elif del_rt == - 0.1:
                                    # can't be equivalent, so something must be different
                                    node_b.next.append(node_a)
                                    node_a.last.append(node_b)
                                elif del_rt == 0:
                                    raise Exception(f'Equivalency error {del_rt}')
                            elif del_at == -200 and del_rt == 0:
                                node_b.next.append(node_a)
                                node_a.last.append(node_b)
            t.update()

    f = open(f'{cwd}/spi30d/tracknodes.pickle', 'wb')
    pickle.dump(all_tracknodes, f, pickle.HIGHEST_PROTOCOL)
    f.close()