"""
Author: Nitchakul Pipitvej
GitHub: wafflescore
"""
import random
from acse_9_irp_wafflescore import MiscHelpers as mh
import numpy as np
import hdbscan
from timeit import default_timer as timer


import logging
import sys

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)

def compute_hdb(in_data, min_cluster_size, min_samples):
    start = timer()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(in_data)
    stop = timer()
    logging.info("HDBScan elapsed time: %.6f", stop - start)

    return clusterer, cluster_labels


def random_search_hdb(in_data, init_guess, max_eval=20, label=None, seed=10,
                      rand_range=(10,10)):
    random.seed(seed)

    g_min_size, g_min_sam = init_guess
    
    g_min_size = g_min_size - rand_range[0] if g_min_size - rand_range[0] > 5 else 5
    g_min_sam = g_min_sam - rand_range[1] if g_min_sam - rand_range[1] > 5 else 5

    param_grid = {
        'g_min_size': list(range(g_min_size, g_min_size+rand_range[0])),
        'g_min_sam': list(range(g_min_sam, g_min_sam+rand_range[1]))
    }
    
    min_size = np.zeros(max_eval)
    min_sam = np.zeros(max_eval)
    avg_sils = np.full(max_eval, np.nan)
    ch_scs = np.full(max_eval, np.nan)
    cluster_labels = np.zeros((max_eval, len(in_data)))

    if(label is not None):
        avg_ents = np.full(max_eval, np.nan)
        avg_purs = np.full(max_eval, np.nan)
    
    i = 0
    while i < max_eval:
        random_params = {k: random.sample(v, 1)[0]
                            for k, v in param_grid.items()}
        min_size[i], min_sam[i] = list(random_params.values())
        clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_size[i]), min_samples=int(min_sam[i]), memory='cache')
        cluster_labels[i] = clusterer.fit_predict(in_data)
        n_clusters = len(np.unique(cluster_labels[i]))
        if(n_clusters < 5 or n_clusters > 30):
            logging.info("Random search using min_size = %d, min_sam = %d result to very small / large number of clusters (n_clusters = %d)" % (int(min_size[i]), int(min_sam[i]), n_clusters))
            continue
        
        avg_sils[i] = mh.int_eval_silhouette(in_data, cluster_labels[i])
        ch_scs[i] = mh.cal_har_sc(in_data, cluster_labels[i])
        logging.info("min_size=%d, min_sam=%d, sil=%.6f, ch=%.6f" % (int(min_size[i]), int(min_sam[i]), avg_sils[i], ch_scs[i]))
        
        if(label is not None):
            avg_ents[i], avg_purs[i] = mh.ext_eval_entropy(label, cluster_labels[i], init_clus=-1) 
            logging.info("ent=%.6f, pur=%.6f" % (avg_ents[i], avg_purs[i]))
            
        i += 1
    
    best_idx = []
    best_idx.append(np.nanargmax(np.array(avg_sils)))       # closest to 1
    best_idx.append(np.nanargmax(ch_scs))                   # higher = better
    if(label is not None):
        best_idx.append(np.nanargmin(np.array(avg_ents)))   # closest to 0
        best_idx.append(np.nanargmax(np.array(avg_purs)))   # closest to 1
    best_idx = np.unique(best_idx)
    return (cluster_labels[best_idx], avg_sils[best_idx],
            ch_scs[best_idx], min_size[best_idx], min_sam[best_idx])