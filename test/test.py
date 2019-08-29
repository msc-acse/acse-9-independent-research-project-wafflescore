"""
Author: Nitchakul Pipitvej
GitHub: wafflescore
"""

from acse_9_irp_wafflescore import MiscHelpers as mh
from acse_9_irp_wafflescore import SOMsHelpers as sh
from acse_9_irp_wafflescore import FCMHelpers as fh
from acse_9_irp_wafflescore import dataPreprocessing as dp

import hdbscan
from sklearn.cluster import KMeans
import numpy as np
from timeit import default_timer as timer

import logging
import sys

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)

def test_perform_HDBSCAN(in_data, min_cluster_size=40, min_samples=20,
                         som=False):
    # min_cluster_size of 40 and min_sample of 20 comes from parameter tuning
    # performed on HDBSCAN - Parameter Tuning Notebook

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(in_data)

    return cluster_labels


def test_perform_kmean(in_data, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=1000).fit(in_data)

    return kmeans.labels_


def test_perform_SOM(in_data, n_clusters=10):
    dim = sh.compute_dim(in_data.shape[0])
    iter_cnt = 4000
    lr = 0.5124390316684666
    sigma = 2.189655172413793
    seed = 10

    som = sh.som_assemble(in_data, seed, dim, lr, sigma)
    som.train_random(in_data, iter_cnt, verbose=False)
    u_matrix = som.distance_map().T
    watershed_bins = sh.histedges_equalN(u_matrix.flatten())
    ws_labels = sh.watershed_level(u_matrix, watershed_bins)
    n_map = som.neuron_map(in_data)

    cluster_labels, _, _ = sh.eval_ws(in_data, ws_labels, n_map)

    _, fcm_cluster_labels = fh.fcm_compute(n_map, n_clusters=n_clusters)

    hdb_cluster_labels = test_perform_HDBSCAN(n_map, som=True)

    # may contain multiple result from SOM
    return cluster_labels, fcm_cluster_labels, hdb_cluster_labels


def test_run(in_data, model="", test=""):
    test_hdb = test_perform_HDBSCAN(in_data)
    logging.info("HDBSCAN done.")

    _, test_fcm = fh.fcm_compute(in_data, 10)
    logging.info("FCM done.")

    test_som, test_som_fcm, test_som_hdb = test_perform_SOM(in_data)
    logging.info("SOMs done.")

    test_kmn = test_perform_kmean(in_data)
    logging.info("KMEAN done.")

    return test_hdb, test_fcm, test_som, test_som_fcm, test_som_hdb, test_kmn


def main(argv):
    model = str(argv[0])

    cdir = '../data/' + model + '_clean_data.npy'
    data = np.load(cdir)

    means, stds = dp.compMeanStd(data)
    norm_data = dp.normalize(data, means, stds)

    means, stds = dp.compMeanStd(data)
    norm_data = dp.normalize(data, means, stds)

    _ = test_run(norm_data, model=model, test=test_case)

    logging.info("Whole system test done.")


if __name__ == "__main__":
    main(sys.argv[1:])


# %run -i 'test.py' 'M1'