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

from numpy.testing import assert_almost_equal, assert_array_almost_equal
from numpy.testing import assert_array_equal
import numpy as np

class unit_test():        
    def test_ent_pur(self):
        a = [1, 1, 1, 1]
        b = [1, 2, 1, 2]
        c = [1, 1, 2, 2]
        assert mh.ext_eval_entropy(a, a) == (0.0, 1.0)
        assert mh.ext_eval_entropy(a, b) == (0.0, 1.0)
        assert_almost_equal(mh.ext_eval_entropy(b, c),(0.693147180, 0.5))

    def test_replace_nan_inf(self):
        a = [[np.nan, 1], [1, 1]]
        b = [[np.inf, 1], [1, 1]]
        
        assert_array_equal((dp.replace_nan_inf(a)), [[0, 1], [1, 1]])
        assert_array_equal((dp.replace_nan_inf(b, re_inf=10)), [[10, 1], [1, 1]])
    
    def test_convLabel(self):
        a = np.array([[1,1],[2,2]])
        assert_array_equal(dp.convLabel(a), [1,1,2,2])
        
    def test_data_cleanup(self):
        data = [[1000,2000,3000,100,100,1000,np.inf,np.nan]]
        col_n = ['vp', 'vs', 'dn', 'vp/vs', 'qp', 'qs', 'x', 'z']
        assert_array_almost_equal(dp.data_cleanup(data, col_n, -9), [[1, 2, 3, 10, 4.60517019, 6.90775528, -9, 0]])
    
    def test_compMeanStd(self):
        d = [[1, 1], 
             [2, 2]]
        assert_array_equal(dp.compMeanStd(d), [[1.5, 1.5], [0.5, 0.5]])
        
    def test_normalize(self):
        mean = [1.5, 1.5]
        std = [0.5, 0.5]
        d = np.array([[1.0, 1.0], 
                      [2.0, 2.0]])
        assert_array_equal(dp.normalize(d, mean, std), np.array([[-1, -1],[1, 1]]))
    
    def test_histedge(self):
        d = np.array([1,2,3,4,5])
        assert_array_equal(sh.histedges_equalN(d, nbin=2), [1, 3.5, 5])
        
    def test_closest_n(self):
        d = np.array([[1, 0]])
        assert_array_equal(sh.closest_n(d), np.array([[1, 1]]))
        
    def test_KNN(self):
        d = np.array([[1,1,2],
                      [0,1,2],
                      [1,1,2]])
        assert_array_equal(sh.KNN(d), [[1,1,2],[1,1,2],[1,1,2]])


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
