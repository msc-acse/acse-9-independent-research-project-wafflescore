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

    start = timer()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(in_data)
    stop = timer()
    logging.info("HDBScan elapsed time: %.6f", stop - start)

    return cluster_labels


def test_perform_kmean(in_data, n_clusters=10):
    start = timer()
    kmeans = KMeans(n_clusters=n_clusters, max_iter=1000).fit(in_data)
    stop = timer()
    logging.info("KMean elapsed time: %.6f", stop - start)

    return kmeans.labels_


def test_perform_SOM(in_data, n_clusters=10):
    start = timer()
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
    stop = timer()
    som_time = stop - start
    logging.info("SOM elapsed time: %.6f", som_time)

    _, fcm_cluster_labels = fh.fcm_compute(n_map, n_clusters=n_clusters)
    stop = timer()
    logging.info("SOM-FCM elapsed time: %.6f", stop - start)

    start = timer()
    hdb_cluster_labels = test_perform_HDBSCAN(n_map, som=True)
    stop = timer()
    logging.info("SOM-HDBScan elapsed time: %.6f", (stop - start) + som_time)

    # may contain multiple result from SOM
    return cluster_labels, fcm_cluster_labels, hdb_cluster_labels


def test_run(in_data, model="", test=""):
    test_hdb = test_perform_HDBSCAN(in_data)
    sdir = model + '_' + test + '_hdb.npy'
    np.save(sdir, test_hdb)
    logging.info('Data saved at: %s' % sdir)

    _, test_fcm = fh.fcm_compute(in_data, 10)
    sdir = model + '_' + test + '_fcm.npy'
    np.save(sdir, test_fcm)
    logging.info('Data saved at: %s' % sdir)

    test_som, test_som_fcm, test_som_hdb = test_perform_SOM(in_data)
    sdir = model + '_' + test + '_som.npy'
    np.save(sdir, test_som)
    logging.info('Data saved at: %s' % sdir)
    sdir = model + '_' + test + '_somfcm.npy'
    np.save(sdir, test_som_fcm)
    logging.info('Data saved at: %s' % sdir)
    sdir = model + '_' + test + '_somhdb.npy'
    np.save(sdir, test_som_hdb)
    logging.info('Data saved at: %s' % sdir)

    kmf = test_perform_kmean(in_data)
    sdir = model + '_' + test + '_kmn.npy'
    np.save(sdir, kmf)
    logging.info('Data saved at: %s' % sdir)


def test_missing_cols(in_data, col_name, model="-"):
    vp_idx = mh.search_list(col_name, 'vp')
    vs_idx = mh.search_list(col_name, 'vs')
    dn_idx = mh.search_list(col_name, 'dn')
    vpvs_idx = mh.search_list(col_name, 'vp/vs')
    qp_idx = mh.search_list(col_name, 'qp')
    # qs_idx = mh.search_list(col_name, 'qs')

    # when the only available data are: Vp, Vs, Vp/Vs
    test_data_1 = np.squeeze(in_data[:, [vp_idx, vs_idx, vpvs_idx]])

    test1_hdb = test_perform_HDBSCAN(test_data_1)
    sdir = model + '_test1_hdb.npy'
    np.save(sdir, test1_hdb)
    logging.info('Data saved at: %s' % sdir)

    _, test1_fcm = fh.fcm_compute(test_data_1, 10)
    sdir = model + '_test1_fcm.npy'
    np.save(sdir, test1_fcm)
    logging.info('Data saved at: %s' % sdir)

    test1_som, test1_som_fcm, test1_som_hdb = test_perform_SOM(test_data_1)
    sdir = model + '_test1_som.npy'
    np.save(sdir, test1_som)
    logging.info('Data saved at: %s' % sdir)
    sdir = model + '_test1_somfcm.npy'
    np.save(sdir, test1_som_fcm)
    logging.info('Data saved at: %s' % sdir)
    sdir = model + '_test1_somhdb.npy'
    np.save(sdir, test1_som_hdb)
    logging.info('Data saved at: %s' % sdir)

    # when the only available data are: Vp, Qp, Density
    test_data_2 = np.squeeze(in_data[:, [vp_idx, qp_idx, dn_idx]])

    test2_hdb = test_perform_HDBSCAN(test_data_2)
    sdir = model + '_test2_hdb.npy'
    np.save(sdir, test2_hdb)
    logging.info('Data saved at: %s' % sdir)

    _, test2_fcm = fh.fcm_compute(test_data_2, 10)
    sdir = model + '_test2_fcm.npy'
    np.save(sdir, test2_fcm)
    logging.info('Data saved at: %s' % sdir)

    test2_som, test2_som_fcm, test2_som_hdb = test_perform_SOM(test_data_2)
    sdir = model + '_test2_som.npy'
    np.save(sdir, test2_som)
    logging.info('Data saved at: %s' % sdir)
    sdir = model + '_test2_somfcm.npy'
    np.save(sdir, test2_som_fcm)
    logging.info('Data saved at: %s' % sdir)
    sdir = model + '_test2_somhdb.npy'
    np.save(sdir, test2_som_hdb)
    logging.info('Data saved at: %s' % sdir)

    # when the only available data are: Vp, Vs
    test_data_3 = np.squeeze(in_data[:, [vp_idx, vs_idx]])

    test3_hdb = test_perform_HDBSCAN(test_data_3)
    sdir = model + '_test3_hdb.npy'
    np.save(sdir, test3_hdb)
    logging.info('Data saved at: %s' % sdir)

    _, test3_fcm = fh.fcm_compute(test_data_3, 10)
    sdir = model + '_test3_fcm.npy'
    np.save(sdir, test3_fcm)
    logging.info('Data saved at: %s' % sdir)

    test3_som, test3_som_fcm, test3_som_hdb = test_perform_SOM(test_data_3)
    sdir = model + '_test3_som.npy'
    np.save(sdir, test3_som)
    logging.info('Data saved at: %s' % sdir)
    sdir = model + '_test3_somfcm.npy'
    np.save(sdir, test3_som_fcm)
    logging.info('Data saved at: %s' % sdir)
    sdir = model + '_test3_somhdb.npy'
    np.save(sdir, test3_som_hdb)
    logging.info('Data saved at: %s' % sdir)

def test_missing_kmean(in_data, col_name, model="-"):
    vp_idx = mh.search_list(col_name, 'vp')
    vs_idx = mh.search_list(col_name, 'vs')
    dn_idx = mh.search_list(col_name, 'dn')
    vpvs_idx = mh.search_list(col_name, 'vp/vs')
    qp_idx = mh.search_list(col_name, 'qp')
    # qs_idx = mh.search_list(col_name, 'qs')

    # when the only available data are: Vp, Vs, Vp/Vs
    test_data_1 = np.squeeze(in_data[:, [vp_idx, vs_idx, vpvs_idx]])

    test1_kmean = test_perform_kmean(test_data_1)
    sdir = model + '_test1_kmn.npy'
    np.save(sdir, test1_kmean)
    logging.info('Data saved at: %s' % sdir)

    # when the only available data are: Vp, Qp, Density
    test_data_2 = np.squeeze(in_data[:, [vp_idx, qp_idx, dn_idx]])

    test2_kmean = test_perform_kmean(test_data_2)
    sdir = model + '_test2_kmn.npy'
    np.save(sdir, test2_kmean)
    logging.info('Data saved at: %s' % sdir)

    # when the only available data are: Vp, Vs
    test_data_3 = np.squeeze(in_data[:, [vp_idx, vs_idx]])

    test3_kmean = test_perform_kmean(test_data_3)
    sdir = model + '_test3_kmn.npy'
    np.save(sdir, test3_kmean)
    logging.info('Data saved at: %s' % sdir)


def test_missing_hdbscan(in_data, col_name, model="-"):
    vp_idx = mh.search_list(col_name, 'vp')
    vs_idx = mh.search_list(col_name, 'vs')
    dn_idx = mh.search_list(col_name, 'dn')
    vpvs_idx = mh.search_list(col_name, 'vp/vs')
    qp_idx = mh.search_list(col_name, 'qp')
    # qs_idx = mh.search_list(col_name, 'qs')

    # when the only available data are: Vp, Vs, Vp/Vs
    test_data_1 = np.squeeze(in_data[:, [vp_idx, vs_idx, vpvs_idx]])

    test1_hdb = test_perform_HDBSCAN(test_data_1)
    sdir = model + '_test1_rehdb.npy'
    np.save(sdir, test1_hdb)
    logging.info('Data saved at: %s' % sdir)

    # when the only available data are: Vp, Qp, Density
    test_data_2 = np.squeeze(in_data[:, [vp_idx, qp_idx, dn_idx]])

    test2_hdb = test_perform_HDBSCAN(test_data_2)
    sdir = model + '_test2_rehdb.npy'
    np.save(sdir, test2_hdb)
    logging.info('Data saved at: %s' % sdir)

    # when the only available data are: Vp, Vs
    test_data_3 = np.squeeze(in_data[:, [vp_idx, vs_idx]])

    test3_hdb = test_perform_HDBSCAN(test_data_3)
    sdir = model + '_test3_rehdb.npy'
    np.save(sdir, test3_hdb)
    logging.info('Data saved at: %s' % sdir)


def main(argv):
    model = str(argv[0])
    col_name = (argv[1]).split()
    test_case = str(argv[2])

    cdir = '../data/' + model + '_clean_data.npy'
    data = np.load(cdir)

    means, stds = dp.compMeanStd(data)
    norm_data = dp.normalize(data, means, stds)

    means, stds = dp.compMeanStd(data)
    norm_data = dp.normalize(data, means, stds)

    if(test_case == "full"):
        test_run(norm_data, model=model, test=test_case)
    elif(test_case == "no_xz"):
        test_run(norm_data[:,:-2], model=model, test=test_case)
    elif(test_case == "missing"):
        test_missing_cols(norm_data, col_name, model=model)
    elif(test_case == "kmean"):
        test_missing_kmean(norm_data, col_name, model=model)

        kmf = test_perform_kmean(norm_data)
        sdir = model + '_full_kmn.npy'
        np.save(sdir, kmf)
        logging.info('Data saved at: %s' % sdir)

        kml = test_perform_kmean(norm_data[:,:-2])
        sdir = model + '_no_xz_kmn.npy'
        np.save(sdir, kml)
        logging.info('Data saved at: %s' % sdir)


if __name__ == "__main__":
    main(sys.argv[1:])


# %run -i 'test.py' 'M1' "vp vs dn vp/vs qp qs x z" "full" -- run in jpynb