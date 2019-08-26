from MiscHelpers import *
from SOMsHelpers import *
from FCMHelpers import *
from dataPreprocessing import *
import hdbscan
from timeit import default_timer as timer


import sys


def main(argv):
    model = str(argv[0])
    col_name = (argv[1]).split()
    input_npz = None
    output_smooth_npz = None
    output_npz = None

    if(model == 'M1'):
        # Original Earth Model
        input_npz = np.load('../Synthetic Model/input_fields.npz')
        output_smooth_npz = np.load('../Synthetic Model/output_fields_smooth.npz')
        output_npz = np.load('../Synthetic Model/output_fields.npz')
    elif(model == 'M5a'):
        # Simplified Earth Model
        input_npz = np.load('../Synthetic Model/Model5a/input_fields.npz')
        output_smooth_npz = np.load('../Synthetic Model/Model5a/output_fields_smooth.npz')
        output_npz = np.load('../Synthetic Model/Model5a/output_fields.npz')
    elif(model == 'M5b'):
        # Simplified Earth Model -- less temperature anomaly
        input_npz = np.load('../Synthetic Model/Model5b/input_fields.npz')
        output_smooth_npz = np.load('../Synthetic Model/Model5b/output_fields_smooth.npz')
        output_npz = np.load('../Synthetic Model/Model5b/output_fields.npz')
    elif(model == 'M6'):
        # Simplified Earth Model -- less temperature anomaly
        output_smooth_npz = np.load('../Synthetic Model/Model6blind/output_fields_smooth.npz')
    elif(model == 'small'):
        input_npz = np.load('../Synthetic Model/input_fields.npz')
        output_smooth_npz = np.load('../Synthetic Model/output_fields_smooth.npz')
    else:
        # invalid model
        print('Invalid model', model)
        print('Data Preprocessing will now terminate.')
        exit()

    # convert npz into 1d, 2d numpy
    if(model != 'M6'):
        init_label = convLabel(input_npz['classes'])
    init_data = convData(output_smooth_npz)
    if(model == 'small'):
        init_data = init_data[:4000]
        init_label = init_label[:4000]
    # remove water and perform data preprocessing
    water_idx = []
    if(model != 'M6'):
        water_idx = np.where(init_label == 0)
        label = np.delete(init_label, water_idx)
    else:
        qp_idx = search_list(col_name, 'qp')
        water_idx = np.argwhere(np.isnan(init_data[:, qp_idx]))

    data = data_cleanup(init_data, water_idx, col_name, re_inf=-9999)
    logging.debug("Water removed shape: (%d, %d)" %
                  (data.shape[0], data.shape[1]))

    if (model):
        fdir = 'data/' + model + '_clean_data.npy'
        np.save(fdir, data)
        logging.info('Data saved at: %s' % fdir)

        if(model != 'M6'):
            fdir = 'data/' + model + '_data_label.npy'
            np.save(fdir, label)
            logging.info('Data label saved at: %s' % fdir)

        fdir = 'data/' + model + '_xz_pos.npy'
        np.save(fdir, data[:, -2:])
        logging.info('XZ positions saved at: %s' % fdir)

    means, stds = compMeanStd(data)
    norm_data = normalize(data, means, stds)

    test_missing_cols(norm_data, col_name, model=model)


def test_perform_SOM(in_data, som_fcm=False, n_clusters=10):
    start = timer()
    dim = compute_dim(in_data.shape[0])
    iter_cnt = 4000
    lr = 0.5124390316684666
    sigma = 2.189655172413793
    seed = 10

    som = som_assemble(in_data, seed, dim, lr, sigma)
    som.train_random(in_data, iter_cnt, verbose=False)
    u_matrix = som.distance_map().T
    watershed_bins = histedges_equalN(u_matrix.flatten())
    ws_labels = watershed_level(u_matrix, watershed_bins)
    n_map = som.neuron_map(in_data)

    cluster_labels = eval_ws(in_data, ws_labels, n_map)
    stop = timer()
    print("SOM elapsed time:", stop - start)

    if(som_fcm):
        _, fcm_cluster_labels = fcm_compute(n_map, n_clusters=n_clusters)

        stop = timer()
        print("SOM-FCM elapsed time:", stop - start)
        return cluster_labels, fcm_cluster_labels

    # may contain multiple best result from SOM
    return cluster_labels


def test_perform_HDBSCAN(in_data, min_cluster_size=10, min_samples=10):
    start = timer()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(in_data)
    stop = timer()
    print("HDBScan elapsed time:", stop - start)

    return cluster_labels


def test_missing_cols(in_data, col_name, model="-"):
    vp_idx = search_list(col_name, 'vp')
    vs_idx = search_list(col_name, 'vs')
    dn_idx = search_list(col_name, 'dn')
    vpvs_idx = search_list(col_name, 'vp/vs')
    qp_idx = search_list(col_name, 'qp')
    # qs_idx = search_list(col_name, 'qs')

    # when the only available data are: Vp, Vs, Vp/Vs
    test_data_1 = np.squeeze(in_data[:, [vp_idx, vs_idx, vpvs_idx]])

    test1_hdb = test_perform_HDBSCAN(test_data_1)
    sdir = model + '_test1_hdb.npy'
    np.save(sdir, test1_hdb)
    logging.info('Data saved at: %s' % sdir)

    _, test1_fcm = fcm_compute(test_data_1, 10)
    sdir = model + '_test1_fcm.npy'
    np.save(sdir, test1_fcm)
    logging.info('Data saved at: %s' % sdir)

    test1_som, test1_som_fcm = test_perform_SOM(test_data_1, som_fcm=True)
    sdir = model + '_test1_som.npy'
    np.save(sdir, test1_som)
    logging.info('Data saved at: %s' % sdir)
    sdir = model + '_test1_somfcm.npy'
    np.save(sdir, test1_som_fcm)
    logging.info('Data saved at: %s' % sdir)

    # when the only available data are: Vp, Qp, Density
    test_data_2 = np.squeeze(in_data[:, [vp_idx, qp_idx, dn_idx]])

    test2_hdb = test_perform_HDBSCAN(test_data_2)
    sdir = model + '_test2_hdb.npy'
    np.save(sdir, test2_hdb)
    logging.info('Data saved at: %s' % sdir)

    _, test2_fcm = fcm_compute(test_data_2, 10)
    sdir = model + '_test2_fcm.npy'
    np.save(sdir, test2_fcm)
    logging.info('Data saved at: %s' % sdir)

    test2_som, test2_som_fcm = test_perform_SOM(test_data_2, som_fcm=True)
    sdir = model + '_test2_som.npy'
    np.save(sdir, test2_som)
    logging.info('Data saved at: %s' % sdir)
    sdir = model + '_test2_somfcm.npy'
    np.save(sdir, test2_som_fcm)
    logging.info('Data saved at: %s' % sdir)

    # when the only available data are: Vp, Vs
    test_data_3 = np.squeeze(in_data[:, [vp_idx, vs_idx]])

    test3_hdb = test_perform_HDBSCAN(test_data_3)
    sdir = model + '_test3_hdb.npy'
    np.save(sdir, test3_hdb)
    logging.info('Data saved at: %s' % sdir)

    _, test3_fcm = fcm_compute(test_data_3, 10)
    sdir = model + '_test3_fcm.npy'
    np.save(sdir, test3_fcm)
    logging.info('Data saved at: %s' % sdir)

    test3_som, test3_som_fcm = test_perform_SOM(test_data_3, som_fcm=True)
    sdir = model + '_test3_som.npy'
    np.save(sdir, test3_som)
    logging.info('Data saved at: %s' % sdir)
    sdir = model + '_test3_somfcm.npy'
    np.save(sdir, test3_som_fcm)
    logging.info('Data saved at: %s' % sdir)

if __name__ == "__main__":
    main(sys.argv[1:])


# %run -i 'full_script.py' 'M1' "vp vs dn vp/vs qp qs x z" -- run in jpynb