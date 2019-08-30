"""
Author: Nitchakul Pipitvej
GitHub: wafflescore
"""

from fuzzy_clustering import FCM

from acse_9_irp_wafflescore import MiscHelpers as mh

from timeit import default_timer as timer
import time as tm
import numpy as np

import logging
import sys

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)

def fcm_compute(in_data, n_clusters, save_name="", save=False):
    """Computes the fuzzy c mean prediction

    Parameters
    ----------
    in_data : np.array or list
        data matrix
    n_clusters : int
        number of clusters
    save_name : str, optional
        the name which will be used to save the plot as png file,
    save : bool, optional
        flag whether to save the model, by default False

    Returns
    -------
    fuzzy_clustering.FCM
        an object of FCM class, see fuzzy_clustering.py for further details
    """
    start = timer()
    fcm = FCM(n_clusters=n_clusters)
    fcm.fit(in_data)
    predicted_mem = fcm.predict(in_data)
    stop = timer()

    logging.info("FCM elapsed time: %.6f", stop - start)

    if(save):
        # pickle the model
        if(save_name == ""):
            timestr = tm.strftime("%Y%m%d-%H%M%S")
            save_name = timestr
        fdir = save_name + '_model.p'
        mh.save_model(fcm, fdir)

    return fcm, predicted_mem


def plot_fcm(predicted_mem, x, z, save_name=""):
    """Plot all the fuzzy result from fuzzy c mean"""
    for i in range(predicted_mem.shape[1]):
        if(save_name != ""):
            save_path = save_name + '_pclass_' + \
                        str(i) + '.png'
            mh.plot_e_model(predicted_mem[:, i], x, z, cmap='Blues',
                         save_path=save_path)
        else:
            mh.plot_e_model(predicted_mem[:, i], x, z, cmap='Blues')


def get_best_fuzz(predicted_mem):
    """Retrieve the best prediction"""
    y_pred = predicted_mem.argmax(axis=1)
    return y_pred


def plot_best_fuzz(predicted_mem, x, z, save_name=""):
    """Plot the best resulf from fuzzy c mean"""
    y_pred = get_best_fuzz(predicted_mem)

    if(save_name != ""):
        save_path = save_name + '_bestFuzz_plot.png'
        mh.plot_e_model(y_pred, x, z, save_path=save_path, sep_label=True)
    else:
        mh.plot_e_model(y_pred, x, z, sep_label=True)

    return y_pred


def iter_n_class(in_data, in_range, save_name="", save=False, label=None):
    """Iterates number of cluster in FCM to plot the elbow

    Parameters
    ----------
    in_data : np.array or list
        data matrix
    in_range : class 'range'
        the range of number of clusters to iterate through
    save_name : str, optional
        the name which will be used to save, by default ""
    save : bool, optional
        flag whether to save the model, by default False
    label : np.array or list
        the true label of each data point

    Returns
    -------
    list
        list of all fcm objects
    list
        list of prediction members
    list
        sum square error result
    """
    fcms = []
    pred_mems = []

    it = len(in_range)

    SSE = np.zeros(it)
    avg_sils = np.full(it, np.nan)
    ch_scs = np.full(it, np.nan)
    if(label is not None):
        avg_ents = np.full(it, np.nan)
        avg_purs = np.full(it, np.nan)

    for i, c in enumerate(in_range):
        save_name = save_name + '_nclass_' + str(c)
        fcm, pred_mem = fcm_compute(in_data, c, save_name, save=save)
        fcms.append(fcm)
        pred_mems.append(pred_mem)
        SSE[i] = fcm.SSE(in_data)
        pred = get_best_fuzz(pred_mem)
        avg_sils[i] = mh.int_eval_silhouette(in_data, pred)
        ch_scs[i] = mh.cal_har_sc(in_data, pred)
        logging.info("sil=%.6f, chs=%.6f" % (avg_sils[i], ch_scs[i]))

        if(label is not None):
            avg_ents[i], avg_purs[i] = mh.ext_eval_entropy(label, pred, init_clus=-1) 
            logging.info("ent=%.6f, pur=%.6f" % (avg_ents[i], avg_purs[i]))

    mh.elbowplot(in_range, SSE)

    best_idx = []
    best_idx.append(np.nanargmax(np.array(avg_sils)))       # closest to 1
    best_idx.append(np.nanargmax(ch_scs))                   # higher = better
    if(label is not None):
        best_idx.append(np.nanargmin(np.array(avg_ents)))   # closest to 0
        best_idx.append(np.nanargmax(np.array(avg_purs)))   # closest to 1
    best_idx = np.unique(best_idx)

    return fcms, pred_mems, SSE, avg_sils, ch_scs, best_idx
