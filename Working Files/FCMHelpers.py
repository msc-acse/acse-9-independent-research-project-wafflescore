#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 02:09:50 2019

@author: laveniier
"""

from fuzzy_clustering import FCM

from MiscHelpers import save_model, plot_e_model, elbowplot
from MiscHelpers import int_eval_silhouette, ext_eval_entropy
from timeit import default_timer as timer
import time as tm
# from fuzzy-c-means-master.fuzzycmeans.fuzzy_clustering import FCM

# import importlib.util
# spec = importlib.util.spec_from_file_location("FCM", "./fuzzy-c-means-master/fuzzycmeans/fuzzy_clustering.py")
# fcm_module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(fcm_module)


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

    print("FCM elapsed time:", stop - start)

    if(save):
        # pickle the model
        if(save_name == ""):
            timestr = tm.strftime("%Y%m%d-%H%M%S")
            save_name = timestr
        fdir = 'FCM_results/' + save_name + '_model.p'
        save_model(fcm, fdir)

    return fcm, predicted_mem


def plot_fcm(predicted_mem, x, z, save_name=""):
    """Plot all the fuzzy result from fuzzy c mean"""
    for i in range(predicted_mem.shape[1]):
        if(save_name != ""):
            save_dir = 'FCM_results/' + save_name + '_pclass_' + \
                       str(i) + '.png'
            plot_e_model(predicted_mem[:, i], x, z, cmap='Blues',
                         save_dir=save_dir)
        else:
            plot_e_model(predicted_mem[:, i], x, z, cmap='Blues')


def get_best_fuzz(predicted_mem):
    """Retrieve the best prediction"""
    y_pred = predicted_mem.argmax(axis=1)
    return y_pred


def plot_best_fuzz(predicted_mem, x, z, save_name=""):
    """Plot the best resulf from fuzzy c mean"""
    y_pred = get_best_fuzz(predicted_mem)

    if(save_name != ""):
        save_dir = 'FCM_results/' + save_name + '_bestFuzz_plot.png'
        plot_e_model(y_pred, x, z, save_dir=save_dir, sep_label=True)
    else:
        plot_e_model(y_pred, x, z, sep_label=True)

    return y_pred


def iter_n_class(in_data, in_range, save_name="", save=False, label=None):
    """Iterates number of cluster in FCM to plot the elbow

    Parameters
    ----------
    in_data : np.array or list
        data matrix
    in_range : class 'range'
        the range of number of clusters to iterate through
    x : np.array or list
        list like array of x position
    z : np.array or list
        list like array of z position
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
    SSE = []
    for c in in_range:
        save_name = save_name + '_nclass_' + str(c)
        fcm, pred_mem = fcm_compute(in_data, c, save_name, save=save)
        fcms.append(fcm)
        pred_mems.append(pred_mem)
        SSE.append(fcm.SSE(in_data))
        y_pred = get_best_fuzz(pred_mem)

        if(save):
            save_dir = 'FCM_results/' + save_name + '_plot.png'
            plot_e_model(y_pred, x, z, save_dir=save_dir)
        else:
            plot_e_model(y_pred, x, z)

    elbowplot(in_range, SSE)

    return fcms, pred_mems, SSE
