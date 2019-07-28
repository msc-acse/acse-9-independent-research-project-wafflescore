#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 02:09:50 2019

@author: laveniier
"""

from fuzzy_c_means_master.fuzzycmeans.fuzzy_clustering import FCM

from MiscHelpers import save_model, show_new_class, cluster_iden
from timeit import default_timer as timer
import time as tm
#from fuzzy-c-means-master.fuzzycmeans.fuzzy_clustering import FCM

#import importlib.util
#spec = importlib.util.spec_from_file_location("FCM", "./fuzzy-c-means-master/fuzzycmeans/fuzzy_clustering.py")
#fcm_module = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(fcm_module)


def fcm_compute(in_data, n_clusters, model_name="", save=False):
    start = timer()
    fcm = FCM(n_clusters=n_clusters)
    fcm.fit(in_data)
    predicted_mem = fcm.predict(in_data)
    stop = timer()

    print("elapsed time:", stop - start)

    if(save):
        # pickle the model
        timestr = tm.strftime("%Y%m%d-%H%M%S")
        if(model_name == ""):
            model_name = timestr
        fdir = 'FCM_results/' + model_name + '_model.p'
        save_model(fcm, fdir)

    return fcm, predicted_mem


def plot_fuzzy(predicted_mem, x, z, save_name=""):
    for i in range(predicted_mem.shape[1]):
        if(save_name != ""):
            save_dir = 'FCM_results/' + save_name + '_pclass_' + \
                       str(i) + '.png'
            show_new_class(predicted_mem[:, i], x, z, cmap='Blues',
                           save_dir=save_dir)
        else:
            show_new_class(predicted_mem[:, i], x, z, cmap='Blues')


def get_best_fuzz(mem_preds):
    y_pred = mem_preds.argmax(axis=1)
    return y_pred


def plot_best_fuzz(mem_preds, x, z, save_name=""):
    y_pred = get_best_fuzz(mem_preds)

    if(save_name != ""):
        save_dir = 'FCM_results/' + save_name + '_bestFuzz_plot.png'
        show_new_class(y_pred, x, z, save_dir=save_dir, sep_label=True)
    else:
        show_new_class(y_pred, x, z, sep_label=True)

    return y_pred


def iter_n_class(in_data, in_range, x, z, label=None, m_name=""):
    fcms = []
    pred_mems = []
    for c in in_range:
        model_name = m_name + '_nclass_' + str(c)
        fcm, pred_mem = fcm_compute(in_data, c, model_name, save=True)
        fcms.append(fcm)
        pred_mems.append(pred_mem)
        y_pred = get_best_fuzz(pred_mem)

        if(label is not None):
            _, _, _, acc = cluster_iden(label, y_pred)
            print(model_name, " accucacy =", acc)

        save_dir = 'FCM_results/' + model_name + '_plot.png'
        show_new_class(y_pred, x, z, save_dir=save_dir)

    return fcms, pred_mems
