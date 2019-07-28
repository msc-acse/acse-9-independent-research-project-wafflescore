#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 02:40:41 2019

@author: laveniier
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def load_model(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


def show_new_class(pred_class, x, z, title="", figsize=(12, 6),
                   cmap='viridis_r', save_dir="", sep_label=False,
                   classes=None):
    if(classes is None):
        classes = range(np.unique(pred_class)[-1])
        n_class = len(np.unique(pred_class))
    else:
        n_class = classes[-1]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if(sep_label):
        sctr = ax.scatter(x=x, y=z, c=pred_class,
                          cmap=plt.cm.get_cmap(cmap, n_class))
        plt.colorbar(sctr, ticks=classes,
                     label='class', ax=ax, format='%d')
    else:
        sctr = ax.scatter(x=x, y=z, c=pred_class,
                          cmap=cmap)
        plt.colorbar(sctr, ax=ax)

    ax.invert_yaxis()
    plt.title(title)

    if(save_dir):
        fig.savefig(save_dir)
        print('Plot saved at:', save_dir)


def cluster_iden(label, pred, x=None, z=None, plot=False):
    cm = confusion_matrix(label, pred)
    class_map = cm.argmax(axis=1)

    comb_label = np.zeros_like(label)
    for i in range(len(label)):
        comb_label[i] = class_map[label[i]]

    comb_cm = confusion_matrix(comb_label, pred)
    comb_acc = accuracy_score(comb_label, pred)
    print(class_map)
    if(plot):
        show_new_class(comb_label, x, z)
        show_new_class(pred, x, z)
    return class_map, comb_cm, comb_label, comb_acc


def plot_fields(in_data, X, Z, titles, model=""):
    for i in range(in_data.shape[1]):
        val = in_data[:, i]
        save_dir = ""
        if(model):
            save_dir = 'data/' + model + titles[i+1] + '.png'
        show_new_class(val, X, Z, title=titles[i+1], figsize=(6, 3),
                       save_dir=save_dir)


def plot_cm(cm):
    df_cm = pd.DataFrame(cm, range(cm.shape[0]), range(cm.shape[1]))
    plt.figure(figsize=(20, 10))
#    sn.set(font_scale=1.4)  #for label size
    sn.heatmap(df_cm, annot=True, fmt='g', cmap='jet')  # font size


def save_model(model, fdir):
    with open(fdir, 'wb') as outfile:
        pickle.dump(model, outfile)

    print('Model saved at:', fdir)
