"""
Author: Nitchakul Pipitvej
GitHub: wafflescore
"""

from pickle import dump, load
import matplotlib.pyplot as plt
import numpy as np
from seaborn import heatmap
from pandas import DataFrame
from sklearn.cluster import KMeans

import matplotlib.cm as cm

from sklearn.metrics import (confusion_matrix, accuracy_score,
                             silhouette_samples, silhouette_score)
from sklearn.metrics import calinski_harabaz_score as cal_har_sc

from scipy.stats import entropy

import logging
import sys

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)


def load_model(file_path):
    """Function to load trained ML model

    Parameters
    ----------
    file_path : string
        File path to the ML model

    Returns
    -------
    object
        Depends on the model loaded
    """
    loaded_model = load(open(file_path, 'rb'))
    return loaded_model


def plot_e_model(value, x, z, title="", figsize=(12, 6),
                 cmap='viridis', save_path="", sep_label=False):
    """Function to plot the Earth model

    Parameters
    ----------
    value : np.array or list
        value to plot
    x : np.array or list
        x coordinate
    z : np.array or list
        z coordinate, depth
    title : str, optional
        title of the plot, by default ""
    figsize : tuple, optional
        size of the displayed figure, by default (12, 6)
    cmap : str, optional
        color map name used for plotting, by default 'viridis'
    save_path : str, optional
        plot's save directory, by default ""
    sep_label : bool, optional
        flag to separate the label on the color bar, by default False
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if(sep_label):
        classes = np.unique(value)
        n_class = len(classes)
        sctr = ax.scatter(x=x, y=z, c=value,
                          cmap=plt.cm.get_cmap(cmap, n_class))
        plt.colorbar(sctr, ticks=classes,
                     label='class', ax=ax, format='%d')
    else:
        sctr = ax.scatter(x=x, y=z, c=value, cmap=cmap)
        plt.colorbar(sctr, ax=ax)

    ax.invert_yaxis()   # invert axis since z is by depth
    plt.title(title)
    plt.show()

    if(save_path):
        fig.savefig(save_path)
        print('Plot saved at:', save_path)


def plot_fields(in_data, x, z, titles, save_name=""):
    """Plot the field of the input vector

    Parameters
    ----------
    in_data : np.array or list
        data matrix
    x : np.array or list
        x coordinate
    z : np.array or list
        z coordinate
    titles : string array
        the title of each column in the data matrix
    save_name : str, optional
        the name which will be used to save the plot as png file, by default ""
        which will not save the plot
    """
    for i in range(in_data.shape[1]):
        val = in_data[:, i]
        save_path = ""
        if(save_name):
            save_path = save_name + titles[i] + '.png'
        plot_e_model(val, x, z, title=titles[i], figsize=(6, 3),
                     save_path=save_path)


def plot_cm(cm):
    """Plotting the confusion matrix"""
    df_cm = DataFrame(cm, range(cm.shape[0]), range(cm.shape[1]))
    plt.figure(figsize=(20, 10))
    ax = plt.subplot()
    heatmap(df_cm, annot=True, fmt='g', cmap='jet', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')


def save_model(model, save_path):
    """Saves model with pickle

    Parameters
    ----------
    model : object
        trained ML model
    save_path : str
        save directory
    """
    with open(save_path, 'wb') as outfile:
        dump(model, outfile)

    print('Model saved at:', save_path)


def search_list(in_list, item):
    """Search for property in the input data

    Parameters
    ----------
    in_list : list
        list of all the available properties
    item : str
        name of the property to search

    Returns
    -------
    list
        list of index of the searched property
    """
    return [i for i, j in enumerate(in_list) if j == item]


def crossplots(in_data, label, col_name):
    """Plotting some key features' crossplot

    Parameters
    ----------
    in_data : np.array or list
        data matrix
    label : list
        class label
    col_name : list of string
        list of the properties presented in the input data
        eg. ['vp', 'vs', 'dn', 'vp/vs', 'qp', 'qs', 'x', 'z']
    """
    titles = ['Vp (m/s)', 'Vs (m/s)', 'Density (kg/m$^3$)', 'Vp/Vs',
              'Qp', 'Qs', 'X', 'Z']

    # find index of each properties
    vp_idx = search_list(col_name, 'vp')
    vs_idx = search_list(col_name, 'vs')
    dn_idx = search_list(col_name, 'dn')
    vpvs_idx = search_list(col_name, 'vp/vs')
    qp_idx = search_list(col_name, 'qp')
    qs_idx = search_list(col_name, 'qs')

    fig, ax = plt.subplots(1, 5, figsize=(30, 5))
    # Vp against Vs
    if(vp_idx and vs_idx):
        c_plot(in_data[:, vp_idx], in_data[:, vs_idx], titles[0], titles[1],
               ax[0], label=label)
    # Vp against density
    if(vp_idx and dn_idx):
        c_plot(in_data[:, vp_idx], in_data[:, dn_idx], titles[0], titles[2],
               ax[1], label=label)
    # Vp against Vp/Vs
    if(vp_idx and vpvs_idx):
        c_plot(in_data[:, vp_idx], in_data[:, vpvs_idx], titles[0], titles[3],
               ax[2], label=label)
    # Vp against Qp
    if(vp_idx and qp_idx):
        c_plot(in_data[:, vp_idx], in_data[:, qp_idx], titles[0], titles[4],
               ax[3], label=label)
    # Vs against Qs
    if(vs_idx and qs_idx):
        c_plot(in_data[:, vs_idx], in_data[:, qs_idx], titles[1], titles[5],
               ax[4], label=label)


def c_plot(x, y, x_label, y_label, ax, alp=0.25, label=None):
    """Cross plot of feature x and y"""
    ax.scatter(x, y, alpha=alp, c=label, s=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def elbowplot(n_class, sse):
    """Elbow plot"""
    fig, ax = plt.subplots(1, 1)
    plt.plot(n_class, sse, 'o-')


def ext_eval_entropy(label, pred, save_path="", init_clus=0):
    """Measures the cluster validity using Entropy and Purity.
    The result will be saved as a csv file.

    Parameters
    ----------
    label : np.array or list
        the true label of each data point
    pred : np.array or list
        the predicted cluster label
    save_path : str, optional
        save directory of the entropy/purity csv file, by default ""
        which will not save the result to file
    init_clus : int, optional
        index of the initial cluster, by default 0
        this can be overwritten since some clustering method, eg DBSCAN,
        contains noise cluster which often depicted as cluter -1

    Returns
    -------
    float
        average entropy value
    float
        average purity value
    """

    cm = confusion_matrix(pred, label)
    idx_y = np.where(np.sum(cm, axis=0) != 0)[0]
    cm = cm[:, idx_y]
    idx_x = np.where(np.sum(cm, axis=1) != 0)[0]
    cm = cm[idx_x, :]

    col_sum = np.sum(cm, axis=1)    # number of data points in each cluster
    ttl_dtp = len(pred)             # total number of data points

    ent = entropy(cm.T)
    purity = np.max(cm, axis=1) / col_sum

    n_class = len(np.unique(label))
    header = ','
    for i in range(n_class):
        header += 'Label = ' + str(i+1) + ', '
    header += 'Entropy, Precision'

    row_head = ['Cluster = '] * (cm.shape[0] + 1)
    for i in range(cm.shape[0]):
        row_head[i] += str(i + init_clus)
    row_head[-1] = 'Total'
    row_head = np.array(row_head).reshape(cm.shape[0]+1, 1)
    report = np.column_stack((cm, ent))
    report = np.column_stack((report, purity))

    report = np.vstack((report, np.sum(report, axis=0)))
    report = np.hstack((row_head, report))

    report[-1, -2] = np.sum((ent * col_sum) / float(ttl_dtp))
    report[-1, -1] = np.sum((purity * col_sum) / float(ttl_dtp))

    if(save_path):
        np.set_printoptions(precision=4, suppress=True)

        np.savetxt(save_path, report, delimiter=',', fmt='%s', header=header)
        print("Entropy and precision file saved at: ", save_path)

    # returns average entropy and purity
    return report[-1, -2], report[-1, -1]


def int_eval_silhouette(in_data, pred, method="", param=None, plot=False):
    """Measures the cluster validity using silhouette method

    Parameters
    ----------
    in_data : np.array or list
        data matrix
    pred : np.array or list
        the predicted cluster label
    method : str, optional
        name of the method used, by default ""
    param : dict, optional
        dict of the parameters and values used for the model, by default None
    plot : bool, optional
        flag whether to plot the silhouette plot or not, by default False

    Returns
    -------
    float
        the average silhouette score, ranges between -1 to 1.
        ideally, the best score is 1
    """
    n_clusters = len(np.unique(pred))
    if(n_clusters <= 1):
        print("The number of cluster is too low, n_cluster =", n_clusters, "\n"
              "Return an average score of -1 by default.")
        return -1
    silhouette_avg = silhouette_score(in_data, pred)

    if(plot):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.set_xlim([-1, 1])

        hyperpara = ''
        for k in param.keys():
            hyperpara += k + "=" + str(param[k]) + ', '

        # adds space between each cluster's plot
        ax.set_ylim([0, len(pred) + (n_clusters + 1) * 10])

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(in_data, pred)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[pred == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_title("The silhouette plot for the various clusters.")
        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster prediction label")

        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.suptitle(("Silhouette analysis for %s method with %s" %
                      (method, hyperpara[:-2])),
                     fontsize=14, fontweight='bold')

        plt.show()

    return silhouette_avg


def iter_kmean(in_data, in_range, label=None):
    """Iterates number of cluster in Kmeans to plot the elbow

    Parameters
    ----------
    in_data : np.array or list
        data matrix
    in_range : class 'range'
        the range of number of clusters to iterate through
    label : np.array or list
        the true label of each data point

    Returns
    -------
    list
        list of prediction members
    list
        sum square error result
    """
    it = len(in_range)

    SSE = np.zeros(it)
    avg_sils = np.full(it, np.nan)
    ch_scs = np.full(it, np.nan)
    cluster_labels = np.zeros((it, len(in_data)))
    if(label is not None):
        avg_ents = np.full(it, np.nan)
        avg_purs = np.full(it, np.nan)

    for i, n_clusters in enumerate(in_range):
        kmeans = KMeans(n_clusters=n_clusters, max_iter=1000).fit(in_data)
        cluster_labels[i] = kmeans.labels_
        
        SSE[i] = kmeans.inertia_
        avg_sils[i] = int_eval_silhouette(in_data, cluster_labels[i])
        ch_scs[i] = cal_har_sc(in_data, cluster_labels[i])
        logging.info("sil=%.6f, chs=%.6f" % (avg_sils[i], ch_scs[i]))

        if(label is not None):
            avg_ents[i], avg_purs[i] = ext_eval_entropy(label, cluster_labels[i], init_clus=-1) 
            logging.info("ent=%.6f, pur=%.6f" % (avg_ents[i], avg_purs[i]))
        
    best_idx = []
    best_idx.append(np.nanargmax(np.array(avg_sils)))       # closest to 1
    best_idx.append(np.nanargmax(ch_scs))                   # higher = better
    if(label is not None):
        best_idx.append(np.nanargmin(np.array(avg_ents)))   # closest to 0
        best_idx.append(np.nanargmax(np.array(avg_purs)))   # closest to 1
    best_idx = np.unique(best_idx)
    
    elbowplot(in_range, SSE)
    
    return (cluster_labels, avg_sils, ch_scs, SSE, best_idx)
