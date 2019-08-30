"""
Author: Nitchakul Pipitvej
GitHub: wafflescore
"""

from minisom import MiniSom, asymptotic_decay

import numpy as np
import matplotlib.pyplot as plt
import itertools

from skimage import measure
from skimage.segmentation import random_walker
from skimage import filters
from scipy.spatial import distance
from collections import Counter

from timeit import default_timer as timer
import random
from acse_9_irp_wafflescore import MiscHelpers as mh

import logging
import sys

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)

def compute_dim(num_sample):
    """
    Compute a default dimension of the SOMs.

    This function returns the dimension size of the SOMs.
    The size returned is sqrt(5 * sqrt(num_sample)), with the exception
    that the minimum dimension size = 10

    Parameters
    ----------
    num_sample : int
        Total number of data points that will populate the SOMs

    Returns
    -------
    int
        Ideal dimension.
    """

    dim = 5 * np.sqrt(num_sample)
    dim = np.int(np.sqrt(dim))
    if dim < 10:
        return 10
    else:
        return dim


def som_assemble(in_data, seed, dim, lr=0.5, sigma=2.5):
    """Initialize the SOMs model for training

    Parameters
    ----------
    in_data : np.array or list
        data matrix
    seed : integer
        random seed for reproducibility
    dim : int
        dimension of the SOMs distance matrix
    lr : float, optional
        learning rate, by default 0.5
    sigma : float, optional
        spread of the neighborhood function, by default 2.5

    Returns
    -------
    MiniSom
        an object of Minisom class, see minisom.py for further details
    """

    # Initialization som and weights
    num_features = np.shape(in_data)[1]
    som = MiniSom(dim, dim, num_features, sigma=sigma, learning_rate=lr,
                  neighborhood_function='gaussian', random_seed=seed)

    som.pca_weights_init(in_data)

    return som


def plot_som(som, in_data, label, save=False, save_name='temp'):
    """plots the distance map / u-matrix of the SOMs along with the label

    Parameters
    ----------
    som : MiniSom
        trained Minisom object
    in_data : np.array or list
        data matrix
    label : np.array or list
        the true label of each data point
    save : bool, optional
        flag, by default False
    save_name : str, optional
        the name which will be used to save the plot as png file,
        by default 'temp'
    """
    plt.figure(figsize=(9, 7))
    # Plotting the response for each litho-class
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    # plotting the distance map as background
    plt.colorbar()

    for t, xx in zip(label, in_data):
        w = som.winner(xx)  # getting the winner
        # palce a marker on the winning position for the sample xx
        plt.text(w[0]+.5, w[1]+.5, str(t),
                 color=plt.cm.rainbow(t/10.))

    plt.axis([0, som.get_weights().shape[0], 0, som.get_weights().shape[1]])
    if(save):
        save_dir = 'SOMs_results/' + save_name + '_plot.png'
        plt.savefig(save_dir)
        print('Plot saved at:', save_dir)
    plt.show()


def save_som_report(som, save_name, it, et, report=None):

    param_vals = str(save_name) + '\n---' + \
             '\niterations,' + str(it) + \
             '\nelapsed time,' + str(et) + '\n\n'

    # save report to file
    fdir = save_name + '_report.csv'
    print('Report saved at', fdir)
    mode = 'w'
    f1 = open(fdir, mode)
    f1.write(param_vals)

    if(report):
        f1.write(str(report))
    f1.write('\n\n--------------------\n\n')
    f1.close()
    print('Report saved at:', fdir)


def histedges_equalN(in_data, nbin=10):
    """generates a histogram where each bin will contain the same number of
       data points

    Parameters
    ----------
    in_data : np.array or list
        data array
    nbin : int
        number of bins to populate, by default 10

    Returns
    -------
    np.array
        numpy array of all the histogram bins
    """
    ttl_dtp = len(in_data)
    return np.interp(np.linspace(0, ttl_dtp, nbin + 1),
                     np.arange(ttl_dtp),
                     np.sort(in_data))


def plot_u_matrix(som_u_mat):
    """Plots the distance map / u-matrix of the SOMs

    Parameters
    ----------
    som : MiniSom
        trained Minisom object

    Returns
    -------
    np.array
        numpy array of all the histogram bins
    """

    f_image = som_u_mat.flatten()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.show()
    ax1.pcolor(som_u_mat, cmap='bone_r')
    hist = plt.hist(f_image, histedges_equalN(f_image, 10), density=True)

    return hist[1]


def gen_e_model(n_map, som_label):
    """generates the Earth model from neuron map"""
    som_class = []
    for i in range(len(n_map)):
        som_class.append(som_label[n_map[i][0]][n_map[i][1]])

    return np.array(som_class)


def closest_n(value):
    """Assign cluster number to the mask's border indexes by using the
       closest neighbor's value

    Parameters
    ----------
    value : np.array
        numpy array of the cluster number, noted that the borders are marked
        with 0

    Returns
    -------
    np.array
        new label with all the border index populated
    """
    borders = np.array(np.where(value == 0)).T
    new_label = np.array(value)

    vals = np.where(value != 0)
    vals = np.array(vals).T

    for b in borders:
        # find index of the closest value
        c_idx = distance.cdist([b], vals).argmin()
        new_label[b[0], b[1]] = value[vals[c_idx, 0]][vals[c_idx, 1]]

    return new_label


def KNN(value, k=5, border_val=0):
    """Assign cluster number to the mask's border indexes by using the
    K-nearest neighbor method

    Parameters
    ----------
    value : np.array
        numpy array of the cluster number, noted that the borders are marked
        with 0
    k : int, optional
        number of neighbor to consider, by default 5

    Returns
    -------
    np.array
        new label with all the border index populated
    """
    borders = np.array(np.where(value == border_val)).T
    new_label = np.array(value)

    vals = np.where(value != 0)
    if(len(vals[0]) < 5):
        logging.info("Not enough labeled neighbor to perform KNN.\n\
                      Will return the original inputted value.")
        return value
    vals = np.array(vals).T

    for b in borders:
        # find index of the closest k neighbors
        dist = distance.cdist([b], vals)
        c_idx = np.argpartition(dist, k)
        c_idx = c_idx[0, :k]

        mins_idx = np.array(list(zip(vals[c_idx, 0], vals[c_idx, 1])))
        class_counter = Counter()
        for idx in mins_idx:
            class_counter[value[idx[0], idx[1]]] += 1
        cl = class_counter.most_common(1)[0][0]

        new_label[b[0], b[1]] = cl

    return new_label


def watershed_level(image, bins, border_width=0.1, plot=False, conn=None):
    num_bins = len(bins)
    """Computes and classify the SOM's u-matrix or total gradient using
    watershed classification method

    Parameters
    ----------
    image : np.array
        u-matrix or total gradient of the SOMs
    bins : np.array
        numpy array of all the histogram bins
    plot : bool, optional
        flag whether to plot the watershed level or not, by default False
    conn : int, optional
        connectivity flag for measure.label, by default None

    Returns
    -------
    np.array
        numpy array of predicted cluster labels from each watershed level
    """
    ncols = 6
    if(plot):
        fig, axes = plt.subplots(ncols=ncols, nrows=num_bins,
                                 figsize=(12, num_bins*3),
                                 sharex=True, sharey=True)

        ax = axes.ravel()
    ws_labels = np.zeros((num_bins * ncols, image.shape[0], image.shape[1]))

    for i in range(num_bins):
        val = filters.threshold_local(image, block_size=3 + 2*i)
        block_mask = (image < val)
        markers = measure.label(block_mask, connectivity=conn)
        ws_labels[i*ncols] = closest_n(markers) - 1
        ws_labels[i*ncols + 1] = KNN(markers) - 1
        ws_labels[i*ncols + 2] = random_walker(image, markers)
        if(plot):
            ax[i*ncols].imshow(ws_labels[i*ncols + 0], origin='lower')
            ax[i*ncols].title.set_text('b_cn: it={} n_class={}'.format(i,
                                       len(np.unique(ws_labels[i*ncols + 0]))))
            ax[i*ncols + 1].imshow(ws_labels[i*ncols + 1], origin='lower')
            ax[i*ncols + 1].title.set_text('b_knn: it={} n_class={}'.format(i,
                                           len(np.unique(ws_labels[i*ncols + 1]))))
            ax[i*ncols + 2].imshow(ws_labels[i*ncols + 2], origin='lower')
            ax[i*ncols + 2].title.set_text('b_rw: it={} n_class={}'.format(i,
                                           len(np.unique(ws_labels[i*ncols + 2]))))

        thres_mask = (image <= bins[i])
        markers = measure.label(thres_mask, connectivity=conn)
        ws_labels[i*ncols + 3] = closest_n(markers) - 1
        ws_labels[i*ncols + 4] = KNN(markers) - 1
        ws_labels[i*ncols + 5] = random_walker(image, markers)
        if(plot):
            ax[i*ncols + 3].imshow(ws_labels[i*ncols + 3], origin='lower')
            ax[i*ncols + 3].title.set_text('b_cn: it={} n_class={}'.format(i,
                                           len(np.unique(ws_labels[i*ncols + 3]))))
            ax[i*ncols + 4].imshow(ws_labels[i*ncols + 4], origin='lower')
            ax[i*ncols + 4].title.set_text('b_knn: it={} n_class={}'.format(i,
                                           len(np.unique(ws_labels[i*ncols + 4]))))
            ax[i*ncols + 5].imshow(ws_labels[i*ncols + 5], origin='lower')
            ax[i*ncols + 5].title.set_text('b_rw: it={} n_class={}'.format(i,
                                           len(np.unique(ws_labels[i*ncols + 5]))))

    return ws_labels


def eval_ws(in_data, ws_labels, n_map, label=None, re_all=False):
    """Evaluate and return the best watershed prediction result

    Parameters
    ----------
    in_data : np.array or list
        data matrix
    ws_labels : np.array
        predicted cluster labels from watershed segmentation
    n_map : np.array
        array of the winner neuron
    label : np.array or list, optional
        the true label of each data point

    Returns
    -------
    np.array
        list of best watershed labels, may contain more than one set
    """
    len_watershed = ws_labels.shape[0]
    cluster_labels = np.zeros((len_watershed, len(in_data)))
    avg_sils = np.full(len_watershed, np.nan)
    ch_scs = np.full(len_watershed, np.nan)

    if(label is not None):
        avg_ents = np.full(len_watershed, np.nan)
        avg_purs = np.full(len_watershed, np.nan)

    for i in range(len_watershed):
        param = {'watershed idx': i}
        if(len(np.unique(ws_labels[i])) > 1):
            cluster_labels[i] = gen_e_model(n_map, ws_labels[i])
            avg_sils[i] = mh.int_eval_silhouette(in_data, cluster_labels[i],
                                              method='som_watershed',
                                              param=param)
            try:
                ch_scs[i] = mh.cal_har_sc(in_data, cluster_labels[i])
            except:
                ch_scs[i] = -1
            if(label is not None):
                avg_ents[i], avg_purs[i] = mh.ext_eval_entropy(label,
                                                            cluster_labels[i])
    best_idx = []
    best_idx.append(np.nanargmax(np.array(avg_sils)))       # closest to 1
    best_idx.append(np.nanargmax(ch_scs))                   # higher = better
    if(label is not None):
        best_idx.append(np.nanargmin(np.array(avg_ents)))   # closest to 0
        best_idx.append(np.nanargmax(np.array(avg_purs)))   # closest to 1
    best_idx = np.unique(best_idx)
    if(re_all):
        return (cluster_labels, avg_sils,
                ch_scs, best_idx)
    else:
        return (cluster_labels[best_idx], avg_sils[best_idx],
                ch_scs[best_idx])


def run_SOMs(in_data, dim, iter_cnt, lr, sigma, seed=10):
    """Method to fully run SOMs
    
    Parameters
    ----------
    in_data : np.array or list
        data matrix
    dim : int
        dimension of the SOMs distance matrix
    iter_cnt : integer
        number of iterations for SOMs to perform
    lr : float
        learning rate
    sigma : float
        spread of the neighborhood function, by default 2.5dim : int
    seed : integer, optional
        random seed for reproducibility, by default 10
    
    Returns
    -------
    minisom
        minisom object
    np.array
        cluster label
    """

    som = som_assemble(in_data, seed, dim, lr, sigma)
    som.train_random(in_data, iter_cnt, verbose=False)
    u_matrix = som.distance_map().T
    watershed_bins = histedges_equalN(u_matrix.flatten())
    ws_labels = watershed_level(u_matrix, watershed_bins)
    n_map = som.neuron_map(in_data)
    
    cluster_labels, _, _ = eval_ws(in_data, ws_labels, n_map)
    return som, cluster_labels


def gen_param_grid(init_guess):
    g_dim, g_it, g_lr, g_sigma = init_guess
    min_dim = g_dim - 10 if g_dim - 5 > 10 else 10
    max_dim = g_dim + 10 if g_dim + 10 > 10 else 20
    param_grid = {
        'dim': list(range(min_dim, max_dim+1)),
        'iter_cnt': list(range(g_it - 500, g_it + 500, 200)),
        'learning_rate': list(np.logspace(np.log10(0.25), np.log10(0.75),
                                          base=10, num=100)),
        'sigma': list(np.linspace(g_sigma-1, g_sigma+1, num=30)),
    }
    return param_grid


def random_search_som(in_data, init_guess, max_eval=20, label=None, seed=10,
                      re_all=False):
    """perform random search for SOMs best parameters.
    
    Parameters
    ----------
    in_data : np.array or list
        data matrix
    init_guess : tuple
        list of initial guess of the parameters, in order of dimension,
        number of iterations, learning rate, and sigma
    max_eval : int, optional
        number of max iterartion to perform the search, by default 20
    label : np.array or list, optional
        the true label of each data point, by default None
    seed : integer, optional
        random seed for reproducibility, by default 10
    
    Returns
    -------
    All cluster label and its counterpart parameters.
    """
    random.seed(seed)
    
    param_grid = gen_param_grid(init_guess)
    
    dims = np.zeros(max_eval)
    iters = np.zeros(max_eval)
    lrs = np.zeros(max_eval)
    sigmas = np.zeros(max_eval)

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

        dims[i], iters[i], lrs[i], sigmas[i] = list(random_params.values())

        som = som_assemble(in_data, seed, int(dims[i]), lr=lrs[i], sigma=sigmas[i])
        som.train_random(in_data, int(iters[i]), verbose=False)
        u_matrix = som.distance_map().T
        watershed_bins = histedges_equalN(u_matrix.flatten())
        ws_labels = watershed_level(u_matrix, watershed_bins)
        n_map = som.neuron_map(in_data)

        _c, _as, _ch = eval_ws(in_data, ws_labels, n_map)
        cluster_labels[i], avg_sils[i], ch_scs[i] = _c[0], _as[0], _ch[0]

        n_clusters = len(np.unique(cluster_labels[i]))
        if(n_clusters < 5 or n_clusters > 30):
            logging.info("Random search using dim=%d, iter=%d, lr=%.6f, sigma=%.6f\
                 result to very small / large number of clusters (n_clusters = %d)\
                 " % (dims[i], iters[i], lrs[i], sigmas[i], n_clusters))
            continue
        
        logging.info("dim=%d, iter=%d, lr=%.6f, sigma=%.6f, sil=%.6f, ch=%.6f" % (dims[i], iters[i], lrs[i], sigmas[i], avg_sils[i], ch_scs[i]))
        
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
    if(re_all):
        return (cluster_labels, avg_sils,
                ch_scs, dims, iters, lrs, sigmas, best_idx)
    else:
        return (cluster_labels[best_idx], avg_sils[best_idx],
                ch_scs[best_idx], dims[best_idx], iters[best_idx], 
                lrs[best_idx], sigmas[best_idx])

