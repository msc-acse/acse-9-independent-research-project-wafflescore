from minisom import MiniSom, asymptotic_decay

#import importlib.util
#spec = importlib.util.spec_from_file_location("minisom", "./minisom-master/minisom.py")
#minisom_module = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(minisom_module)

import numpy as np
import matplotlib.pyplot as plt
import itertools

from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import measure
from skimage.segmentation import random_walker
from scipy import ndimage

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import calinski_harabaz_score as cal_har_sc

from skimage import filters
from skimage import measure
from scipy.spatial import distance
from collections import Counter

from timeit import default_timer as timer
import random
from MiscHelpers import save_model, ext_eval_entropy, int_eval_silhouette


def compute_dim(num_sample):
    """
    Compute ideal dimension of the SOMs.

    This function returns the ideal dimension size of the SOMs.
    The ideal size is sqrt(5 * sqrt(num_sample)), with the exception
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
    in_data : np.array
        [description]
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
    minisom.MiniSom
        an object of Minisom class, see minisom.py for further details
    """

    # Initialization som and weights
    num_features = np.shape(in_data)[1]
    som = MiniSom(dim, dim, num_features, sigma=sigma, learning_rate=lr,
                  neighborhood_function='gaussian', random_seed=seed)

    som.pca_weights_init(in_data)

    return som


def plot_som(som, norm_data, label, save=False, model_name='temp'):
    plt.figure(figsize=(9, 7))
    # Plotting the response for each litho-class
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    # plotting the distance map as background
    plt.colorbar()

    for t, xx in zip(label, norm_data):
        w = som.winner(xx)  # getting the winner
        # palce a marker on the winning position for the sample xx
        plt.text(w[0]+.5, w[1]+.5, str(t),
                 color=plt.cm.rainbow(t/10.))

    plt.axis([0, som.get_weights().shape[0], 0, som.get_weights().shape[1]])
    if(save):
        save_dir = 'SOMs_results/' + model_name + '_plot.png'
        plt.savefig(save_dir)
        print('Plot saved at:', save_dir)
    plt.show()


def save_som_report(som, model_name, it, et, report=None,
                    type=''):

    param_vals = str(model_name) + '\n---' + \
             '\niterations,' + str(it) + \
             '\nelapsed time,' + str(et) + '\n\n'

    # save report to file
    fdir, mode = '', ''
    if(type == 'grid'):
        fdir = model_name + '_gridsearch_report.csv'
        mode = 'a+'
    elif(type == 'rand'):
        fdir = model_name + '_randomsearch_report.csv'
        mode = 'a+'
    else:
        fdir = model_name + type + '_report.csv'
        mode = 'w'
    f1 = open(fdir, mode)
    f1.write(param_vals)

    if(type == 'grid' or type == 'rand'):
        f1.write(str(report))
    f1.write('\n\n--------------------\n\n')
    f1.close()
    print('Report saved at:', fdir)


def som_lloss(y_pred, y_actual, n_class):
    y_pred_agg = []

    for num in list(y_actual):
        t = np.array(np.zeros(n_class))
        t[num-1] = 1
        y_pred_agg.append(list(t))

    lloss = log_loss(y_actual, y_pred_agg, eps=1e-15)

    return lloss


def gen_report(som, norm_data, label, n_class, gen_train=False,
               log_print=False, seed=10):

    X_train, X_test, y_train, y_test = train_test_split(norm_data, label,
                                                        random_state=seed)
    som.labels_map(X_train, y_train)

    y_pred_test = np.array(som.classify(X_test))
    test_report = classification_report(y_test, y_pred_test, digits=4)
    test_c_matrix = confusion_matrix(y_test, y_pred_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    lloss_test = som_lloss(y_pred_test, y_test, n_class)

    full_report = "Test portion" + \
                  "\n" + str(test_report) + \
                  "\n" + str(test_c_matrix) + \
                  "\nLogloss=" + str(lloss_test) + \
                  "\nOverall probability=" + str(np.exp(-1 * lloss_test)) + \
                  "\nAccuracy=" + str(test_acc)

    if(gen_train):
        y_pred_train = np.array(som.classify(X_train))
        train_report = classification_report(y_train, y_pred_train, digits=4)
        train_c_matrix = confusion_matrix(y_train, y_pred_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        lloss_train = som_lloss(y_pred_train, y_train, n_class)

        full_report += "\nTrain portion" + \
                       "\n" + str(train_report) + \
                       "\n" + str(train_c_matrix) + \
                       "\nLogloss=" + str(lloss_train) + \
                       "\nOverall probability=" + str(np.exp(-1 * lloss_train)) + \
                       "\nAccuracy=" + str(train_acc)

    if(log_print):
        print(full_report)

    return full_report


def histedges_equalN(data, nbin=10):
    """generates a histogram where each bin will contain the same number of
       data points

    Parameters
    ----------
    data : np.array or list
        data array
    nbin : int
        number of bins to populate, by default 10

    Returns
    -------
    np.array
        
    """
    npt = len(data)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(data))


def plot_u_matrix(som_u_mat):
    """Plots the distance map / u-matrix of the SOMs

    Parameters
    ----------
    som : numpy.ndarray
        the distance map / u-matrix

    Returns
    -------
    numpy.ndarray
        numpy array of all the histogram bins
    """

    f_image = som_u_mat.flatten()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.show()
    ax1.pcolor(som_u_mat, cmap='bone_r')
    hist = plt.hist(f_image, histedges_equalN(f_image, 10), density=True)

    return hist[1]


def neuron_map(som, norm_data):
    n_map = []
    for xx in norm_data:
        n_map.append(som.winner(xx))

    return np.array(n_map)


def gen_e_model(n_map, som_label):
    som_class = []
    for i in range(len(n_map)):
        som_class.append(som_label[n_map[i][0]][n_map[i][1]])

    return np.array(som_class)


def plot_class_som(som, norm_data, som_label):
    som_class = []
    for xx in norm_data:
        wx, wy = som.winner(xx)   # getting the winner neuron location
        som_class.append(som_label[wx][wy])   # get som's class label

    return som_class


def closest_n(label):
    """Assign cluster number to the border indexes by using the 
       closest neighbor's label

    Parameters
    ----------
    label : numpy.ndarray
        numpy array of the cluster label, noted that the borders are labeled
        with 0

    Returns
    -------
    numpy.ndarray
        new label with all the border index populated
    """
    borders = np.array(np.where(label == 0)).T
    new_label = np.array(label)

    vals = np.where(label != 0)
    vals = np.array(vals).T

    for b in borders:
        distance.cdist([b], vals)
        c_idx = distance.cdist([b], vals).argmin()
        cl = label[vals[c_idx, 0]][vals[c_idx, 1]]

        new_label[b[0], b[1]] = cl

    return new_label


def watershed_level(image, bins, border_width=0.1, plot=False, conn=None):
    num_bins = len(bins)
    """Computes and classify the SOM's u-matrix or total gradient using 
    watershed classification method

    Parameters
    ----------
    image : numpy.ndarray
        u-matrix or total gradient of the SOMs
    bins : numpy.ndarray
        numpy array of the 
    plot : bool, optional
        [description], by default False
    conn : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    ncols = 6
    if(plot):
        fig, axes = plt.subplots(ncols=ncols, nrows=num_bins,
                                 figsize=(12, num_bins*3),
                                 sharex=True, sharey=True)

        ax = axes.ravel()
    all_label = np.zeros((num_bins * ncols, image.shape[0], image.shape[1]))

    for i in range(num_bins):
        val = filters.threshold_local(image, block_size=3 + 2*i)
        block_mask = (image < val)
        markers = measure.label(block_mask, connectivity=conn)
        all_label[i*ncols] = closest_n(markers) - 1
        all_label[i*ncols + 1] = KNN(markers) - 1
        all_label[i*ncols + 2] = random_walker(image, markers)
        if(plot):
            ax[i*ncols].imshow(all_label[i*ncols + 0], origin='lower')
            ax[i*ncols].title.set_text('b_cn: it={} n_class={}'.format(i,
                                       len(np.unique(all_label[i*ncols + 0]))))
            ax[i*ncols + 1].imshow(all_label[i*ncols + 1], origin='lower')
            ax[i*ncols + 1].title.set_text('b_knn: it={} n_class={}'.format(i,
                                           len(np.unique(all_label[i*ncols + 1]))))
            ax[i*ncols + 2].imshow(all_label[i*ncols + 2], origin='lower')
            ax[i*ncols + 2].title.set_text('b_rw: it={} n_class={}'.format(i,
                                           len(np.unique(all_label[i*ncols + 2]))))

        thres_mask = (image <= bins[i])
        markers = measure.label(thres_mask, connectivity=conn)
        all_label[i*ncols + 3] = closest_n(markers) - 1
        try:
            all_label[i*ncols + 4] = KNN(markers) - 1
        except:
            print('error on knn')
            return markers
        all_label[i*ncols + 5] = random_walker(image, markers)
        if(plot):
            ax[i*ncols + 3].imshow(all_label[i*ncols + 3], origin='lower')
            ax[i*ncols + 3].title.set_text('b_cn: it={} n_class={}'.format(i,
                                           len(np.unique(all_label[i*ncols + 3]))))
            ax[i*ncols + 4].imshow(all_label[i*ncols + 4], origin='lower')
            ax[i*ncols + 4].title.set_text('b_knn: it={} n_class={}'.format(i,
                                           len(np.unique(all_label[i*ncols + 4]))))
            ax[i*ncols + 5].imshow(all_label[i*ncols + 5], origin='lower')
            ax[i*ncols + 5].title.set_text('b_rw: it={} n_class={}'.format(i,
                                           len(np.unique(all_label[i*ncols + 5]))))

    return all_label


def KNN(value, k=5):
    borders = np.array(np.where(value == 0)).T
    new_label = np.array(value)

    vals = np.where(value != 0)
    if(len(vals[0]) < 5):
        print("Not enough labeled neighbor to perform KNN.")
        return value
    vals = np.array(vals).T

    for b in borders:
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


def eval_ws(in_data, ws_labels, n_map, actual_label=None):
    len_watershed = ws_labels.shape[0]
    cluster_labels = np.zeros((len_watershed, len(in_data)))
    avg_sils = np.full(len_watershed, np.nan)
    ch_scs = np.full(len_watershed, np.nan)

    if(actual_label is not None):
        avg_ents = np.full(len_watershed, np.nan)
        avg_purs = np.full(len_watershed, np.nan)

    for i in range(len_watershed):
        param = {'watershed idx': i}
        if(len(np.unique(ws_labels[i])) > 1):
            cluster_labels[i] = gen_e_model(n_map, ws_labels[i])
            avg_sils[i] = int_eval_silhouette(in_data, cluster_labels[i],
                                              method='som_watershed',
                                              param=param)
            try:
                ch_scs[i] = cal_har_sc(in_data, cluster_labels[i])
            except:
                ch_scs[i] = -1
            if(actual_label is not None):
                avg_ents[i], avg_purs[i] = ext_eval_entropy(actual_label,
                                                            cluster_labels[i])
    best_idx = []
    best_idx.append(np.nanargmax(np.array(avg_sils)))       # closest to 1
    best_idx.append(np.nanargmax(ch_scs))                   # higher = better
    if(actual_label is not None):
        best_idx.append(np.nanargmin(np.array(avg_ents)))   # closest to 0
        best_idx.append(np.nanargmax(np.array(avg_purs)))   # closest to 1

    return cluster_labels[np.unique(best_idx)]


def comp_ttl_grad(u_mat):
    return 0


# parameter tuning
# grid search
def grid_search_som(init_num, norm_data, label, m_name='', seed=10):

    dims = [-5, 0, 5, 10]
    dims += np.array(compute_dim(norm_data.shape[0]))
    idx = np.argwhere(dims <= 0)
    if(idx.size != 0):
        dims = dims[idx[-1][0]:]
    iter_cnts = [2000, 4000, 5000, 6000]
    lr = [0.25, 0.5, 0.75]
    sigma = [1, 2, 2.5, 3, 4]

    # small grid test
    # dims = [5]
    # iter_cnts = [10, 20]
    # lr = [0.75]
    # sigma = [1]

    test_num = init_num
    X_train, X_test, y_train, y_test = train_test_split(norm_data, label,
                                                        random_state=seed)

    hyperpara = [dims, iter_cnts, lr, sigma]
    hyperpara_perm = list(itertools.product(*hyperpara))

    best_acc = 0
    best_comb = []
    best_model = ''

    for i in range(init_num, len(hyperpara_perm)):
        comb = hyperpara_perm[i]
        print(comb[0], " ", comb[1], " ", comb[2], " ", comb[3])

        test_num += 1
        model_name = 'grid_search/g_' + m_name + '_' + str(test_num)

        som = som_assemble(X_train, seed, comb[0], lr=comb[2], sigma=comb[3])

        start = timer()
        som.train_random(X_train, comb[1], verbose=False)
        end = timer()
        elapsed_time = end - start

        som.labels_map(X_train, y_train)
        y_pred = np.array(som.classify(X_test))
        curr_acc = accuracy_score(y_test, y_pred)

        if(best_acc < curr_acc):
            best_acc = curr_acc
            best_comb = comb
            best_model = model_name
            print('current best model:', best_model)
            print('current best comb:', best_comb)
            print('current best acc:', best_acc)

        # save the current model
        # plot_som(som, norm_data, label, save=True, model_name=model_name)
        report = classification_report(y_test, y_pred, digits=4)
        c_matrix = confusion_matrix(y_test, y_pred)

        save_som_report(som, model_name, comb[1], elapsed_time, report=report,
                        type='grid')
    print('Best model found:', best_model)
    print('Best comb found:', best_comb)
    print('Best acc found:', best_acc)

    return best_comb


def gen_param_grid(grid_search_res):
    g_dim, g_it, g_lr, g_sigma = grid_search_res
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


# random search
def random_search_som(init_num, norm_data, label, grid_search_res,
                      seed=10, max_evals=10, m_name=''):

    param_grid = gen_param_grid(grid_search_res)
    test_num = init_num
    X_train, X_test, y_train, y_test = train_test_split(norm_data, label,
                                                        random_state=seed)

    best_acc = 0
    best_comb = []
    best_model = ''
    random.seed(seed)

    for i in range(max_evals):
        random_params = {k: random.sample(v, 1)[0]
                         for k, v in param_grid.items()}
        c_dim, c_it, c_lr, c_sig = list(random_params.values())

        print(c_dim, " ", c_it, " ", c_lr, " ", c_sig)

        test_num += 1
        model_name = 'random_search/r_' + m_name + '_' + str(test_num)

        som = som_assemble(norm_data, seed, c_dim, lr=c_lr, sigma=c_sig)

        start = timer()
        som.train_random(X_train, c_it, verbose=False)
        end = timer()
        elapsed_time = end - start

        som.labels_map(X_train, y_train)
        y_pred = np.array(som.classify(X_test))
        curr_acc = accuracy_score(y_test, y_pred)

        if(best_acc < curr_acc):
            best_acc = curr_acc
            best_comb = c_dim, c_it, c_lr, c_sig
            best_model = model_name
            print('current best model:', best_model)
            print('current best comb:', best_comb)
            print('current best acc:', best_acc)

        # save the current model
        # plot_som(som, norm_data, label, save=True, model_name=model_name)
        report = classification_report(y_test, y_pred, digits=4)
        c_matrix = confusion_matrix(y_test, y_pred)

        save_som_report(som, model_name, c_it, elapsed_time, report=report,
                        type='rand')

    print('Best model found:', best_model)
    print('Best comb found:', best_comb)
    print('Best acc found:', best_acc)

    return best_comb, best_acc
