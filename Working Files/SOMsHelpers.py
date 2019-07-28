from minisom_master.minisom import MiniSom, asymptotic_decay

#import importlib.util
#spec = importlib.util.spec_from_file_location("minisom", "./minisom-master/minisom.py")
#minisom_module = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(minisom_module)

import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

from skimage import filters
from skimage import measure
from scipy.spatial import distance
from collections import Counter

from timeit import default_timer as timer
import random
from MiscHelpers import save_model


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


def save_som(som, model_name, dim_x, dim_y, it, lr, sigma, et,
             report=None, c_matrix=None, grid=False, rand=False):

    param_vals = str(model_name) + '\n---' + \
             '\ndim,' + str(dim_x) + ' x ' + str(dim_y) + \
             '\niterations,' + str(it) + \
             '\nlearning rate,' + str(lr) + \
             '\nsigma,' + str(sigma) + \
             '\nelapsed time,' + str(et) + '\n\n'

#    save report to file
    fdir, mode = '', ''
    if(grid):
        fdir = 'SOMs_results/grid_search/gridsearch_report.csv'
        mode = 'a+'
    elif(rand):
        fdir = 'SOMs_results/random_search/randomsearch_report.csv'
        mode = 'a+'
    else:
        fdir = 'SOMs_results/' + model_name + '_report.csv'
        mode = 'w'
    f1 = open(fdir, mode)
    f1.write(param_vals)

    if(grid or rand):
        f1.write(report)
        f1.write('\nConfusion Matrix\n')
        np.savetxt(f1, c_matrix, fmt='%i', delimiter=",")
    f1.write('\n\n--------------------\n\n')
    f1.close()
    print('Report saved at:', fdir)

#    pickle the model
    fdir = 'SOMs_results/' + model_name + '_model.p'
    save_model(som, fdir)


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
                  "\nOverall probability=" + str(np.exp(-lloss_test)) + \
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
                       "\nOverall probability=" + str(np.exp(-lloss_train)) + \
                       "\nAccuracy=" + str(train_acc)

    if(log_print):
        print(full_report)

    return full_report


def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))


def plot_u_matrix(som, plot=False):

    image = som.distance_map().T

    if(plot):
        f_image = image.flatten()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig
        ax1.pcolor(som.distance_map().T, cmap='bone_r')
        hist = plt.hist(f_image, histedges_equalN(f_image, 10), density=True)

    return image, hist[1]


def neuron_map(som, norm_data):
    n_map = []
    for xx in norm_data:
        n_map.append(som.winner(xx))

    return np.array(n_map)


def gen_e_model(n_map, som_label, x, z):
    som_class = []
    for i in range(len(n_map)):
        som_class.append(som_label[n_map[i][0]][n_map[i][1]])

    return som_class


def plot_class_som(som, norm_data, som_label):
    som_class = []
    for xx in norm_data:
        wx, wy = som.winner(xx)   # getting the winner neuron location
        som_class.append(som_label[wx][wy])   # get som's class label

    return som_class


def closest_n(som_label):
    borders = np.array(np.where(som_label == 0)).T
    new_label = np.array(som_label)

    vals = np.where(som_label != 0)
    vals = np.array(vals).T

    for b in borders:
        distance.cdist([b], vals)
        c_idx = distance.cdist([b], vals).argmin()
        cl = som_label[vals[c_idx, 0]][vals[c_idx, 1]]

#         print(b, 'is nearest to', vals[c_idx], 'with class=', cl)
        new_label[b[0], b[1]] = cl

    return new_label


def watershed_level(image, bins, border_width=0.1, plot=False):
    num_bins = len(bins)
    fig, axes = plt.subplots(ncols=4, nrows=num_bins, figsize=(12, num_bins*3),
                             sharex=True, sharey=True)
    ax = axes.ravel()
    mask = []
    all_label = []

    mask2 = []
    all_label2 = []

    for i in range(num_bins):
        val = filters.threshold_local(image, block_size=3 + 2*i)

        mask.append(image < val)
        ax[i*4].imshow(mask[i], cmap=plt.cm.gray, origin='lower')
        all_label.append(measure.label(mask[i]))
#         all_label[i] = KNN(all_label[i])
        all_label[i] = closest_n(all_label[i])
        ax[i*4 + 1].imshow(all_label[i], origin='lower')
        ax[i*4 + 1].title.set_text('it={} n_class={}'.format(i,
                                   np.unique(all_label[i])[-1]))

        mask2.append(image <= bins[i])
        ax[i*4 + 2].imshow(mask2[i], cmap=plt.cm.gray, origin='lower')
        all_label2.append(measure.label(mask2[i]))
#         all_label2[i] = KNN(all_label2[i])
        all_label2[i] = closest_n(all_label2[i])
        ax[i*4 + 3].imshow(all_label2[i], origin='lower')
        ax[i*4 + 3].title.set_text('it={} n_class={}'.format(i,
                                   np.unique(all_label2[i])[-1]))

    return all_label, all_label2


def KNN(som_label, k=5):
    borders = np.array(np.where(som_label == 0)).T
    new_label = np.array(som_label)

    vals = np.where(som_label != 0)
    vals = np.array(vals).T

    for b in borders:
        dist = distance.cdist([b], vals)
        c_idx = np.argpartition(dist, k)
        c_idx = c_idx[0, :k]

        mins_idx = np.array(list(zip(vals[c_idx, 0], vals[c_idx, 1])))
        class_counter = Counter()
        for idx in mins_idx:
            class_counter[som_label[idx[0], idx[1]]] += 1
        cl = class_counter.most_common(1)[0][0]

        new_label[b[0], b[1]] = cl

    return new_label


#def map_new_label(n_map, som_label, x, z):
#    # check distance map
#    plt.figure(figsize=(3, 3))
#    plt.imshow(som_label, origin='lower', cmap='viridis_r')
#
#    som_class = plot_class_som(som, norm_data, som_label)
#    show_new_class(som_class, x, z)


# parameter tuning
# grid search
def grid_search_som(init_num, norm_data, label, seed=10):

    dims = [-5, 0, 5, 10]
    dims += np.array(compute_dim(norm_data.shape[0]))
    idx = np.argwhere(dims <= 0)
    if(idx.size != 0):
        dims = dims[idx[-1][0]:]
    iter_cnts = [2000, 4000, 5000, 6000]
    lr = [0.25, 0.5, 0.75]
    sigma = [1, 2, 2.5, 3, 4]

# # small grid test
#    dims = [5]
#    iter_cnts = [10, 20]
#    lr = [0.75]
#    sigma = [1]

    test_num = init_num
    X_train, X_test, y_train, y_test = train_test_split(norm_data, label,
                                                        random_state=seed)

    hyperpara = [dims, iter_cnts, lr, sigma]
    hyperpara_perm = list(itertools.product(*hyperpara))

    best_acc = 0
    best_comb = []
    best_model = ''

    for comb in hyperpara_perm:
        print(comb[0], " ", comb[1], " ", comb[2], " ", comb[3])

        test_num += 1
        model_name = 'grid_search/grid_XZ_' + str(test_num)

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
        plot_som(som, norm_data, label, save=True, model_name=model_name)
        report = classification_report(y_test, y_pred, digits=4)
        c_matrix = confusion_matrix(y_test, y_pred)

        save_som(som, model_name, report, comb[0],  comb[0], comb[1], comb[2],
                 comb[3], elapsed_time, c_matrix, grid=True)
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
        'learning_rate': list(np.logspace(np.log10(0.5), np.log10(0.75),
                                          base=10, num=100)),
        'sigma': list(np.linspace(g_sigma-1, g_sigma+1, num=30)),
    }
    return param_grid


# random search
def random_search_som(init_num, norm_data, label, grid_search_res,
                      seed=10, max_evals=10):

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
        model_name = 'random_search/random_' + str(test_num)

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

        save_som(som, model_name, c_dim, c_dim, c_it, c_lr, c_sig,
                 elapsed_time, report, c_matrix, rand=True)

    print('Best model found:', best_model)
    print('Best comb found:', best_comb)
    print('Best acc found:', best_acc)

    return best_comb, best_acc
