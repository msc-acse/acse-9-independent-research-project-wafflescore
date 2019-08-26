import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from MiscHelpers import search_list

# import logging
# logging.basicConfig(filename='log_filename.txt', level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# import logging
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# fh = logging.FileHandler('log_filename.txt')
# fh.setLevel(logging.DEBUG)
# fh.setFormatter(formatter)
# logger.addHandler(fh)

# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# ch.setFormatter(formatter)
# logger.addHandler(ch)

# logger.debug('This is a test log message.')


def check_nan_inf(in_data):
    """Check the input data for infinite and NaN values.

    Parameters
    ----------
    in_data : np.array or list
        data matrix

    Returns
    -------
    np.array
        numpy array of the index of NaNs
    np.array
        numpy array of the index of infinite
    """
    nan_pos = np.argwhere(np.isnan(in_data))
    inf_pos = np.argwhere(np.isinf(in_data))
    # logger.debug("Index with NaN:%s\n" % nan_pos)
    # logger.debug("Index with INF:%s\n" % inf_pos)

    return nan_pos, inf_pos


def replace_nan_inf(in_data, re_inf=-9999):
    """Replace the NaN and infinite values in the input data.

    Parameters
    ----------
    in_data : np.array or list
        data matrix
    re_inf : int, optional
        integer number to replace the infinite value, by default -9999

    Returns
    -------
    np.array
        numpy array of the input data without NaN and infinite values.
    """
    logging.info("Replacing INF with %d" % re_inf)
    inf_pos = np.argwhere(np.isinf(in_data))

    # replace nan with number
    in_data = np.nan_to_num(in_data)

    # replace inf with number
    in_data[inf_pos[:, 0], inf_pos[:, 1]] = re_inf
    return in_data


def plot_hist(in_data, label, col_name):
    """Plotting the histogram of the input data.

    Parameters
    ----------
    in_data : np.array
        data matrix
    label : np.array or list
        the true label of each data point
    col_name : list of string
        list of the properties presented in the input data
        eg. ['vp', 'vs', 'dn', 'vp/vs', 'qp', 'qs', 'x', 'z']

    Returns
    -------
    pandas.core.frame.DataFrame
        dataframe of the input data
    """
    classes = np.unique(label)
    n_class = len(classes)
    col_name.append('Class')

    hist_data = np.append(in_data, label, axis=1)
    df = pd.DataFrame(hist_data,            # values
                      columns=col_name)     # column name
    df.describe()

    # Plot histogram of each physical property
    fig, ax = plt.subplots(ncols=len(col_name)-1, nrows=n_class,
                           figsize=(20, 30))
    fig.tight_layout()
    for i in range(len(col_name)-1):
        logging.debug("Plotting the histogram %s of each class." % col_name[i])
        df.hist(column=col_name[i], by=col_name[-1],
                figsize=(4, 20), ax=ax[:, i])
        ax[-1, i].set_xlabel(col_name[i], fontsize=10)

    for i in range(n_class):
        ax[i, 0].set_ylabel('Class %d' % classes[i], fontsize=10)
        for j in range(len(col_name)-1):
            ax[i, j].set_title("")

    return df.describe()


def convLabel(in_label):
    """Convert 2d label into 1d numpy array

    Parameters
    ----------
    in_label : np.array
        2d numpy of the class label

    Return
    ------
    numpy.ndarray
        flatten numpy array of the class label
    """
    out_label = np.reshape(in_label, in_label.shape[0] * in_label.shape[1])

    return out_label


def convData(data_file):
    """Convert npz data into 2d numpy array

    Parameters
    ----------
    data_file : numpy.lib.npyio.NpzFile
        npz file which stored all the available properties

    Returns
    -------
    np.array
        2d numpy array
    """
    init_data = []
    for key in data_file.files:
        val = data_file[key]
        if(len(val.shape) == 2):
            val = np.reshape(val, val.shape[0] * val.shape[1])

        init_data.append(val)

    # convert X and Z to meshgrid coordinate
    x_idx = search_list(data_file.files, 'x')[0]
    z_idx = search_list(data_file.files, 'z')[0]
    grid_X, grid_Z = np.meshgrid(init_data[x_idx], init_data[z_idx])   # X, Z
    if(x_idx > z_idx):
        del init_data[x_idx]
        del init_data[z_idx]
    else:
        del init_data[z_idx]
        del init_data[x_idx]

    val = np.reshape(grid_X, grid_X.shape[0] * grid_X.shape[1])
    init_data.append(val)

    val = np.reshape(grid_Z, grid_Z.shape[0] * grid_Z.shape[1])
    init_data.append(val)

    init_data = np.transpose(np.array(init_data))
    logging.debug("Aggregated shape: (%d, %d)" %
                  (init_data.shape[0], init_data.shape[1]))
    return init_data


# data cleanup
def data_cleanup(in_data, water_idx, col_name, re_inf):
    """Cleaning up the input data by removing the known water index,
    dividing Vp, Vs, and density value by 1000,
    capped Vp/Vs ratio to the a maximum value of 10,
    and applying log to Qp and Qs.

    Parameters
    ----------
    init_data : np.array or list
        data matrix
    water_idx : np.array or list
        list of all the water index
    col_name : list of string
        list of the properties presented in the input data
        eg. ['vp', 'vs', 'dn', 'vp/vs', 'qp', 'qs', 'x', 'z']
    re_inf : int
        integer number to replace the infinite value

    Returns
    -------
    numpy.ndarray
        2d numpy array of data that was cleaned up
    """

    # find index of each properties
    vp_idx = search_list(col_name, 'vp')
    vs_idx = search_list(col_name, 'vs')
    dn_idx = search_list(col_name, 'dn')
    vpvs_idx = search_list(col_name, 'vp/vs')
    qp_idx = search_list(col_name, 'qp')
    qs_idx = search_list(col_name, 'qs')

    logging.debug('Index of Vp: %s' % vp_idx)
    logging.debug('Index of Vs: %s' % vs_idx)
    logging.debug('Index of Density: %s' % dn_idx)
    logging.debug('Index of Vp/Vs: %s' % vpvs_idx)
    logging.debug('Index of Qp: %s' % qp_idx)
    logging.debug('Index of Qs: %s' % qs_idx)

    # check and replace NaN and INF values
    data = replace_nan_inf(in_data, re_inf=re_inf)

    # priori info = water location
    data = np.delete(data, water_idx, axis=0)

    # divide by 1000 on Vp, Vs, and density
    if(vp_idx):
        data[:, vp_idx] = data[:, vp_idx] / 1000
    if(vs_idx):
        data[:, vs_idx] = data[:, vs_idx] / 1000
    if(dn_idx):
        data[:, dn_idx] = data[:, dn_idx] / 1000

    # capped Vp/Vs ratio to maximum of 10
    if(vpvs_idx):
        cap_num = 10    # try with 3 or 4
        vpvs_capped_idx = np.where(data[:, vpvs_idx] > cap_num)[0]
        data[vpvs_capped_idx, vpvs_idx] = cap_num

    # apply log to Qp and Qs
    if(qp_idx):
        data[:, qp_idx] = np.log(data[:, qp_idx])
    if(qs_idx):
        data[:, qs_idx] = np.log(data[:, qs_idx])

    return data


def compMeanStd(in_data):
    """Computes the mean and standard derivation

    Parameters
    ----------
    in_data : np.array or list
        data matrix

    Returns
    -------
    list
        list of each column's mean
    list
        list of each column's starndard derivation
    """
    means = np.mean(in_data, axis=0)
    stds = np.std(in_data, axis=0)

    return means, stds


def normalize(in_data, means, stds, model=""):
    """Normalize using inputted mean and standard deviation

    Parameters
    ----------
    in_data : np.array or list
        data matrix
    means : list
        list of each property's mean
    stds : list
        list of each property's starndard derivation
    model : str, optional
        model name, will be used as part of the normalized data savefile name,
        by default ""

    Returns
    -------
    numpy.ndarray
        2d numpy array of the normalized data
    """
    for i in range(in_data.shape[1]):
        in_data[:, i] -= means[i]
        in_data[:, i] /= stds[i]
    if(model):
        fdir = 'data/' + model + '_norm_data.npy'
        np.save(fdir, in_data)
        logging.info('Normalized data saved at: %s' % fdir)

    return in_data

# ================ unused =====================

# apply gaussian smoothing to preprocessed data (not yet normalized)
def smoothing(in_data, sigma, model=""):
    sm_data = np.zeros_like(in_data)
    for i in range(6):
        sm_data[:, i] = gaussian_filter(in_data[:, i], sigma=sigma)

    if(model):
        fdir = 'data/' + model + 'sm' + str(sigma) + '_data.npy'
        print('Smooth data saved at:', fdir)
        np.save(fdir, sm_data)

    return sm_data


def addNoise(in_data, noise_deg, model=""):
    means, stds = compMeanStd(in_data)

    noisy_data = np.array(in_data)

    for i in range(6):
        noise = np.random.normal(means[i], stds[i], in_data[:, i].shape)
        noisy_data[:, i] += noise * noise_deg

    if(model):
        fdir = 'data/' + model + 'ns' + str(noise_deg) + '_data.npy'
        print('Noisy data saved at:', fdir)
        np.save(fdir, noisy_data)

    return noisy_data


# data2 = dataPreprocess(input_npz['classes'], output_smooth_npz, col_n, model=model)
def dataPreprocess(label_file, data_file, col_name, model=""):
    init_label = convLabel(label_file)
    init_data = convData(data_file)

    # remove water and perform data preprocessing
    water_idx = np.where(init_label == 0)
    label = np.delete(init_label, water_idx)
    data = data_cleanup(init_data, water_idx, col_name, re_inf=-9999)
    logging.debug("Water removed shape: (%d, %d)" %
                  (data.shape[0], data.shape[1]))

    if (model):
        fdir = 'data/' + model + '_clean_data.npy'
        np.save(fdir, data)
        logging.info('Data saved at: %s' % fdir)

        fdir = 'data/' + model + '_data_label.npy'
        np.save(fdir, label)
        logging.info('Data label saved at: %s' % fdir)

        fdir = 'data/' + model + '_xz_pos.npy'
        np.save(fdir, data[:, -2:])
        logging.info('XZ positions saved at: %s' % fdir)

    return data
