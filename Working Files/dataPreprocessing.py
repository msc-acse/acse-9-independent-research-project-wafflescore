import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
#from skimage.util import random_noise


def check_inf_nan(in_data):

    # display NaN index
    nan_pos = np.argwhere(np.isnan(in_data))
    print("Index with NaN:\n", nan_pos)

    # display inf index
    inf_pos = np.argwhere(np.isinf(in_data))
    print("Index with inf:\n", inf_pos)

    return nan_pos, inf_pos


def replace_inf_nan(in_data, re_inf=9999):
    print("Replacing inf with", re_inf)
    inf_pos = np.argwhere(np.isinf(in_data))

    # replace nan with number
    in_data = np.nan_to_num(in_data)

    # replace inf with number
    in_data[inf_pos[:, 0], inf_pos[:, 1]] = re_inf
    return in_data, inf_pos

# def normalize_old(in_data):
#     norm_data = np.apply_along_axis(lambda x: x/np.linalg.norm(in_data), 1, in_data)
#     return norm_data

# def normalize(in_data, mean, std):
#     norm_data = in_data - mean
#     norm_data /= std
#     return norm_data


def plot_hist(in_data, label, n_class, titles):
    # explore the data
    hist_data = np.append(label, in_data, axis=1)
#    print(hist_data.shape)
    df = pd.DataFrame(hist_data,            # values
                      columns=titles[:-2])  # column name
    df.describe()
    
    # Plot histogram of each graph
#    print(n_class)
    fig, ax = plt.subplots(ncols=6, nrows=n_class, figsize=(20, 30))
    fig.tight_layout()
    for i in range(6):
        print("Plotting the histogram", titles[i+1], "of each class.")
        df.hist(column=titles[i+1], by=titles[0], figsize=(4, 20), ax=ax[:, i])

    return df.describe()


# function to convert npz to 3d numpy array.
def convLabel(label_file):
    label = label_file
    label = np.reshape(label, label.shape[0] * label.shape[1])

#    print("Class number")
#    print(np.shape(label), np.nanmin(label), np.nanmax(label))

    return label


def convData(data_file):
    init_data = []
    for key in data_file.files:
        val = data_file[key]
        if(len(val.shape) == 2):
#            print(key)
#            print(val.shape, np.nanmin(val), np.nanmax(val))
            val = np.reshape(val, val.shape[0] * val.shape[1])

        init_data.append(val)

    # convert X and Z to meshgrid coordinate
    grid_X, grid_Z = np.meshgrid(init_data[-2], init_data[-1])
                                 # X, Z
    del init_data[-1]
    del init_data[-1]

    val = np.reshape(grid_X, grid_X.shape[0] * grid_X.shape[1])
    init_data.append(val)
#    print("X\n", val.shape, np.nanmin(val), np.nanmax(val))

    val = np.reshape(grid_Z, grid_Z.shape[0] * grid_Z.shape[1])
    init_data.append(val)
#    print("Z\n", val.shape, np.nanmin(val), np.nanmax(val))

    init_data = np.transpose(np.array(init_data))
    print("Aggregated shape:", init_data.shape)
    return init_data


# data cleanup
def data_cleanup(init_data, water_idx, re_inf):
    # check and replace NaN and INF values
    data, inf_pos = replace_inf_nan(init_data, re_inf=re_inf)

    # priori info = water location
    data = np.delete(data, water_idx, axis=0)
    # capped Vp/Vs ratio to maximum of 10
    vpvs_capped_idx = np.where(data[:, 3] > 10)
    data[vpvs_capped_idx, 3] = 10

    # apply log to Qp and Qs
    data[:, 4:6] = np.log(data[:, 4:6])

    # divide by 1000 on Vp, Vs, and density
    data[:, 0:3] = data[:, 0:3] / 1000

    return data


def compMeanStd(in_data):
    means = np.mean(in_data, axis=0)
    stds = np.std(in_data, axis=0)

    return means, stds


def dataPreprocess(label_file, data_file, model=""):
    init_label = convLabel(label_file)
    init_data = convData(data_file)

    # remove water and perform data preprocessing
    water_idx = np.where(init_label == 0)
    label = np.delete(init_label, water_idx)
    re_inf = -9999
    data = data_cleanup(init_data, water_idx, re_inf)
    print("Water removed shape:", data.shape)

    fdir = 'data/' + model + 'clean_data.npy'
    np.save(fdir, data)
    print('Data saved at:', fdir)

    fdir = 'data/' + model + 'data_label.npy'
    np.save(fdir, label)
    print('Data label saved at:', fdir)

    fdir = 'data/' + model + 'xz_pos.npy'
    np.save(fdir, data[:, -2:])
    print('XZ positions saved at:', fdir)

    return data


def normalize(in_data, means, stds, model=""):
    for i in range(in_data.shape[1]):
        in_data[:, i] -= means[i]
        in_data[:, i] /= stds[i]

    fdir = 'data/' + model + 'norm_data.npy'
    np.save(fdir, in_data)
    print('Normalized data saved at:', fdir)

    return in_data, means, stds


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
#        noise = random_noise(in_data[:, i], mean=means[i], var=stds[i],
#                             mode='gaussian', seed=10)
        noise = np.random.normal(means[i], stds[i], in_data[:, i].shape)
        noisy_data[:, i] += noise * noise_deg

    if(model):
        fdir = 'data/' + model + 'ns' + str(noise_deg) + '_data.npy'
        print('Noisy data saved at:', fdir)
        np.save(fdir, noisy_data)

    return noisy_data
