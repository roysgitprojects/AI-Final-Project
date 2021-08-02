from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.decomposition import FastICA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
import matplotlib.pyplot as plt
import csv

import main_file

DIM = 3


def apply_dimension_reduction_method(data, method, plot=True):
    # methods = ["PCA", "CMDS", "NCMDS", "ICA", "ISOMAP", "LLE", "LAPLACIAN EIGENMAPS"]
    if method == "PCA":
        reduced_data = apply_pca(data)
    elif method == "CMDS":
        reduced_data = apply_cmds(data)
    elif method == "NCMDS":
        reduced_data = apply_ncmds(data)
    elif method == "ICA":
        reduced_data = apply_ica(data)
    elif method == "ISOMAP":
        reduced_data = apply_isomap(data)
    elif method == "LLE":
        reduced_data = apply_lle(data)
    elif method == "LAPLACIAN EIGENMAPS":
        reduced_data = apply_laplacian_eigenmaps(data)
    else:
        raise Exception("No such dimension reduction algorithm")
    if DIM == 3:
        x, y, z = zip(*reduced_data)
        if plot:
            fig = plt.figure()
            # ax = fig.add_subplot(111)
            ax = plt.axes(projection="3d")
            plt.title(method)
            ax.set_xlabel('First Principal Component')
            ax.set_ylabel('Second Principal Component')
            ax.set_zlabel('Third Principal Component')
            ax.scatter3D(x, y, z, c=main_file.get_real_labels(), alpha=0.8, s=8)
        with open("dimension_reduction/" + method + "_3d.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows([x, y, z])
    else:
        x, y = zip(*reduced_data)
        if plot:
            fig = plt.figure()
            # ax = fig.add_subplot(111)
            ax = plt.axes()
            plt.title(method)
            ax.set_xlabel('First Principal Component')
            ax.set_ylabel('Second Principal Component')
            ax.scatter(x, y, c=main_file.get_real_labels(), alpha=0.8, s=8)
            # plt.show()
    return reduced_data


def apply_pca(data):
    model = PCA(n_components=DIM)
    return model.fit_transform(data)


def apply_cmds(data):
    model = MDS(n_components=DIM)
    return model.fit_transform(data)


def apply_ncmds(data):
    model = MDS(n_components=DIM, metric=False)
    return model.fit_transform(data)


def apply_ica(data):
    model = FastICA(n_components=DIM, max_iter=1500)
    return model.fit_transform(data)


def apply_isomap(data):
    model = Isomap(n_components=DIM)
    return model.fit_transform(data)


def apply_lle(data):
    model = LocallyLinearEmbedding(n_components=DIM, eigen_solver='dense')
    return model.fit_transform(data)


def apply_laplacian_eigenmaps(data):
    model = SpectralEmbedding(n_components=DIM)
    return model.fit_transform(data)


def read_dimension_reduction_results_to_lil(method):
    with open("dimension_reduction/" + method + '_3d.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    x_str, y_str, z_str = data
    x = [float(i) for i in x_str]
    y = [float(i) for i in y_str]
    z = [float(i) for i in z_str]
    return [x, y, z]
