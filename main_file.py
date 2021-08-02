import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, classification_report, silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
import csv
import autoencoder
import clustering
from dimension_reduction import read_dimension_reduction_results_to_lil, apply_dimension_reduction_method
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, normalize, Normalizer
import nn
from collections import Counter


def prepare_dataset():
    column_names = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
                    "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-surface-above-ring",
                    "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
                    "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
    df = pd.read_csv("mushrooms_data.txt", names=column_names)
    df.describe().to_csv("described_data.csv")  # note there's only one veil-type
    # the data is categorical, do a label encoder will be used to make it ordinal
    df = df.astype('category')
    label_encoder = LabelEncoder()
    for i in df.columns:
        df[i] = label_encoder.fit_transform(df[i])
    odor = df["odor"]
    # since we want to predict the mushrooms' odor it's needed to drop this feature from the df
    X = df.drop(["odor"], axis=1)
    print(df.head())
    # X = pd.get_dummies(X)
    # print(X.head())
    # save the ordinal data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    pd.DataFrame(X).to_csv("ordinal_mushrooms_data.csv")
    print("data set is:")
    print(X)
    return X, odor


def get_dataset():
    X, _ = prepare_dataset()
    return X


def get_real_labels():
    _, labels = prepare_dataset()
    return labels


def plot_boxplot():
    """
    Plot a boxplot to visualize anomalies
    :return: None
    """
    column_names = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "gill-attachment",
                    "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-surface-above-ring",
                    "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
                    "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
    data = pd.DataFrame(get_dataset(), columns=column_names)
    plt.figure(figsize=(10, 7))
    plt.title("Box Plot", fontdict={'fontsize': 24})
    sns.boxplot(data=data)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.show()


def read_nmi(path):
    with open(path) as f:
        reader = csv.reader(f)
        data = list(reader)
    return [float(nmi) for sublist in data for nmi in sublist]


if __name__ == '__main__':
    X, odor = prepare_dataset()
    points = X

    # print(Counter(odor))

    # labels = nn.main(X, odor)
    methods = ["ae", "PCA", "CMDS",
               "ICA"]  # ["ae", "PCA", "CMDS", "NCMDS", "ICA", "ISOMAP", "LLE", "LAPLACIAN EIGENMAPS"]
    clustering_methods = ['K means', 'GMM', 'Fuzzy C Means', 'Hierarchical',
                          'DBSCAN']  # ['K means', 'GMM', 'Fuzzy C Means', 'Hierarchical', 'Spectral', 'DBSCAN']
    linkages = ['ward', 'average', 'complete', 'single']
    plot_boxplot()
    for clustering_method in clustering_methods:
        if clustering_method == "Hierarchical":
            for linkage in linkages:
                clustering.cluster_20_times_and_save_results(points, clustering_method, linkage)
        else:
            clustering.cluster_20_times_and_save_results(points, clustering_method)

    # calculate silhouette scores
    for clustering_method in clustering_methods:
        if clustering_method == "Hierarchical":
            for linkage in linkages:
                clustering.calculate_silhouette_scores(points, clustering_method, linkage)
        else:
            clustering.calculate_silhouette_scores(points, clustering_method)

    # compare silhouette scores to find best parameters for each method
    for clustering_method in clustering_methods:
        if clustering_method == "Hierarchical":
            for linkage in linkages:
                clustering.compare_silhouette_scores_same_method(clustering_method, linkage)
        else:
            clustering.compare_silhouette_scores_same_method(clustering_method)

    # compare the different method to find the best one
    best_params_dict = {'K means': 11, 'GMM': 10, 'Fuzzy C Means': 3, 'ward': 10, 'average': 19, 'complete': 19,
                        'single': 19, 'Spectral': 6, 'DBSCAN': 10}
    scores = clustering.get_silhouette_scores_all_methods_best_params(best_params_dict)
    print(clustering.compare_silhouette_scores_all_methods_best_params(scores))

    for i in range(20):
        print(normalized_mutual_info_score(odor, clustering.read_clustering_results(method="DBSCAN", eps10=10)[i]))
    print("after dim red")
    for method in methods:
        apply_dimension_reduction_method(X, method)  # , odor)

    # plot best clustering
    list_of_xyz = read_dimension_reduction_results_to_lil("PCA")
    [x, y, z] = list_of_xyz
    reduced_data_for_visualization = (np.array([np.array(i) for i in list_of_xyz])).T
    labels = clustering.cluster(points=X, method='DBSCAN', eps=1, n_clusters=0)
    n_clusters = len(np.unique(labels))
    print(labels)
    print(n_clusters)
    clustering.plot_best_clustering_and_silhouette_score(X, clustering_method="DBSCAN", n_clusters=n_clusters,
                                                         cluster_labels=labels, real_labels=odor,
                                                         reducedX=reduced_data_for_visualization)

    # Third approach
    for method in methods:
        list_of_xyz = read_dimension_reduction_results_to_lil(method)
        [x, y, z] = list_of_xyz
        points = (np.array([np.array(i) for i in list_of_xyz])).T
        print(points)
        # cluster
        for clustering_method in clustering_methods:
            if clustering_method == "Hierarchical":
                for linkage in linkages:
                    clustering.cluster_20_times_and_save_results(points, clustering_method, linkage,
                                                                 dim_red_method=method)
            else:
                clustering.cluster_20_times_and_save_results(points, clustering_method, dim_red_method=method)
        # calculate silhouette scores
        for clustering_method in clustering_methods:
            print("silhouette for " + clustering_method + " " + method)
            if clustering_method == "Hierarchical":
                for linkage in linkages:
                    print(linkage)
                    clustering.calculate_silhouette_scores(points, clustering_method, linkage, dim_red_method=method)
            else:
                clustering.calculate_silhouette_scores(points, method=clustering_method, dim_red_method=method)

    # compare silhouette scores to find best parameters for each method
    for method in methods:
        list_of_xyz = read_dimension_reduction_results_to_lil(method)
        [x, y, z] = list_of_xyz
        points = (np.array([np.array(i) for i in list_of_xyz])).T
        print(points)
        for clustering_method in clustering_methods:
            if clustering_method == "Hierarchical":
                for linkage in linkages:
                    clustering.compare_silhouette_scores_same_method(clustering_method, linkage, dim_red_method=method)
            else:
                clustering.compare_silhouette_scores_same_method(clustering_method, dim_red_method=method)

    # compare the different methods to find the best one for each dimension reduction method
    for method in methods:
        print(method)
        if method == "ae":
            best_params_dict = {'K means': 4, 'Fuzzy C Means': 4, 'ward': 4, 'average': 15, 'complete': 5,
                                'single': 3}
        elif method == "PCA":
            best_params_dict = {'K means': 11, 'GMM': 10, 'Fuzzy C Means': 11, 'ward': 11, 'average': 18,
                                'complete': 12, 'single': 19, 'DBSCAN': 2}
        elif method == "CMDS":
            best_params_dict = {'K means': 11, 'GMM': 14, 'Fuzzy C Means': 8, 'ward': 12, 'average': 7, 'complete': 7,
                                'single': 18, 'DBSCAN': 3}
        elif method == "ICA":
            best_params_dict = {'K means': 11, 'GMM': 10, 'Fuzzy C Means': 10, 'ward': 11, 'average': 15,
                                'complete': 18, 'single': 17}
        else:
            raise Exception("This Dimension Reduction method is not supported or there's a typo")
        scores = clustering.get_silhouette_scores_all_methods_best_params(best_params_dict, dim_red_method=method)
        clustering.compare_silhouette_scores_all_methods_best_params(scores)

    # compare the results of the different dimension reduction methods to find the best one
    dimension_reduction_dict = {"ae": {'K means': 4}, "PCA": {'K means': 11}, "CMDS": {'ward': 12},
                                "ICA": {'K means': 11}}
    best_scores_dict = {}
    for dim_red_method in dimension_reduction_dict:
        for key in dimension_reduction_dict[dim_red_method]:
            if key in linkages:
                method = "Hierarchical"
            else:
                method = key
            best_scores_dict[dim_red_method + " " + key] = clustering.read_silhouette_scores(method=method, linkage=key,
                                                                                             n_clusters=
                                                                                             dimension_reduction_dict[
                                                                                                 dim_red_method][key],
                                                                                             dim_red_method=dim_red_method)
    clustering.compare_silhouette_scores_all_methods_best_params(best_scores_dict)

    for i in range(20):
        print(normalized_mutual_info_score(odor, clustering.read_clustering_results(method="K means", n_clusters=11,
                                                                                    dim_red_method="PCA")[i]))

    # plot best clustering
    list_of_xyz = read_dimension_reduction_results_to_lil("PCA")
    [x, y, z] = list_of_xyz
    X = (np.array([np.array(i) for i in list_of_xyz])).T
    labels = clustering.cluster(points=X, method='K means', n_clusters=11)
    n_clusters = 11
    print(labels)
    print(n_clusters)
    clustering.plot_best_clustering_and_silhouette_score(X, clustering_method="K means", n_clusters=n_clusters,
                                                         cluster_labels=labels, real_labels=odor,
                                                         reducedX=X)
    # Compare approaches
    nmi_first_approach = read_nmi("compare approaches/first_approach_nmi_best.txt")
    nmi_second_approach = read_nmi("compare approaches/second_approach_nmi_best.txt")
    nmi_third_approach = read_nmi("compare approaches/third_approach_nmi_best_classify.txt")
    nmi_scores = {"first approach": nmi_first_approach, "second approach": nmi_second_approach,
                  "third approach": nmi_third_approach}
    for key1 in nmi_scores:
        for key2 in nmi_scores:
            if key1 != key2:
                p_val = clustering.u_test(nmi_scores[key1], nmi_scores[key2])
                if p_val < 0.05:
                    print(key1 + "is better than " + key2 + "with p-value = " + str(p_val) + "<<0.05")

    # labels = nn.main(X, odor)
    # for method in methods:
    #     list_of_xyz = read_dimension_reduction_results_to_lil(method)
    #     [x, y, z] = list_of_xyz
    #     array = (np.array([np.array(i) for i in list_of_xyz])).T
    #     labels = clustering.cluster(points=points, eps=0.9, method="DBSCAN",n_clusters=0)
    #     print(labels)
    #     # print(classification_report(odor, labels))
    #     fig = plt.figure()
    #     ax = fig.add_subplot(121, projection='3d')
    #     ax.title.set_text("NN")
    #     ax.set_xlabel('First Principal Component')
    #     ax.set_ylabel('Second Principal Component')
    #     ax.scatter3D(x, y, z, c=labels, alpha=0.8, s=8)
    #     ax = fig.add_subplot(122, projection='3d')
    #     ax.title.set_text(method)
    #     ax.set_xlabel('First Principal Component')
    #     ax.set_ylabel('Second Principal Component')
    #     ax.scatter3D(x, y, z, c=odor, alpha=0.8, s=8)
    # plt.show()
