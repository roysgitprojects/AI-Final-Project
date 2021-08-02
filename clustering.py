import time
from datetime import timedelta

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import skfuzzy
import numpy as np
import csv
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import mannwhitneyu
import matplotlib.cm as cm


def perform_fuzzy_cmeans(points, n_clusters):
    """
    Perform FCM clustering and return predictions
    :param points: points to cluster
    :param n_clusters: number of clusters
    :returns: clustering labels
    """
    cntr, u, _, _, _, _, _ = skfuzzy.cluster.cmeans(points.T, c=n_clusters, m=2, error=0.005, maxiter=1000)
    predictions = np.argmax(u, axis=0)
    return predictions


def perform_hierarchical_clustering(points, n_clusters, linkage='ward'):
    """
    Perform Hierarchical clustering and return predictions
    :param points: points to cluster
    :param n_clusters: number of clusters
    :param linkage: the sub method
    :returns: clustering labels
    """
    # linkages = ['ward', 'average', 'complete', 'single']
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    predictions = hc.fit_predict(points)
    return predictions


def perform_kmeans(points, n_clusters):
    """
    Perform K means clustering and return predictions
    :param points: points to cluster
    :param n_clusters: number of clusters
    :returns: clustering labels
    """
    predictions = KMeans(n_clusters=n_clusters).fit_predict(points)
    return predictions


def perform_gmm(points, n_clusters):
    """
    Perform GMM clustering and return predictions
    :param points: points to cluster
    :param n_clusters: number of clusters
    :returns: clustering labels
    """
    gmm = GaussianMixture(n_components=n_clusters)
    gmm = gmm.fit(points)
    predictions = gmm.predict(points)
    return predictions


def perform_spectral_clustering(points, n_clusters):
    """
    Perform Spectral clustering and return predictions
    :param points: points to cluster
    :param n_clusters: number of clusters
    :returns: clustering labels
    """
    spectral = SpectralClustering(n_clusters=n_clusters)
    predictions = spectral.fit_predict(points)
    return predictions


def perform_dbscan_clustering(points, eps=3):
    dbscan = DBSCAN(eps=eps)  # eps=0.0001, min_samples=100)
    dbscan = dbscan.fit(points)
    return dbscan.labels_


def cluster(points, n_clusters, method, linkage='ward', eps=3):
    """
    Perform clustering and return predictions.
    :param eps: epsilon for dbscan
    :param points: points to cluster
    :param n_clusters: number of clusters
    :param method: clustering method
    :param linkage: if hierarchical clustering than linkage stands for the sub method
    :return: clustering predictions
    """
    if method == 'K means':
        return perform_kmeans(points, n_clusters)
    elif method == 'GMM':
        return perform_gmm(points, n_clusters)
    elif method == 'Fuzzy C Means':
        return perform_fuzzy_cmeans(points, n_clusters)
    elif method == 'Hierarchical':
        return perform_hierarchical_clustering(points, n_clusters, linkage=linkage)
    elif method == 'Spectral':
        return perform_spectral_clustering(points, n_clusters)
    elif method == 'DBSCAN':
        return perform_dbscan_clustering(points, eps)


def plot_clustering(points, predictions, method):
    """
    Visualize the clustering results
    :param points: points to plot
    :param predictions: points labels according to cluster algorithms
    :param method: clustering method
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('%s clustering on the dataset (reduced data)' % method)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    ax.scatter(points[:, 0], points[:, 1], c=predictions, cmap='tab10', alpha=0.8, s=8)
    plt.show()


def cluster_20_times_and_save_results(points, method, linkage='ward', dim_red_method="None"):
    if method == "DBSCAN":
        for eps10 in range(2, 20, 1):
            eps = eps10 / 10
            if dim_red_method == "None":
                path = "clustering results/" + method + "/" + str(eps10) + ".csv"
            else:
                path = "clustering results after dimension reduction/" + dim_red_method + "/" + method + "/" + str(
                    eps10) + ".csv"
            with open(path, "w", newline="") as f:
                for i in range(20):
                    start = time.time()
                    labels = cluster(points=points, method=method, linkage=linkage, n_clusters=0, eps=eps).tolist()
                    writer = csv.writer(f)
                    writer.writerow(labels)
                    end = time.time()
                    print(method + " " + str(eps) + " epsilon" + "number " + str(i) + "time: " + str(
                        timedelta(seconds=end - start)))
    else:
        for n_clusters in range(2, 20):
            if dim_red_method == "None":
                path = "clustering results/" + method + "/" + str(n_clusters) + ".csv"
            else:
                path = "clustering results after dimension reduction/" + dim_red_method + "/" + method + "/" + str(
                    n_clusters) + ".csv"
            if method == "Hierarchical":
                if dim_red_method == "None":
                    path = "clustering results/" + method + " " + linkage + "/" + str(n_clusters) + ".csv"
                else:
                    path = "clustering results after dimension reduction/" + dim_red_method + "/" + method + " " + linkage + "/" + str(
                        n_clusters) + ".csv"
            with open(path, "w", newline="") as f:
                for i in range(20):
                    start = time.time()
                    labels = cluster(points, n_clusters, method, linkage).tolist()
                    writer = csv.writer(f)
                    writer.writerow(labels)
                    end = time.time()
                    print(method + " " + str(n_clusters) + " clusters" + "number " + str(i) + "time: " + str(
                        timedelta(seconds=end - start)))
                    if method == "Hierarchical":
                        print(linkage)


def read_clustering_results(method, linkage='ward', n_clusters=2, eps10=15, dim_red_method="None"):
    """

    :param method: clustering method
    :param linkage: for hierarchical
    :param n_clusters: number of clusters
    :param eps10: epsilon*10 for DBSCAN
    :return: list of 20 times the clustering labels
    """
    prefix = "clustering results" if dim_red_method == "None" else (
            "clustering results after dimension reduction/" + dim_red_method)
    path = prefix + "/" + method + "/" + str(n_clusters) + ".csv"
    if method == "Hierarchical":
        path = prefix + "/" + method + " " + linkage + "/" + str(n_clusters) + ".csv"
    elif method == "DBSCAN":
        path = prefix + "/" + method + "/" + str(eps10) + ".csv"
    with open(path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


def calculate_silhouette_scores(points, method, linkage='ward', dim_red_method="None"):
    prefix = "clustering results" if dim_red_method == "None" else (
            "clustering results after dimension reduction/" + dim_red_method)
    if method == "DBSCAN":
        for eps10 in range(2, 20, 1):
            path = prefix + "/" + method + "/" + str(eps10) + "_silhouette_scores" + ".csv"
            with open(path, "w", newline="") as f:
                list_of_labels = read_clustering_results(method, linkage, eps10=eps10, dim_red_method=dim_red_method)
                for label_time in range(20):
                    try:
                        silhouette_avg = silhouette_score(points, list_of_labels[label_time])
                        writer = csv.writer(f)
                        writer.writerow([silhouette_avg])
                    except:
                        continue
        return

    for n_clusters in range(3, 20):
        prefix = "clustering results" if dim_red_method == "None" else (
                "clustering results after dimension reduction/" + dim_red_method)
        path = prefix + "/" + method + "/" + str(n_clusters) + "_silhouette_scores" + ".csv"
        if method == "Hierarchical":
            path = prefix + "/" + method + " " + linkage + "/" + str(
                n_clusters) + "_silhouette_scores" + ".csv"

        with open(path, "w", newline="") as f:
            list_of_labels = read_clustering_results(method, linkage,
                                                     n_clusters=n_clusters, dim_red_method=dim_red_method)
            for label_time in range(20):
                try:
                    silhouette_avg = silhouette_score(points, list_of_labels[label_time])
                    writer = csv.writer(f)
                    writer.writerow([silhouette_avg])
                except:
                    print("Error in " + method + " linkage " + linkage + "DIM_RED" + dim_red_method)


def u_test(scores_method_1, scores_method2):
    """
    Returns P value. if p<<0.05 the first scores better than the second
    :param scores_method_1: first method's scores
    :param scores_method2: second method's scores
    :returns: p value
    """
    mann_whitneyu = mannwhitneyu(scores_method_1, scores_method2, alternative='greater')
    # if p value<0.05 than we can say nmi1>nmi2. Therefore, clustering method 1 is better than 2.
    return mann_whitneyu.pvalue


def read_silhouette_scores(method, linkage='ward', n_clusters=2, eps10=15, dim_red_method="None"):
    prefix = "clustering results" if dim_red_method == "None" else (
            "clustering results after dimension reduction/" + dim_red_method)
    path = prefix + "/" + method + "/" + str(n_clusters) + "_silhouette_scores" + ".csv"
    if method == "DBSCAN":
        path = prefix + "/" + method + "/" + str(eps10) + "_silhouette_scores" + ".csv"

    elif method == "Hierarchical":
        path = prefix + "/" + method + " " + linkage + "/" + str(
            n_clusters) + "_silhouette_scores" + ".csv"

    with open(path, newline="") as f:
        reader = csv.reader(f)
        data = list(reader)
    scores = [float(score) for sublist in data for score in sublist]
    return scores


def get_dictionary_of_silhouette_scores(method, linkage='ward', dim_red_method="None"):
    scores = {}
    if method == "DBSCAN":
        for eps10 in range(2, 18, 1):
            scores[eps10] = read_silhouette_scores(method, eps10=eps10, dim_red_method=dim_red_method)
    else:
        for n_clusters in range(3, 20):
            scores[n_clusters] = read_silhouette_scores(method, n_clusters=n_clusters, linkage=linkage,
                                                        dim_red_method=dim_red_method)
    return scores


def compare_silhouette_scores_same_method(method, linkage='ward', dim_red_method="None"):
    print(method)
    if method == "Hierarchical":
        print(linkage)
    scores = get_dictionary_of_silhouette_scores(method=method, linkage=linkage, dim_red_method=dim_red_method)
    for key1 in scores:
        better_than = 0
        for key2 in scores:
            if key1 != key2:
                try:
                    pvalue = u_test(scores_method_1=scores[key1], scores_method2=scores[key2])
                    if pvalue < 0.05:
                        better_than += 1
                        print(
                            str(key1) + " is better than " + str(key2) + " with $p-value = " + str(
                                pvalue) + " << 0.05$")
                except:
                    continue
        print("better than: " + str(better_than))


def get_silhouette_scores_all_methods_best_params(best_params_dict, dim_red_method="None"):
    scores = {}
    linkages = ['ward', 'average', 'complete', 'single']
    for method in best_params_dict:
        if method in linkages:  # its a sub-method of hierarchical
            scores[method] = read_silhouette_scores(method="Hierarchical", linkage=method,
                                                    n_clusters=best_params_dict[method], dim_red_method=dim_red_method)
        elif method == "DBSCAN":
            scores[method] = read_silhouette_scores(method=method, eps10=best_params_dict[method],
                                                    dim_red_method=dim_red_method)
        else:
            scores[method] = read_silhouette_scores(method=method, n_clusters=best_params_dict[method],
                                                    dim_red_method=dim_red_method)
    return scores


def compare_silhouette_scores_all_methods_best_params(scores):
    for method1 in scores:
        better_than = 0
        for method2 in scores:
            if method1 != method2:
                try:
                    pvalue = u_test(scores_method_1=scores[method1], scores_method2=scores[method2])
                    if pvalue < 0.05:
                        better_than += 1
                        print(
                            str(method1) + " is better than " + str(method2) + " with $p-value = " + str(
                                pvalue) + " << 0.05$")
                except:
                    continue
        print("better than: " + str(better_than))


def plot_best_clustering_and_silhouette_score(X, clustering_method, n_clusters, cluster_labels, real_labels, reducedX):
    # Create a subplot with 1 row and 3 columns
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    print([0, len(X) + (n_clusters + 1) * 10])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    print(sample_silhouette_values)
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()
        print(cluster_labels)

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        print(np.arange(y_lower, y_upper))
        print(ith_cluster_silhouette_values)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter3D(reducedX[:, 0], reducedX[:, 1], reducedX[:, 2], marker='.', s=30, lw=0, alpha=0.7,
                  c=colors, edgecolor='k')

    ax2.set_title("Visualization of the clustered data.")
    ax2.set_xlabel("The 1st feature")
    ax2.set_ylabel("The 2nd feature")
    ax2.set_zlabel("The 3rd feature")

    # 3rd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(real_labels.astype(float) / n_clusters)
    ax3.scatter(reducedX[:, 0], reducedX[:, 1], reducedX[:, 2], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    ax3.set_title("Visualization of the real labels of the data.")
    ax3.set_xlabel("The 1st feature")
    ax3.set_ylabel("The 2nd feature")
    ax3.set_zlabel("The 3rd feature")

    plt.suptitle(("Silhouette analysis for " + clustering_method + " clustering on the data "
                                                                   "with eps = %d" % 1 + " visualized by PCA"),
                 fontsize=16, fontweight='bold')

    plt.show()

# if __name__ == '__main__':
# best_params_dict = {'K means': 11, 'GMM': 10, 'Fuzzy C Means': 3, 'ward': 10, 'average': 19, 'complete': 19,
#                     'single': 19, 'Spectral': 6, 'DBSCAN': 10}
# scores = get_silhouette_scores_all_methods_best_params(best_params_dict)
# print(compare_silhouette_scores_all_methods_best_params(scores))
