import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, classification_report, silhouette_score, confusion_matrix, \
    explained_variance_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
import csv
import autoencoder
import clustering
import dimension_reduction
from dimension_reduction import read_dimension_reduction_results_to_lil, apply_dimension_reduction_method
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, normalize, Normalizer
import nn
from collections import Counter
from sklearn.svm import SVC
import main_file
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
import seaborn as sns


def prepare_all():
    encoder = LabelEncoder()
    column_names = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
                    "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-surface-above-ring",
                    "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
                    "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
    data_as_is = pd.read_csv("mushrooms_data.txt", names=column_names)
    data_as_is = data_as_is.astype('category')
    # data_as_is = data_as_is.apply(lambda series: pd.Series(
    #     encoder.fit_transform(series[series.notnull()]),
    #     index=series[series.notnull()].index
    # ))
    series_stack = data_as_is.stack().astype(str)
    data_as_is = pd.Series(
        encoder.fit_transform(series_stack),
        index=series_stack.index
    ).unstack()

    missing_data_as_is = pd.read_csv("missing data/mushrooms_data_missing.txt", names=column_names)
    missing_data_as_is = missing_data_as_is.astype('category')
    missing_data_as_is = missing_data_as_is.replace('-', np.nan)  # mark missing values with None
    print("Missing values")
    print(missing_data_as_is.isna().sum())
    # missing_data_as_is = missing_data_as_is.apply(lambda series: pd.Series(
    #     encoder.transform(series[series.notnull()]),
    #     index=series[series.notnull()].index
    # ))
    series_stack = missing_data_as_is.stack().astype(str)
    missing_data_as_is = pd.Series(
        encoder.transform(series_stack),
        index=series_stack.index
    ).unstack()
    missing_data_as_is = missing_data_as_is.values

    # nan_idxs
    df = pd.read_csv("missing data/mushrooms_data_missing.txt", names=column_names)
    df = df.astype('category')
    nan_idxs = []
    for i in range(len(df)):
        for j in range(len(df.columns)):
            if df.iloc[i, j] == '-':
                if [i, j] in nan_idxs:
                    print([i, j])
                nan_idxs.append([i, j])
    print(len(nan_idxs))
    print(len((np.array(nan_idxs))))

    # imputed with median
    imputed_with_median = pd.read_csv("missing data/mushrooms_data_missing.txt", names=column_names)
    imputed_with_median = imputed_with_median.astype('category')
    imputed_with_median = imputed_with_median.replace('-', np.nan)  # mark missing values with None
    # imputed_with_median = imputed_with_median.apply(lambda series: pd.Series(
    #     encoder.transform(series[series.notnull()]),
    #     index=series[series.notnull()].index
    # ))
    series_stack = imputed_with_median.stack().astype(str)
    imputed_with_median = pd.Series(
        encoder.transform(series_stack),
        index=series_stack.index
    ).unstack()
    for column in imputed_with_median.columns:
        imputed_with_median.loc[imputed_with_median[column].isnull(), column] = imputed_with_median[column].median()
    print("Missing values")
    print(imputed_with_median.isna().sum())

    return data_as_is, missing_data_as_is, nan_idxs, imputed_with_median


# def prepare_missing_vals_dataset():
#     column_names = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
#                     "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-surface-above-ring",
#                     "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
#                     "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
#     df = pd.read_csv("missing data/mushrooms_data_missing.txt", names=column_names)
#     df.describe().to_csv("missing data/described_data_missing.csv")  # note there's only one veil-type
#     # the data is categorical, do a label encoder will be used to make it ordinal
#     df = df.astype('category')
#     df = df.replace('-', np.nan)  # mark missing values with None
#     print("Missing values")
#     print(df.isna().sum())
#     df = df.apply(lambda series: pd.Series(
#         LabelEncoder().fit_transform(series[series.notnull()]),
#         index=series[series.notnull()].index
#     ))
#     print("Missing values")
#     print(df.isna().sum())
#     print("Impute missing values with the median value and check again for missing values:")
#     # impute with median
#     # for column in df.columns:
#     #     df.loc[df[column].isnull(), column] = df[column].median()
#     # print(df.isna().sum())
#     # print("There are no missing values now")
#     odor = df["odor"]
#     # since we want to predict the mushrooms' odor it's needed to drop this feature from the df
#     X = df.drop(["odor"], axis=1)
#     print(X.head())
#     # save the ordinal data
#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(X)
#     pd.DataFrame(X).to_csv("missing data/ordinal_mushrooms_missing_data.csv")
#     print("data set is:")
#     print(X)
#     return X, odor


# def get_missing_dataset():
#     X, _ = prepare_missing_vals_dataset()
#     return X


def get_impute_options(data_as_is, col):
    return np.unique(data_as_is.iloc[:, col])


if __name__ == '__main__':
    # data_as_is, missing_data_as_is, nan_idxs, imputed_with_median = prepare_all()
    # real_labels = data_as_is["odor"]
    # # data_as_is = data_as_is.drop("odor", axis=1)
    # data = data_as_is
    # imputed_data = imputed_with_median.copy().values
    # model = IsolationForest()
    # model.fit(data)
    # print(imputed_data)
    # for i, j in nan_idxs:
    #     row = imputed_data[i]
    #     # print(j)
    #     impute_options = get_impute_options(data_as_is, j)
    #     copy = [e for e in row]
    #     scores = {}
    #     # print(len(row))
    #     for option in impute_options:
    #         copy[j] = option
    #         # print(copy)
    #         scores[option] = model.score_samples([copy])
    #     best_option = max(scores, key=scores.get)
    #     row[j] = best_option
    #     imputed_data[i] = row
    # column_names = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
    #                 "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-surface-above-ring",
    #                 "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
    #                 "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
    #
    # df = pd.DataFrame(imputed_data, columns=column_names)
    # odor = df["odor"]
    # # since we want to predict the mushrooms' odor it's needed to drop this feature from the df
    # X = df.drop(["odor"], axis=1)
    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)
    # pd.DataFrame(X).to_csv("missing data/imputed_data.csv")
    # real_data = pd.DataFrame(data, columns=column_names)
    # real_data = real_data.drop(["odor"], axis=1)
    # real_data = scaler.transform(real_data)
    # real_data_list = []
    # real_labels_list = []
    # for i in range(20):
    #     real_data_i, _, real_labels_i, _ = train_test_split(real_data, real_labels, train_size=0.48)
    #     real_data_list.append(real_data_i)
    #     real_labels_list.append(real_labels_i)
    # print("Isolation Forest")
    # for i in range(20):
    #     clf = SVC(C=0.8)
    #     clf.fit(real_data_list[i], real_labels_list[i])
    #     labels = clf.predict(X)
    #     print(f1_score(odor, labels, average='micro'))
    #     # print(clf.score(X, odor))
    #
    # print("Imputing with median")
    # df = imputed_with_median
    # odor = df["odor"]
    # X = df.drop(["odor"], axis=1)
    # X = scaler.transform(X)
    # for i in range(20):
    #     clf = SVC(C=0.8)
    #     clf.fit(real_data_list[i], real_labels_list[i])
    #     labels = clf.predict(X)
    #     print(f1_score(odor, labels, average='micro'))
    #     # print(clf.score(X, odor))
    #
    # print("Iterative Imputer")
    # imp_mean = IterativeImputer()
    # imp_mean.fit(data_as_is)
    # imputed_data = imp_mean.transform(missing_data_as_is)
    # df = pd.DataFrame(imputed_data, columns=column_names)
    # odor = df["odor"]
    # odor = odor.astype('int')
    # # print(odor)
    # # print(odor.dtype)
    # # since we want to predict the mushrooms' odor it's needed to drop this feature from the df
    # X = df.drop(["odor"], axis=1)
    # X = scaler.transform(X)
    # for i in range(20):
    #     clf = SVC(C=0.8)
    #     clf.fit(real_data_list[i], real_labels_list[i])
    #     labels = clf.predict(X)
    #     print(f1_score(odor, labels, average='micro'))
    #     # print(clf.score(X, odor))
    #
    # print("KNN Imputer")
    # imp_mean = KNNImputer()
    # imp_mean.fit(data_as_is)
    # imputed_data = imp_mean.transform(missing_data_as_is)
    # df = pd.DataFrame(imputed_data, columns=column_names)
    # odor = df["odor"]
    # odor = odor.astype('int')
    # # since we want to predict the mushrooms' odor it's needed to drop this feature from the df
    # X = df.drop(["odor"], axis=1)
    # X = scaler.transform(X)
    # for i in range(20):
    #     clf = SVC(C=0.8)
    #     clf.fit(real_data_list[i], real_labels_list[i])
    #     labels = clf.predict(X)
    #     print(f1_score(odor, labels, average='micro'))
    #     # print(clf.score(X, odor))

    # scores = {'Isolation Forest': [0.5138461538461538, 0.5046153846153846, 0.5046153846153846, 0.5064615384615384,
    #                                0.5095384615384615, 0.5132307692307693, 0.5027692307692307, 0.5132307692307693,
    #                                0.512, 0.5021538461538462, 0.5046153846153846, 0.5113846153846154,
    #                                0.5107692307692308, 0.5175384615384615, 0.4990769230769231, 0.5175384615384615,
    #                                0.5212307692307693, 0.5101538461538462, 0.5107692307692308, 0.5101538461538462],
    #           'Imputing with median': [0.5206153846153846, 0.5126153846153846, 0.5138461538461538, 0.5138461538461538,
    #                                    0.5150769230769231, 0.5193846153846153, 0.5163076923076924, 0.5230769230769231,
    #                                    0.5187692307692308, 0.512, 0.5052307692307693, 0.52, 0.5156923076923077,
    #                                    0.5212307692307693, 0.5095384615384615, 0.5255384615384615, 0.5292307692307693,
    #                                    0.5212307692307693, 0.5156923076923077, 0.5163076923076924],
    #           'Iterative Imputer': [0.5052307692307693, 0.4916923076923077, 0.4929230769230769, 0.4929230769230769,
    #                                 0.4990769230769231, 0.5033846153846154, 0.5003076923076923, 0.5064615384615384,
    #                                 0.5015384615384615, 0.49846153846153846, 0.49415384615384617, 0.5033846153846154,
    #                                 0.5009230769230769, 0.5083076923076923, 0.49538461538461537, 0.5083076923076923,
    #                                 0.5101538461538462, 0.5046153846153846, 0.5027692307692307, 0.5033846153846154],
    #           'KNN Imputer': [0.5132307692307693, 0.5046153846153846, 0.5058461538461538, 0.5064615384615384,
    #                           0.5083076923076923, 0.5132307692307693, 0.5089230769230769, 0.5163076923076924,
    #                           0.5101538461538462, 0.5089230769230769, 0.5046153846153846, 0.5138461538461538,
    #                           0.5169230769230769, 0.5212307692307693, 0.504, 0.5187692307692308, 0.5218461538461538,
    #                           0.5126153846153846, 0.5132307692307693, 0.5107692307692308]}
    # for key1 in scores:
    #     for key2 in scores:
    #         if key1 != key2:
    #             p_val = clustering.u_test(scores[key1], scores[key2])
    #             if p_val < 0.05:
    #                 print(key1 + "is better than " + key2 + "with p-value = " + str(p_val) + "<<0.05")

    # plot best
    print("Imputing with median")
    data_as_is, missing_data_as_is, nan_idxs, imputed_with_median = prepare_all()
    real_labels = data_as_is["odor"]
    data = data_as_is
    imputed_data = imputed_with_median.copy().values

    column_names = ["class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
                    "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-surface-above-ring",
                    "stalk-surface-below-ring", "stalk-color-above-ring", "stalk-color-below-ring", "veil-type",
                    "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]

    df = pd.DataFrame(imputed_data, columns=column_names)
    odor = df["odor"]
    # since we want to predict the mushrooms' odor it's needed to drop this feature from the df
    X = df.drop(["odor"], axis=1)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    pd.DataFrame(X).to_csv("missing data/imputed_data.csv")
    real_data = pd.DataFrame(data, columns=column_names)
    real_data = real_data.drop(["odor"], axis=1)
    real_data = scaler.transform(real_data)
    real_data, _, real_labels, _ = train_test_split(real_data, real_labels, train_size=0.48)
    clf = SVC(C=0.8)
    clf.fit(real_data, real_labels)
    labels = clf.predict(X)
    print(f1_score(odor, labels, average='micro'))

    # plot
    array = confusion_matrix(odor, labels)
    df_cm = pd.DataFrame(array)

    # Create a subplot with 1 row and 3 columns
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    fig.set_size_inches(20, 7)

    sns.heatmap(df_cm, annot=True, fmt="d", ax=ax1)

    method = 'CMDS'
    reducedX = dimension_reduction.apply_dimension_reduction_method(X, method, plot=False)

    ax2.scatter3D(reducedX[:, 0], reducedX[:, 1], reducedX[:, 2], marker='.', s=30, lw=0, alpha=0.7, edgecolor='k',
                  c=labels)
    ax2.set_title("Visualization of the classified data.", fontdict={'fontsize': 12})
    ax2.set_xlabel("The 1st feature")
    ax2.set_ylabel("The 2nd feature")
    ax2.set_zlabel("The 3rd feature")

    ax3.scatter3D(reducedX[:, 0], reducedX[:, 1], reducedX[:, 2], marker='.', s=30, lw=0, alpha=0.7, c=odor)
    ax3.set_title("Visualization of the real labels of the data.", fontdict={'fontsize': 12})
    ax3.set_xlabel("The 1st feature")
    ax3.set_ylabel("The 2nd feature")
    ax3.set_zlabel("The 3rd feature")

    plt.suptitle(("Classifying the data with the missing values using SVM" +
                  " with RBF kernel Box Constraint = %.1f" % 0.8 + " visualized by CMDS"),
                 fontsize=14, fontweight='bold')

    plt.show()
