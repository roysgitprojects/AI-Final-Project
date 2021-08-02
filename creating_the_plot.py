from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.neural_network import MLPClassifier
# from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.metrics import f1_score
# import smogn
from scipy.optimize import minimize
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn import svm
import numpy as np
import seaborn as sns
from scipy.stats import norm
import random
import math
import matplotlib.collections
from itertools import combinations
import networkx as nx
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import time
from datetime import timedelta

np.set_printoptions(threshold=np.inf)


def read_dimension_reduction_results_to_lil(method):
    with open("dimension_reduction/" + method + '_3d.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    x_str, y_str, z_str = data
    x = [float(i) for i in x_str]
    y = [float(i) for i in y_str]
    z = [float(i) for i in z_str]
    return [x, y, z]


if __name__ == "__main__":
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

fig = plt.figure()
plt.suptitle("Confusion matrix analysis for SVM RBF classification on the original data visualized by CMDS",
             fontsize=20, fontweight='bold')
ax1 = fig.add_subplot(1, 3, 1)
fig.set_size_inches(18, 7)
# creating the confusing matrix on original data
X_train, X_test, Y_train, Y_test = train_test_split(X, odor, random_state=0, test_size=0.52)
clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=8 * 0.1, gamma='auto')).fit(X_train, Y_train)
Y_total_pred = clf.predict(X)
array = confusion_matrix(odor, Y_total_pred)
df_cm = pd.DataFrame(array, range(9), range(9))
# plt.figure(figsize = (10,7))
sn.set(font_scale=1.3)  # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt='d', linecolor='blue', ax=ax1)  # font size
plt.xlabel('Predicted class')
plt.ylabel('Actual class')

Y_total_pred = clf.predict(X)
method = 'CMDS'
list_of_xyz = read_dimension_reduction_results_to_lil(method)
[x, y, z] = list_of_xyz
points = (np.array([np.array(i) for i in list_of_xyz])).T
# print(points)
reducedX = points
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.scatter(reducedX[:, 0], reducedX[:, 1], reducedX[:, 2], lw=0, alpha=0.7, c=Y_total_pred, marker='.', s=30,
            edgecolor='k')
ax2.set_title("Visualization of the classification.")
ax2.set_xlabel("The 1st feature")
ax2.set_ylabel("The 2nd feature")
ax2.set_zlabel("The 3rd feature")

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.scatter(reducedX[:, 0], reducedX[:, 1], reducedX[:, 2], lw=0, alpha=0.7, c=odor, marker='.', s=30,
            edgecolor='k')
ax3.set_title("Visualization of the real labels.")
ax3.set_xlabel("The 1st feature")
ax3.set_ylabel("The 2nd feature")
ax3.set_zlabel("The 3rd feature")
plt.show()

fig = plt.figure()
plt.suptitle((
    "Analysis for Multi Layer Perceptron classification on the data after CMDS dimension reduction"),
    fontweight='bold')
ax1 = fig.add_subplot(1, 3, 1)
fig.set_size_inches(18, 7)
X_train, X_test, Y_train, Y_test = train_test_split(reducedX, odor, random_state=0, test_size=0.52)
clf = MLPClassifier(max_iter=1400).fit(X_train, Y_train)
Y = clf.predict(reducedX)
array = confusion_matrix(odor, Y)
df_cm = pd.DataFrame(array, range(9), range(9))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt='d', linecolor='blue', ax=ax1)  # font size
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
Y_total_pred = clf.predict(reducedX)

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.scatter(reducedX[:, 0], reducedX[:, 1], reducedX[:, 2], lw=0, alpha=0.7, c=Y_total_pred, marker='.', s=30,
            edgecolor='k')
ax2.set_title("Visualization of the classification.")
ax2.set_xlabel("The 1st feature")
ax2.set_ylabel("The 2nd feature")
ax2.set_zlabel("The 3rd feature")

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.scatter(reducedX[:, 0], reducedX[:, 1], reducedX[:, 2], lw=0, alpha=0.7, c=odor, marker='.', s=30,
            edgecolor='k')
ax3.set_title("Visualization of the real labels.")
ax3.set_xlabel("The 1st feature")
ax3.set_ylabel("The 2nd feature")
ax3.set_zlabel("The 3rd feature")

plt.show()
