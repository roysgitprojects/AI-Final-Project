from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
# import smogn
from scipy.optimize import minimize
from scipy import stats
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
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

np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

np.set_printoptions(threshold=np.inf)
import time
from datetime import timedelta
from scipy.stats import mannwhitneyu


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


def read_champion(classification_name):
    path_f1_results_val = "Champions/" + classification_name + ".csv"
    with open(path_f1_results_val, newline="") as f_f1_results_val:
        reader = csv.reader(f_f1_results_val)
        data = list(reader)
        data = [float(element) for sublist in data for element in sublist]
    # print(data)
    return data


def calculate_best_champion():
    names = ["rbf_normal", "poly_normal", "MLP_PCA", "rbf_ICA", "MLP_CMDS", "entropy_ae"]
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                p_val = u_test(read_champion(names[i]), read_champion(names[j]))
                if (p_val) < 0.05:
                    print(names[i] + " is better than " + names[j] + "with $p-value = " + str(p_val) + " << 0.05$")


if __name__ == "__main__":
    calculate_best_champion()
