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


def LogisticRegression():
    path_val_pred = "LogisticRegression/val_pred.csv"
    path_val = "LogisticRegression/val.csv"
    path_test_pred = "LogisticRegression/test_pred.csv"
    path_test = "LogisticRegression/test.csv"
    with open(path_val_pred, "w", newline="") as f_val_pred:
        with open(path_val, "w", newline="") as f_val:
            with open(path_test_pred, "w", newline="") as f_test_pred:
                with open(path_test, "w", newline="") as f_test:
                    for i in range(20):
                        X_train, X_test, Y_train, Y_test = train_test_split(X, odor, test_size=0.36)
                        # X_train,X_val,Y_train,Y_val = train_test_split(X_trainval,Y_trainval, test_size=0.20)
                        start = time.time()
                        # writer_val_pred = csv.writer(f_val_pred)
                        # writer_val = csv.writer(f_val)
                        writer_test_pred = csv.writer(f_test_pred)
                        writer_test = csv.writer(f_test)
                        clf = LogisticRegression().fit(X_train, Y_train)
                        Y_val_pred = clf.predict(X_val)
                        Y_test_pred = clf.predict(X_test)
                        # print(classification_report(Y_test,labels))
                        writer_val_pred.writerow(Y_val_pred)
                        writer_val.writerow(Y_val)
                        writer_test.writerow(Y_test)
                        end = time.time()
                        print("number " + str(i) + "time: " + str(
                            timedelta(seconds=end - start)))


def LogisticRegressionCV():
    path_val_pred = "LogisticRegressionCV/val_pred.csv"
    path_val = "LogisticRegressionCV/val.csv"
    path_test_pred = "LogisticRegressionCV/test_pred.csv"
    path_test = "LogisticRegressionCV/test.csv"
    with open(path_val_pred, "w", newline="") as f_val_pred:
        with open(path_val, "w", newline="") as f_val:
            with open(path_test_pred, "w", newline="") as f_test_pred:
                with open(path_test, "w", newline="") as f_test:
                    for i in range(20):
                        X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, odor, test_size=0.20)
                        X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.20)
                        start = time.time()
                        writer_val_pred = csv.writer(f_val_pred)
                        writer_val = csv.writer(f_val)
                        writer_test_pred = csv.writer(f_test_pred)
                        writer_test = csv.writer(f_test)
                        clf = LogisticRegressionCV().fit(X_train, Y_train)
                        Y_val_pred = clf.predict(X_val)
                        Y_test_pred = clf.predict(X_test)
                        # print(classification_report(Y_test,labels))
                        # writer_val_pred.writerow(Y_val_pred)
                        # writer_val.writerow(Y_val)
                        writer_test_pred.writerow(Y_test_pred)
                        writer_test.writerow(Y_test)
                        end = time.time()
                        print("number " + str(i) + "time: " + str(
                            timedelta(seconds=end - start)))


def svm_poly():
    for poly_degree in range(1, 11, 1):
        for c in range(1, 11, 1):
            path_val_pred = "svm_poly/val_pred/poly_degree_" + str(poly_degree) + "/c_value" + str(c) + ".csv"
            path_val = "svm_poly/val/poly_degree_" + str(poly_degree) + "/c_value" + str(c) + ".csv"
            path_test_pred = "svm_poly/test_pred/poly_degree_" + str(poly_degree) + "/c_value" + str(c) + ".csv"
            path_test = "svm_poly/test/poly_degree_" + str(poly_degree) + "/c_value" + str(c) + ".csv"
            path_f1_results_test = "svm_poly/results_f1_test/poly_degree_" + str(poly_degree) + "/c_value" + str(
                c) + ".csv"
            path_recall_results_test = "svm_poly/results_recall_test/poly_degree_" + str(
                poly_degree) + "/c_value" + str(c) + ".csv"
            path_f1_results_val = "svm_poly/results_f1_val/poly_degree_" + str(poly_degree) + "/c_value" + str(
                c) + ".csv"
            path_recall_results_val = "svm_poly/results_recall_val/poly_degree_" + str(poly_degree) + "/c_value" + str(
                c) + ".csv"
            with open(path_val_pred, "w", newline="") as f_val_pred:
                with open(path_val, "w", newline="") as f_val:
                    with open(path_test_pred, "w", newline="") as f_test_pred:
                        with open(path_test, "w", newline="") as f_test:
                            with open(path_f1_results_test, "w", newline="") as f_f1_results_test:
                                with open(path_f1_results_val, "w", newline="") as f_f1_results_val:
                                    with open(path_recall_results_test, "w", newline="") as f_recall_results_test:
                                        with open(path_recall_results_val, "w", newline="") as f_recall_results_val:
                                            for i in range(20):
                                                X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, odor,
                                                                                                          test_size=0.40)
                                                X_train, X_val, Y_train, Y_val = train_test_split(X_trainval,
                                                                                                  Y_trainval,
                                                                                                  test_size=0.20)
                                                start = time.time()
                                                writer_val_pred = csv.writer(f_val_pred)
                                                writer_val_test = csv.writer(f_val)
                                                writer_test_pred = csv.writer(f_test_pred)
                                                writer_test = csv.writer(f_test)
                                                writer_f1_results_test = csv.writer(f_f1_results_test)
                                                writer_recall_results_test = csv.writer(f_recall_results_test)
                                                writer_f1_results_val = csv.writer(f_f1_results_val)
                                                writer_recall_results_val = csv.writer(f_recall_results_val)
                                                clf = make_pipeline(StandardScaler(),
                                                                    SVC(kernel='poly', C=c * 0.1, gamma='auto')).fit(
                                                    X_train, Y_train)
                                                labels_val_pred = clf.predict(X_val)
                                                labels_test_pred = clf.predict(X_test)
                                                f1_results = f1_score(Y_val, labels_val_pred, average='micro')
                                                f1_score_val = [f1_results]
                                                recall_results = recall_score(Y_val, labels_val_pred, average='micro')
                                                recall_score_val = [recall_results]
                                                f1_results = f1_score(Y_test, labels_test_pred, average='micro')
                                                f1_score_test = [f1_results]
                                                recall_results = recall_score(Y_test, labels_test_pred, average='micro')
                                                recall_score_test = [recall_results]
                                                writer_val_pred.writerow(labels_val_pred)
                                                writer_val_test.writerow(Y_val)
                                                writer_test_pred.writerow(labels_test_pred)
                                                writer_test.writerow(Y_test)
                                                writer_f1_results_val.writerow(f1_score_val)
                                                writer_recall_results_val.writerow(recall_score_val)
                                                writer_f1_results_test.writerow(f1_score_test)
                                                writer_recall_results_test.writerow(recall_score_test)
                                                end = time.time()
                                                print("number " + str(i) + "time: " + str(
                                                    timedelta(seconds=end - start)))

                                            #             clf = make_pipeline(StandardScaler(), SVC(kernel='poly',degree = poly_degree, C = c*0.1,gamma='auto')).fit(X_train, Y_train)


def svm_linear():
    for c in range(1, 11, 1):
        path_val_pred = "svm_linear/val_pred/c_value" + str(c * 0.1) + ".csv"
        path_val = "svm_linear/val/c_value" + str(c * 0.1) + ".csv"
        path_test_pred = "svm_linear/test_pred/c_value" + str(c * 0.1) + ".csv"
        path_test = "svm_linear/test/c_value" + str(c * 0.1) + ".csv"
        path_f1_results_test = "svm_linear/results_f1_test/c_value" + str(c * 0.1) + ".csv"
        path_recall_results_test = "svm_linear/results_recall_test/c_value" + str(c * 0.1) + ".csv"
        path_f1_results_val = "svm_linear/results_f1_val/c_value" + str(c * 0.1) + ".csv"
        path_recall_results_val = "svm_linear/results_recall_val/c_value" + str(c * 0.1) + ".csv"
        with open(path_val_pred, "w", newline="") as f_val_pred:
            with open(path_val, "w", newline="") as f_val:
                with open(path_test_pred, "w", newline="") as f_test_pred:
                    with open(path_test, "w", newline="") as f_test:
                        with open(path_f1_results_test, "w", newline="") as f_f1_results_test:
                            with open(path_f1_results_val, "w", newline="") as f_f1_results_val:
                                with open(path_recall_results_test, "w", newline="") as f_recall_results_test:
                                    with open(path_recall_results_val, "w", newline="") as f_recall_results_val:
                                        for i in range(20):
                                            X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, odor,
                                                                                                      test_size=0.40)
                                            X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval,
                                                                                              test_size=0.20)
                                            start = time.time()
                                            writer_val_pred = csv.writer(f_val_pred)
                                            writer_val_test = csv.writer(f_val)
                                            writer_test_pred = csv.writer(f_test_pred)
                                            writer_test = csv.writer(f_test)
                                            writer_f1_results_test = csv.writer(f_f1_results_test)
                                            writer_recall_results_test = csv.writer(f_recall_results_test)
                                            writer_f1_results_val = csv.writer(f_f1_results_val)
                                            writer_recall_results_val = csv.writer(f_recall_results_val)
                                            clf = make_pipeline(StandardScaler(),
                                                                SVC(kernel='linear', C=c * 0.1, gamma='auto')).fit(
                                                X_train, Y_train)
                                            labels_val_pred = clf.predict(X_val)
                                            labels_test_pred = clf.predict(X_test)
                                            f1_results = f1_score(Y_val, labels_val_pred, average='micro')
                                            f1_score_val = [f1_results]
                                            recall_results = recall_score(Y_val, labels_val_pred, average='micro')
                                            recall_score_val = [recall_results]
                                            f1_results = f1_score(Y_test, labels_test_pred, average='micro')
                                            f1_score_test = [f1_results]
                                            recall_results = recall_score(Y_test, labels_test_pred, average='micro')
                                            recall_score_test = [recall_results]
                                            writer_val_pred.writerow(labels_val_pred)
                                            writer_val_test.writerow(Y_val)
                                            writer_test_pred.writerow(labels_test_pred)
                                            writer_test.writerow(Y_test)
                                            writer_f1_results_val.writerow(f1_score_val)
                                            writer_recall_results_val.writerow(recall_score_val)
                                            writer_f1_results_test.writerow(f1_score_test)
                                            writer_recall_results_test.writerow(recall_score_test)
                                            end = time.time()
                                            print("number " + str(i) + "time: " + str(
                                                timedelta(seconds=end - start)))


def svm_rbf():
    for c in range(1, 11, 1):
        path_val_pred = "svm_rbf/val_pred/c_value" + str(c * 0.1) + ".csv"
        path_val = "svm_rbf/val/c_value" + str(c * 0.1) + ".csv"
        path_test_pred = "svm_rbf/test_pred/c_value" + str(c * 0.1) + ".csv"
        path_test = "svm_rbf/test/c_value" + str(c * 0.1) + ".csv"
        path_f1_results_test = "svm_rbf/results_f1_test/c_value" + str(c * 0.1) + ".csv"
        path_recall_results_test = "svm_rbf/results_recall_test/c_value" + str(c * 0.1) + ".csv"
        path_f1_results_val = "svm_rbf/results_f1_val/c_value" + str(c * 0.1) + ".csv"
        path_recall_results_val = "svm_rbf/results_recall_val/c_value" + str(c * 0.1) + ".csv"
        with open(path_val_pred, "w", newline="") as f_val_pred:
            with open(path_val, "w", newline="") as f_val:
                with open(path_test_pred, "w", newline="") as f_test_pred:
                    with open(path_test, "w", newline="") as f_test:
                        with open(path_f1_results_test, "w", newline="") as f_f1_results_test:
                            with open(path_f1_results_val, "w", newline="") as f_f1_results_val:
                                with open(path_recall_results_test, "w", newline="") as f_recall_results_test:
                                    with open(path_recall_results_val, "w", newline="") as f_recall_results_val:
                                        for i in range(20):
                                            X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, odor,
                                                                                                      test_size=0.40)
                                            X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval,
                                                                                              test_size=0.20)
                                            start = time.time()
                                            writer_val_pred = csv.writer(f_val_pred)
                                            writer_val_test = csv.writer(f_val)
                                            writer_test_pred = csv.writer(f_test_pred)
                                            writer_test = csv.writer(f_test)
                                            writer_f1_results_test = csv.writer(f_f1_results_test)
                                            writer_recall_results_test = csv.writer(f_recall_results_test)
                                            writer_f1_results_val = csv.writer(f_f1_results_val)
                                            writer_recall_results_val = csv.writer(f_recall_results_val)
                                            clf = make_pipeline(StandardScaler(),
                                                                SVC(kernel='rbf', C=c * 0.1, gamma='auto')).fit(X_train,
                                                                                                                Y_train)
                                            labels_val_pred = clf.predict(X_val)
                                            labels_test_pred = clf.predict(X_test)
                                            f1_results = f1_score(Y_val, labels_val_pred, average='micro')
                                            f1_score_val = [f1_results]
                                            recall_results = recall_score(Y_val, labels_val_pred, average='micro')
                                            recall_score_val = [recall_results]
                                            f1_results = f1_score(Y_test, labels_test_pred, average='micro')
                                            f1_score_test = [f1_results]
                                            recall_results = recall_score(Y_test, labels_test_pred, average='micro')
                                            recall_score_test = [recall_results]
                                            writer_val_pred.writerow(labels_val_pred)
                                            writer_val_test.writerow(Y_val)
                                            writer_test_pred.writerow(labels_test_pred)
                                            writer_test.writerow(Y_test)
                                            writer_f1_results_val.writerow(f1_score_val)
                                            writer_recall_results_val.writerow(recall_score_val)
                                            writer_f1_results_test.writerow(f1_score_test)
                                            writer_recall_results_test.writerow(recall_score_test)
                                            end = time.time()
                                            print("number " + str(i) + "time: " + str(
                                                timedelta(seconds=end - start)))


def svm_sigmoid():
    for c in range(1, 11, 1):
        path_val_pred = "svm_sigmoid/val_pred/c_value" + str(c * 0.1) + ".csv"
        path_val = "svm_sigmoid/val/c_value" + str(c * 0.1) + ".csv"
        path_test_pred = "svm_sigmoid/test_pred/c_value" + str(c * 0.1) + ".csv"
        path_test = "svm_sigmoid/test/c_value" + str(c * 0.1) + ".csv"
        path_f1_results_test = "svm_sigmoid/results_f1_test/c_value" + str(c * 0.1) + ".csv"
        path_recall_results_test = "svm_sigmoid/results_recall_test/c_value" + str(c * 0.1) + ".csv"
        path_f1_results_val = "svm_sigmoid/results_f1_val/c_value" + str(c * 0.1) + ".csv"
        path_recall_results_val = "svm_sigmoid/results_recall_val/c_value" + str(c * 0.1) + ".csv"
        with open(path_val_pred, "w", newline="") as f_val_pred:
            with open(path_val, "w", newline="") as f_val:
                with open(path_test_pred, "w", newline="") as f_test_pred:
                    with open(path_test, "w", newline="") as f_test:
                        with open(path_f1_results_test, "w", newline="") as f_f1_results_test:
                            with open(path_f1_results_val, "w", newline="") as f_f1_results_val:
                                with open(path_recall_results_test, "w", newline="") as f_recall_results_test:
                                    with open(path_recall_results_val, "w", newline="") as f_recall_results_val:
                                        for i in range(20):
                                            X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, odor,
                                                                                                      test_size=0.40)
                                            X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval,
                                                                                              test_size=0.20)
                                            start = time.time()
                                            writer_val_pred = csv.writer(f_val_pred)
                                            writer_val_test = csv.writer(f_val)
                                            writer_test_pred = csv.writer(f_test_pred)
                                            writer_test = csv.writer(f_test)
                                            writer_f1_results_test = csv.writer(f_f1_results_test)
                                            writer_recall_results_test = csv.writer(f_recall_results_test)
                                            writer_f1_results_val = csv.writer(f_f1_results_val)
                                            writer_recall_results_val = csv.writer(f_recall_results_val)
                                            clf = make_pipeline(StandardScaler(),
                                                                SVC(kernel='sigmoid', C=c * 0.1, gamma='auto')).fit(
                                                X_train, Y_train)
                                            labels_val_pred = clf.predict(X_val)
                                            labels_test_pred = clf.predict(X_test)
                                            f1_results = f1_score(Y_val, labels_val_pred, average='micro')
                                            f1_score_val = [f1_results]
                                            recall_results = recall_score(Y_val, labels_val_pred, average='micro')
                                            recall_score_val = [recall_results]
                                            f1_results = f1_score(Y_test, labels_test_pred, average='micro')
                                            f1_score_test = [f1_results]
                                            recall_results = recall_score(Y_test, labels_test_pred, average='micro')
                                            recall_score_test = [recall_results]
                                            writer_val_pred.writerow(labels_val_pred)
                                            writer_val_test.writerow(Y_val)
                                            writer_test_pred.writerow(labels_test_pred)
                                            writer_test.writerow(Y_test)
                                            writer_f1_results_val.writerow(f1_score_val)
                                            writer_recall_results_val.writerow(recall_score_val)
                                            writer_f1_results_test.writerow(f1_score_test)
                                            writer_recall_results_test.writerow(recall_score_test)
                                            end = time.time()
                                            print("number " + str(i) + "time: " + str(
                                                timedelta(seconds=end - start)))


def XGBoost():
    path_val_pred = "XGBoost/val_pred.csv"
    path_val_test = "XGBoost/val_test.csv"
    path_test_pred = "XGBoost/test_pred.csv"
    path_test = "XGBoost/test.csv"
    path_f1_results_test = "XGBoost/results_f1_test.csv"
    path_f1_results_val = "XGBoost/results_f1_val.csv"
    path_recall_results_test = "XGBoost/results_recall_test.csv"
    path_recall_results_val = "XGBoost/results_recall_val.csv"
    with open(path_val_pred, "w", newline="") as f_val_pred:
        with open(path_val_test, "w", newline="") as f_val_test:
            with open(path_test_pred, "w", newline="") as f_test_pred:
                with open(path_test, "w", newline="") as f_test:
                    with open(path_f1_results_test, "w", newline="") as f_f1_results_test:
                        with open(path_f1_results_val, "w", newline="") as f_f1_results_val:
                            with open(path_recall_results_test, "w", newline="") as f_recall_results_test:
                                with open(path_recall_results_val, "w", newline="") as f_recall_results_val:
                                    for i in range(20):
                                        X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, odor,
                                                                                                  test_size=0.40)
                                        X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval,
                                                                                          test_size=0.20)
                                        start = time.time()
                                        writer_val_pred = csv.writer(f_val_pred)
                                        writer_val_test = csv.writer(f_val_test)
                                        writer_test_pred = csv.writer(f_test_pred)
                                        writer_test = csv.writer(f_test)
                                        writer_f1_results_test = csv.writer(f_f1_results_test)
                                        writer_recall_results_test = csv.writer(f_recall_results_test)
                                        writer_f1_results_val = csv.writer(f_f1_results_val)
                                        writer_recall_results_val = csv.writer(f_recall_results_val)
                                        clf = XGBClassifier().fit(X_train, Y_train)
                                        labels_val_pred = clf.predict(X_val)
                                        labels_test_pred = clf.predict(X_test)
                                        f1_results = f1_score(Y_val, labels_val_pred, average='micro')
                                        f1_score_val = [f1_results]
                                        recall_results = recall_score(Y_val, labels_val_pred, average='micro')
                                        recall_score_val = [recall_results]
                                        f1_results = f1_score(Y_test, labels_test_pred, average='micro')
                                        f1_score_test = [f1_results]
                                        recall_results = recall_score(Y_test, labels_test_pred, average='micro')
                                        recall_score_test = [recall_results]
                                        writer_val_pred.writerow(labels_val_pred)
                                        writer_val_test.writerow(Y_val)
                                        writer_test_pred.writerow(labels_test_pred)
                                        writer_test.writerow(Y_test)
                                        writer_f1_results_val.writerow(f1_score_val)
                                        writer_recall_results_val.writerow(recall_score_val)
                                        writer_f1_results_test.writerow(f1_score_test)
                                        writer_recall_results_test.writerow(recall_score_test)
                                        end = time.time()
                                        print("number " + str(i) + "time: " + str(
                                            timedelta(seconds=end - start)))


def Gradient_Boosting():
    path_val_pred = "Gradient_Boosting/val_pred.csv"
    path_val_test = "Gradient_Boosting/val_test.csv"
    path_test_pred = "Gradient_Boosting/test_pred.csv"
    path_test = "Gradient_Boosting/test.csv"
    path_f1_results_test = "Gradient_Boosting/results_f1_test.csv"
    path_f1_results_val = "Gradient_Boosting/results_f1_val.csv"
    path_recall_results_test = "Gradient_Boosting/results_recall_test.csv"
    path_recall_results_val = "Gradient_Boosting/results_recall_val.csv"
    with open(path_val_pred, "w", newline="") as f_val_pred:
        with open(path_val_test, "w", newline="") as f_val_test:
            with open(path_test_pred, "w", newline="") as f_test_pred:
                with open(path_test, "w", newline="") as f_test:
                    with open(path_f1_results_test, "w", newline="") as f_f1_results_test:
                        with open(path_f1_results_val, "w", newline="") as f_f1_results_val:
                            with open(path_recall_results_test, "w", newline="") as f_recall_results_test:
                                with open(path_recall_results_val, "w", newline="") as f_recall_results_val:
                                    for i in range(20):
                                        X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, odor,
                                                                                                  test_size=0.40)
                                        X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval,
                                                                                          test_size=0.20)
                                        start = time.time()
                                        writer_val_pred = csv.writer(f_val_pred)
                                        writer_val_test = csv.writer(f_val_test)
                                        writer_test_pred = csv.writer(f_test_pred)
                                        writer_test = csv.writer(f_test)
                                        writer_f1_results_test = csv.writer(f_f1_results_test)
                                        writer_recall_results_test = csv.writer(f_recall_results_test)
                                        writer_f1_results_val = csv.writer(f_f1_results_val)
                                        writer_recall_results_val = csv.writer(f_recall_results_val)
                                        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.04).fit(
                                            X_train, Y_train)
                                        labels_val_pred = clf.predict(X_val)
                                        labels_test_pred = clf.predict(X_test)
                                        f1_results = f1_score(Y_val, labels_val_pred, average='micro')
                                        f1_score_val = [f1_results]
                                        recall_results = recall_score(Y_val, labels_val_pred, average='micro')
                                        recall_score_val = [recall_results]
                                        f1_results = f1_score(Y_test, labels_test_pred, average='micro')
                                        f1_score_test = [f1_results]
                                        recall_results = recall_score(Y_test, labels_test_pred, average='micro')
                                        recall_score_test = [recall_results]
                                        writer_val_pred.writerow(labels_val_pred)
                                        writer_val_test.writerow(Y_val)
                                        writer_test_pred.writerow(labels_test_pred)
                                        writer_test.writerow(Y_test)
                                        writer_f1_results_val.writerow(f1_score_val)
                                        writer_recall_results_val.writerow(recall_score_val)
                                        writer_f1_results_test.writerow(f1_score_test)
                                        writer_recall_results_test.writerow(recall_score_test)
                                        end = time.time()
                                        print("number " + str(i) + "time: " + str(
                                            timedelta(seconds=end - start)))


def Random_Forest():
    for tree_length in range(2, 9, 1):
        path_val_pred = "Random_Forest/gini/val_pred/tree_length" + str(tree_length) + ".csv"
        path_val_test = "Random_Forest/gini/val_test/tree_length" + str(tree_length) + ".csv"
        path_test_pred = "Random_Forest/gini/test_pred/tree_length" + str(tree_length) + ".csv"
        path_test = "Random_Forest/gini/test/tree_length" + str(tree_length) + ".csv"
        path_f1_results_test = "Random_Forest/gini/f1_results_test/tree_length" + str(tree_length) + ".csv"
        path_recall_results_test = "Random_Forest/gini/recall_results_test/tree_length" + str(tree_length) + ".csv"
        path_f1_results_val = "Random_Forest/gini/f1_results_val/tree_length" + str(tree_length) + ".csv"
        path_recall_results_val = "Random_Forest/gini/recall_results_val/tree_length" + str(tree_length) + ".csv"
        with open(path_val_pred, "w", newline="") as f_val_pred:
            with open(path_val_test, "w", newline="") as f_val_test:
                with open(path_test_pred, "w", newline="") as f_test_pred:
                    with open(path_test, "w", newline="") as f_test:
                        with open(path_f1_results_test, "w", newline="") as f_f1_results_test:
                            with open(path_f1_results_val, "w", newline="") as f_f1_results_val:
                                with open(path_recall_results_test, "w", newline="") as f_recall_results_test:
                                    with open(path_recall_results_val, "w", newline="") as f_recall_results_val:
                                        for i in range(20):
                                            X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, odor,
                                                                                                      test_size=0.4)
                                            X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval,
                                                                                              test_size=0.20)
                                            start = time.time()
                                            writer_val_pred = csv.writer(f_val_pred)
                                            writer_val_test = csv.writer(f_val_test)
                                            writer_test_pred = csv.writer(f_test_pred)
                                            writer_test = csv.writer(f_test)
                                            writer_f1_results_test = csv.writer(f_f1_results_test)
                                            writer_recall_results_test = csv.writer(f_recall_results_test)
                                            writer_f1_results_val = csv.writer(f_f1_results_val)
                                            writer_recall_results_val = csv.writer(f_recall_results_val)
                                            clf = RandomForestClassifier(n_estimators=1000, max_depth=tree_length,
                                                                         random_state=42, criterion='gini').fit(X_train,
                                                                                                                Y_train)
                                            labels_val_pred = clf.predict(X_val)
                                            labels_test_pred = clf.predict(X_test)
                                            f1_results = f1_score(Y_val, labels_val_pred, average='micro')
                                            f1_score_val = [f1_results]
                                            recall_results = recall_score(Y_val, labels_val_pred, average='micro')
                                            recall_score_val = [recall_results]
                                            f1_results = f1_score(Y_test, labels_test_pred, average='micro')
                                            f1_score_test = [f1_results]
                                            recall_results = recall_score(Y_test, labels_test_pred, average='micro')
                                            recall_score_test = [recall_results]
                                            writer_val_pred.writerow(labels_val_pred)
                                            writer_val_test.writerow(Y_val)
                                            writer_test_pred.writerow(labels_test_pred)
                                            writer_test.writerow(Y_test)
                                            writer_f1_results_val.writerow(f1_score_val)
                                            writer_recall_results_val.writerow(recall_score_val)
                                            writer_f1_results_test.writerow(f1_score_test)
                                            writer_recall_results_test.writerow(recall_score_test)
                                            end = time.time()
                                            print("number " + str(i) + "time: " + str(
                                                timedelta(seconds=end - start)))

    for tree_length in range(2, 9, 1):
        path_val_pred = "Random_Forest/entropy/val_pred/tree_length" + str(tree_length) + ".csv"
        path_val_test = "Random_Forest/entropy/val_test/tree_length" + str(tree_length) + ".csv"
        path_test_pred = "Random_Forest/entropy/test_pred/tree_length" + str(tree_length) + ".csv"
        path_test = "Random_Forest/entropy/test/tree_length" + str(tree_length) + ".csv"
        path_f1_results_test = "Random_Forest/entropy/f1_results_test/tree_length" + str(tree_length) + ".csv"
        path_recall_results_test = "Random_Forest/entropy/recall_results_test/tree_length" + str(tree_length) + ".csv"
        path_f1_results_val = "Random_Forest/entropy/f1_results_val/tree_length" + str(tree_length) + ".csv"
        path_recall_results_val = "Random_Forest/entropy/recall_results_val/tree_length" + str(tree_length) + ".csv"
        with open(path_val_pred, "w", newline="") as f_val_pred:
            with open(path_val_test, "w", newline="") as f_val_test:
                with open(path_test_pred, "w", newline="") as f_test_pred:
                    with open(path_test, "w", newline="") as f_test:
                        with open(path_f1_results_test, "w", newline="") as f_f1_results_test:
                            with open(path_f1_results_val, "w", newline="") as f_f1_results_val:
                                with open(path_recall_results_test, "w", newline="") as f_recall_results_test:
                                    with open(path_recall_results_val, "w", newline="") as f_recall_results_val:
                                        for i in range(20):
                                            X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, odor,
                                                                                                      test_size=0.4)
                                            X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval,
                                                                                              test_size=0.20)
                                            start = time.time()
                                            writer_val_pred = csv.writer(f_val_pred)
                                            writer_val_test = csv.writer(f_val_test)
                                            writer_test_pred = csv.writer(f_test_pred)
                                            writer_test = csv.writer(f_test)
                                            writer_f1_results_test = csv.writer(f_f1_results_test)
                                            writer_recall_results_test = csv.writer(f_recall_results_test)
                                            writer_f1_results_val = csv.writer(f_f1_results_val)
                                            writer_recall_results_val = csv.writer(f_recall_results_val)
                                            clf = RandomForestClassifier(n_estimators=1000, max_depth=tree_length,
                                                                         random_state=42, criterion='entropy').fit(
                                                X_train, Y_train)
                                            labels_val_pred = clf.predict(X_val)
                                            labels_test_pred = clf.predict(X_test)
                                            f1_results = f1_score(Y_val, labels_val_pred, average='micro')
                                            f1_score_val = [f1_results]
                                            recall_results = recall_score(Y_val, labels_val_pred, average='micro')
                                            recall_score_val = [recall_results]
                                            f1_results = f1_score(Y_test, labels_test_pred, average='micro')
                                            f1_score_test = [f1_results]
                                            recall_results = recall_score(Y_test, labels_test_pred, average='micro')
                                            recall_score_test = [recall_results]
                                            writer_val_pred.writerow(labels_val_pred)
                                            writer_val_test.writerow(Y_val)
                                            writer_test_pred.writerow(labels_test_pred)
                                            writer_test.writerow(Y_test)
                                            writer_f1_results_val.writerow(f1_score_val)
                                            writer_recall_results_val.writerow(recall_score_val)
                                            writer_f1_results_test.writerow(f1_score_test)
                                            writer_recall_results_test.writerow(recall_score_test)
                                            end = time.time()
                                            print("number " + str(i) + "time: " + str(
                                                timedelta(seconds=end - start)))


def Stochastic_gradient_descent():
    path_val_pred = "SGD/val_pred.csv"
    path_val_test = "SGD/val_test.csv"
    path_test_pred = "SGD/test_pred.csv"
    path_test = "SGD/test.csv"
    path_f1_results_test = "SGD/results_f1_test.csv"
    path_f1_results_val = "SGD/results_f1_val.csv"
    path_recall_results_test = "SGD/results_recall_test.csv"
    path_recall_results_val = "SGD/results_recall_val.csv"
    with open(path_val_pred, "w", newline="") as f_val_pred:
        with open(path_val_test, "w", newline="") as f_val_test:
            with open(path_test_pred, "w", newline="") as f_test_pred:
                with open(path_test, "w", newline="") as f_test:
                    with open(path_f1_results_test, "w", newline="") as f_f1_results_test:
                        with open(path_f1_results_val, "w", newline="") as f_f1_results_val:
                            with open(path_recall_results_test, "w", newline="") as f_recall_results_test:
                                with open(path_recall_results_val, "w", newline="") as f_recall_results_val:
                                    for i in range(20):
                                        X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, odor,
                                                                                                  test_size=0.40)
                                        X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval,
                                                                                          test_size=0.20)
                                        start = time.time()
                                        writer_val_pred = csv.writer(f_val_pred)
                                        writer_val_test = csv.writer(f_val_test)
                                        writer_test_pred = csv.writer(f_test_pred)
                                        writer_test = csv.writer(f_test)
                                        writer_f1_results_test = csv.writer(f_f1_results_test)
                                        writer_recall_results_test = csv.writer(f_recall_results_test)
                                        writer_f1_results_val = csv.writer(f_f1_results_val)
                                        writer_recall_results_val = csv.writer(f_recall_results_val)
                                        clf = make_pipeline(StandardScaler(),
                                                            SGDClassifier(max_iter=10000, tol=1e-4)).fit(X_train,
                                                                                                         Y_train)
                                        labels_val_pred = clf.predict(X_val)
                                        labels_test_pred = clf.predict(X_test)
                                        f1_results = f1_score(Y_val, labels_val_pred, average='micro')
                                        f1_score_val = [f1_results]
                                        recall_results = recall_score(Y_val, labels_val_pred, average='micro')
                                        recall_score_val = [recall_results]
                                        f1_results = f1_score(Y_test, labels_test_pred, average='micro')
                                        f1_score_test = [f1_results]
                                        recall_results = recall_score(Y_test, labels_test_pred, average='micro')
                                        recall_score_test = [recall_results]
                                        writer_val_pred.writerow(labels_val_pred)
                                        writer_val_test.writerow(Y_val)
                                        writer_test_pred.writerow(labels_test_pred)
                                        writer_test.writerow(Y_test)
                                        writer_f1_results_val.writerow(f1_score_val)
                                        writer_recall_results_val.writerow(recall_score_val)
                                        writer_f1_results_test.writerow(f1_score_test)
                                        writer_recall_results_test.writerow(recall_score_test)
                                        end = time.time()
                                        print("number " + str(i) + "time: " + str(
                                            timedelta(seconds=end - start)))


def Perceptron():
    path_val_pred = "Perceptron/val_pred.csv"
    path_val_test = "Perceptron/val_test.csv"
    path_test_pred = "Perceptron/test_pred.csv"
    path_test = "Perceptron/test.csv"
    with open(path_val_pred, "w", newline="") as f_val_pred:
        with open(path_val_test, "w", newline="") as f_val_test:
            with open(path_test_pred, "w", newline="") as f_test_pred:
                with open(path_test, "w", newline="") as f_test:
                    for i in range(20):
                        X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, odor, test_size=0.20)
                        X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.20)
                        start = time.time()
                        writer_val_pred = csv.writer(f_val_pred)
                        writer_val_test = csv.writer(f_val_test)
                        writer_test_pred = csv.writer(f_test_pred)
                        writer_test = csv.writer(f_test)
                        clf = Perceptron().fit(X_train, Y_train)
                        labels_val_pred = clf.predict(X_val)
                        labels_test_pred = clf.predict(X_test)
                        # print(classification_report(Y_test,labels))
                        writer_val_pred.writerow(labels_val_pred)
                        writer_val_test.writerow(Y_val)
                        writer_test_pred.writerow(labels_test_pred)
                        writer_test.writerow(Y_test)
                        end = time.time()
                        print("number " + str(i) + "time: " + str(
                            timedelta(seconds=end - start)))


def MLP():
    path_val_pred = "MLP/val_pred.csv"
    path_val_test = "MLP/val_test.csv"
    path_test_pred = "MLP/test_pred.csv"
    path_test = "MLP/test.csv"
    path_f1_results_test = "MLP/results_f1_test.csv"
    path_f1_results_val = "MLP/results_f1_val.csv"
    path_recall_results_test = "MLP/results_recall_test.csv"
    path_recall_results_val = "MLP/results_recall_val.csv"
    with open(path_val_pred, "w", newline="") as f_val_pred:
        with open(path_val_test, "w", newline="") as f_val_test:
            with open(path_test_pred, "w", newline="") as f_test_pred:
                with open(path_test, "w", newline="") as f_test:
                    with open(path_f1_results_test, "w", newline="") as f_f1_results_test:
                        with open(path_f1_results_val, "w", newline="") as f_f1_results_val:
                            with open(path_recall_results_test, "w", newline="") as f_recall_results_test:
                                with open(path_recall_results_val, "w", newline="") as f_recall_results_val:
                                    for i in range(20):
                                        X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, odor,
                                                                                                  test_size=0.40)
                                        X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval,
                                                                                          test_size=0.20)
                                        start = time.time()
                                        writer_val_pred = csv.writer(f_val_pred)
                                        writer_val_test = csv.writer(f_val_test)
                                        writer_test_pred = csv.writer(f_test_pred)
                                        writer_test = csv.writer(f_test)
                                        writer_f1_results_test = csv.writer(f_f1_results_test)
                                        writer_recall_results_test = csv.writer(f_recall_results_test)
                                        writer_f1_results_val = csv.writer(f_f1_results_val)
                                        writer_recall_results_val = csv.writer(f_recall_results_val)
                                        clf = MLPClassifier(max_iter=1400).fit(X_train, Y_train)
                                        labels_val_pred = clf.predict(X_val)
                                        labels_test_pred = clf.predict(X_test)
                                        f1_results = f1_score(Y_val, labels_val_pred, average='micro')
                                        f1_score_val = [f1_results]
                                        recall_results = recall_score(Y_val, labels_val_pred, average='micro')
                                        recall_score_val = [recall_results]
                                        f1_results = f1_score(Y_test, labels_test_pred, average='micro')
                                        f1_score_test = [f1_results]
                                        recall_results = recall_score(Y_test, labels_test_pred, average='micro')
                                        recall_score_test = [recall_results]
                                        writer_val_pred.writerow(labels_val_pred)
                                        writer_val_test.writerow(Y_val)
                                        writer_test_pred.writerow(labels_test_pred)
                                        writer_test.writerow(Y_test)
                                        writer_f1_results_val.writerow(f1_score_val)
                                        writer_recall_results_val.writerow(recall_score_val)
                                        writer_f1_results_test.writerow(f1_score_test)
                                        writer_recall_results_test.writerow(recall_score_test)
                                        end = time.time()
                                        print("accuracy is (%) =", accuracy_score(Y_test, labels_test_pred))
                                        print("number " + str(i) + "time: " + str(
                                            timedelta(seconds=end - start)))


'''
def acc():
classification_methods=[]
for method in classification_methods:
   path_val_pred = method+"/val_pred.csv"
   path_val_test = method+"/val_test.csv"
   path_test_pred = method+"/test_pred.csv"
   path_test = method+"/test.csv"
   with open(path_val_pred, "w", newline="") as f_val_pred:
     reader = csv.reader(f_val_pred)
     val_pred = list(reader)
   with open(path_val_test, "w", newline="") as f_val_test:
     reader = csv.reader(f_val_test)
     val_pred = list(reader)
   with open(path_test_pred, "w", newline="") as f_test_pred:
     reader = csv.reader(f_test_pred)
     val_pred = list(reader)
   with open(path_test, "w", newline="") as f_test:
     reader = csv.reader(f_test)
     val_pred = list(reader)
'''


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


def read_svm_scores(kernel, box_constraint=0):
    if box_constraint == 0:
        box_constraint = int(input(kernel + "Champion :"))
    if kernel == 'poly':
        path_f1_results_val = "svm_poly/results_f1_val/poly_degree_" + str(box_constraint) + "/c_value" + str(
            10) + ".csv"
    else:
        path_f1_results_val = "svm_" + kernel + "/results_f1_val/c_value" + str(box_constraint * 0.1) + ".csv"
    with open(path_f1_results_val, newline="") as f_f1_results_val:
        reader = csv.reader(f_f1_results_val)
        data = list(reader)
        data = [float(element) for sublist in data for element in sublist]
        # print(data)
    return data


def read_random_forest_scores(criterion, tree_length=0):
    if tree_length == 0:
        tree_length = int(input(criterion + "Champion :"))
    path_f1_results_val = "Random_Forest/" + criterion + "/f1_results_val/tree_length" + str(tree_length) + ".csv"
    with open(path_f1_results_val, newline="") as f_f1_results_val:
        reader = csv.reader(f_f1_results_val)
        data = list(reader)
        data = [float(element) for sublist in data for element in sublist]
    # print(data)
    return data


def read_normal_classification_score(classification_name, number):
    path_f1_results_val = classification_name + "/results_f1_val.csv"
    with open(path_f1_results_val, newline="") as f_f1_results_val:
        reader = csv.reader(f_f1_results_val)
        data = list(reader)
        data = [float(element) for sublist in data for element in sublist]
    # print(data)
    return data


def read_champion(classification_name):
    path_f1_results_val = "Champion's/" + classification_name + ".csv"
    with open(path_f1_results_val, newline="") as f_f1_results_val:
        reader = csv.reader(f_f1_results_val)
        data = list(reader)
        data = [float(element) for sublist in data for element in sublist]
    # print(data)
    return data


def calculate_u_test_svm(kernel):
    maxi = 0
    num = 0
    for i in range(1, 11):
        counter = 0
        for j in range(1, 11):
            if i != j:
                p_val = u_test(read_svm_scores(kernel, i), read_svm_scores(kernel, j))
                if (p_val) < 0.05:
                    counter = counter + 1
                    print(str(i) + " is better than " + str(j) + "with $p-value = " + str(p_val) + " << 0.05$")
        if counter >= maxi:
            maxi = counter
            num = i
    print(kernel, maxi, num)
    return (num)


def calculate_u_test_random_forest(kernel):
    maxi = 0
    num = 0
    for i in range(2, 9):
        counter = 0
        for j in range(2, 9):
            if i != j:
                p_val = u_test(read_random_forest_scores(kernel, i), read_random_forest_scores(kernel, j))
                if (p_val) < 0.05:
                    counter = counter + 1
                    print(str(i) + " is better than " + str(j) + "with $p-value = " + str(p_val) + " << 0.05$")
        if counter >= maxi:
            maxi = counter
            num = i
    print(kernel, maxi, num)
    return (num)


def calculate_champion(a=1, b=1, c=1, d=1, e=1, f=1):
    maxi = 0
    place = 0
    class_functions = [read_normal_classification_score, read_normal_classification_score,
                       read_normal_classification_score, read_normal_classification_score, read_svm_scores,
                       read_svm_scores, read_svm_scores, read_svm_scores, read_random_forest_scores,
                       read_random_forest_scores]
    names = ["MLP", "SGD", "Gradient_Boosting", "XGBoost", "linear", "rbf", "sigmoid", "poly", "gini", "entropy"]
    number = [0, 0, 0, 0, a, b, c, d, e, f]
    for i in range(len(names)):
        counter = 0
        for j in range(len(names)):
            if i != j:
                p_val = u_test(class_functions[i](names[i], number[i]), class_functions[j](names[j], number[j]))
                if (p_val) < 0.05:
                    counter = counter + 1
                    print(names[i] + " is better than " + names[j] + "with $p-value = " + str(p_val) + " << 0.05$")
            if counter >= maxi:
                maxi = counter
                place = i
    return (names[place], number[place])


def calculate_best_champion():
    names = ["rbf_normal", "poly_normal", "MLP_PCA", "rbf_ICA", "MLP_CMDS", "entropy_ae"]
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                p_val = u_test(read_champion(names[i]), read_champion(names[j]))
                if (p_val) < 0.05:
                    print(names[i] + " is better than " + names[j] + "with $p-value = " + str(p_val) + " << 0.05$")


def create_classification():
    Stochastic_gradient_descent()
    Random_Forest()
    Gradient_Boosting()
    svm_poly()
    svm_linear()
    svm_rbf()
    svm_sigmoid()
    MLP()
    XGBoost()


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

method = input(
    "enter the dimention reduction method ")  # The possible methods are ae(auto encoder), PCA, CMDS, ICA, normal(without dimension reduction)

if method != "normal":
    list_of_xyz = read_dimension_reduction_results_to_lil(method)
    [x, y, z] = list_of_xyz
    points = (np.array([np.array(i) for i in list_of_xyz])).T
    print(points)
    X = points

print(method)

create_classification()  # function that creates all the classification's

best = [1, 1, 1, 1, 1, 1]  # using statistical test to determine the best features
names = ['linear', 'rbf', 'sigmoid', 'poly']
for i in range(len(names)): best[i] = calculate_u_test_svm(names[i])
names = ['gini', 'entropy']
for i in range(len(names)): best[i + 4] = calculate_u_test_random_forest(names[i])

print(best)
print(calculate_champion(best[0], best[1], best[2], best[3], best[4], best[5]))
