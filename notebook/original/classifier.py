#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler


def init_classifier_knn(knn_param):

    return KNeighborsClassifier(
            n_neighbors=knn_param["k_classifier"], 
            n_jobs=-1, 
            metric=knn_param["distance_method"])

def init_classifier_svm(svm_param):
    
    models = (
            (svm.SVC(kernel=svm_param["kernel"][0], C=svm_param["C_linear"]), "linear1"),
            (svm.LinearSVC(C=svm_param["C_linear2"]), "linear2"),
            (svm.SVC(kernel=svm_param["kernel"][1], gamma=svm_param["gamma"], C=svm_param["C_rbf"]), "rbf"),
            (svm.SVC(kernel=svm_param["kernel"][2], degree=svm_param["degree"], C=svm_param["C_poly"]), "poly"),
            (svm.SVC(kernel=svm_param["kernel"][3], C=svm_param["C_inter"]), "inter")
            )
        
    return models


def histogram_intersection(set1, set2):
    set1 = abs(set1)
    set2 = abs(set2)

    inter = np.zeros((set1.shape[0], set2.shape[0]))

# Anem a fer que els histogrames estan en vertical
# Visual words: (1250, 128)
# Visual words test: 631, 128

# 1 2 3 4
# 4 3 6 7
# 0 0 0 0

# (3, 4)  4 bins

    for col in range(set1.shape[1]):
        col_1 = set1[:, col].reshape(-1, 1)
        col_2 = set2[:, col].reshape(-1, 1)

        inter += np.minimum(col_1, col_2.T)

    return inter


def compute_intersection_kernel(data_test, data_train):

    scld = StandardScaler().fit(data_train)
    scaled_train = scld.transform(data_train)
    scaled_test = scld.transform(data_test)

    return histogram_intersection(scaled_train, scaled_test)


def compute_regular_kernel(data_test, data_train):
    return data_test