#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


def init_classifier_knn(knn_param):
    

    return KNeighborsClassifier(
            n_neighbors=knn_param["k_classifier"], 
            n_jobs=-1, 
            metric=knn_param["distance_method"])

def init_classifier_svm(svm_param):
    
    models = (svm.SVC(kernel=svm_param["kernel"][0], C=svm_param["C_linear"]),
          svm.LinearSVC(C=svm_param["C_linear2"]),
          svm.SVC(kernel=svm_param["kernel"][1], gamma=svm_param["gamma"], C=svm_param["C_rbf"]),
          svm.SVC(kernel=svm_param["kernel"][2], degree=svm_param["degree"], C=svm_param["C_poly"]))
    
    return models
