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
    
    models = (svm.SVC(kernel='linear', C=svm_param["C"]),
          svm.LinearSVC(C=svm_param["C"]),
          svm.SVC(kernel='rbf', gamma=0.1, C=svm_param["C"]),
          svm.SVC(kernel='poly', degree=3, C=svm_param["C"]))
    
    return models
