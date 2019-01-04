#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

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
    

    inter = np.zeros( (len(set1),len(set2)) )
    
    for x, hist1 in enumerate(set1):
        for y,hist2 in enumerate(set2):
            minima = np.minimum(hist1, hist2)
            inter[x][y] = sum(minima)     
            
    return inter
