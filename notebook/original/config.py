#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def variables():
    
    # Determines total number of kps in an given image (set composed of 256x256px img)
    sift_step_size = 20 #int(2**(5))
    # List providing scale values to compute at each kp
    sift_scale = [16] #[int(2**(3)),int(2**(4))]
    # Dense/Normal Sift 
    dense = True
    
    # Number of clusters in KMeans, size of codebook (words)
    k_codebook = 128
    
    type_classifier = "SVM"

    knn_dict =	{
      "k_classifier": 5,
      "distance_method": "euclidean",
    }
    
    svm_dict ={
        "kernel": ["linear", "rbf", "poly", "precomputed"],
        "C_linear": 0.1,
        "C_linear2": 0.1,
        "C_rbf": 1,
        "C_poly": 0.1,
        "C_inter": 1,
        "gamma": 0.001,
        "degree": 1,
    }
    
    pyramid_level = 0

    
    return (sift_step_size, sift_scale, dense, k_codebook, type_classifier, 
            svm_dict, knn_dict, pyramid_level)