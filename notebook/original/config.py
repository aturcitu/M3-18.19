# Default values from week 1
def variables():
    
    # Determines total number of kps in an given image (set composed of 256x256px img)
    sift_step_size = 20

    # List providing scale values to compute at each kp
    sift_scale = [20] 

    # Dense/Normal Sift 
    dense = True
    
    # Number of clusters in KMeans, size of codebook (words)
    k_codebook = 128
    
    type_classifier = "KNN"

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
    
    # Number of pyramid levels used
    pyramid_level = 0
    # CrosValidation division kfold
    number_splits = 3
    # Intersection Kernel for SVN 
    intersection = False
    # Distance method used in order to normalize bincounts for the BoW
    norm_method = "L2"

    return (sift_step_size, sift_scale, dense, k_codebook, type_classifier, 
            svm_dict, knn_dict, pyramid_level, number_splits, intersection, norm_method)