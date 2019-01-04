import cv2
import numpy as np
import pickle
import time
from sklearn.preprocessing import normalize, StandardScaler

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import sys

from pyramid_words import pyramid_visual_word
from classifier import init_classifier_svm, init_classifier_knn, histogram_intersection
from visualization import plot_accuracy_vs_time, plot_confusion_matrix
from config import variables

def open_pkl(pkl_file):
    """
    This function opens pkl files providing file name on WD.
    """    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)    
    return data


def compute_detector(sift_step_size, sift_scale, n_features = 300):
    """
    Computes Sift detector object.
    Computes mesh of KPs using a custom step size and scale value(s).
    Points are shifted by sift_step_size/2 in order to avoid points on 
    image borders
    """    
    SIFTdetector = cv2.xfeatures2d.SIFT_create(nfeatures=n_features)
    
    if(isinstance(sift_scale, list) == False):
        sift_scale = [sift_scale]
    
    kpt = [cv2.KeyPoint(x, y, scale) for y in 
       range(int(sift_step_size/2)-1, 256-int(sift_step_size/2), sift_step_size) for x in 
       range(int(sift_step_size/2)-1, 256-int(sift_step_size/2), sift_step_size) for scale in sift_scale ]
    return (SIFTdetector, kpt)    

def compute_des_pyramid(dataset_desc, pyramid_level, img_px = 256):
    """
    Computes Pyramid divison of the kp descriptors dataset
    It uses KPs values to descriminate to which level each descriptor belongs
    """       
    div_level = int(2**(pyramid_level))
    pyramid_res = img_px/div_level
    pyramid_desc = []
    
    for image_desc in dataset_desc:
        im_pyramid_desc = []
        # axis 0 divisions
        for n in range(1,div_level+1):
            # axis 1 divisions
            for m in range(1,div_level+1):
                sub_desc = []
                
                for kp_desc, kp in zip(image_desc, kpt):               
                    x,y = kp.pt
                    # sub resolution area
                    if ((x>=(n-1)*pyramid_res and x<n*pyramid_res) and 
                        (y>=(m-1)*pyramid_res and y<m*pyramid_res)):
                        sub_desc.append(kp_desc)     
                        
                im_pyramid_desc.append(np.array(sub_desc, dtype='f'))
                
        pyramid_desc.append(im_pyramid_desc)  
        
    return pyramid_desc

def compute_BOW(train_images_filenames, train_labels, dense, SIFTdetector, kpt, 
               k_codebook, pyramid_level, norm_method):

    train_descriptors = []
    # Compute SIFT descriptors for whole DS 
    for filename in train_images_filenames:
        ima=cv2.imread(filename)
        gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)           
        if dense:
            (_, des) = SIFTdetector.compute(gray, kpt)        
        else:
            (_, des) = SIFTdetector.detectAndCompute(gray, None)  
        # Creates a list with all the descriptors  
        train_descriptors.append(des)
        
    # Descriptors are clustered with KMeans (whole image, e.g pyramid_level = 0)
    D =  np.vstack(train_descriptors) 
    codebook = MiniBatchKMeans(n_clusters=k_codebook, batch_size=k_codebook*20,
                               compute_labels=False, reassignment_ratio=10**-4, 
                               random_state=42)
    codebook.fit(D)
    
    # Pyramid Representation of n Levels
    pyramid_descriptors = []
    while(pyramid_level >= 0):
        pyramid_descriptors.append(compute_des_pyramid(train_descriptors, pyramid_level))
        pyramid_level -= 1
     
    # Create visual words with normalized bins for each image and subimage
    # After individually normalized, bins are concatenated for each image
    visual_words = pyramid_visual_word(pyramid_descriptors, codebook, k_codebook, train_descriptors)
    
    return codebook, visual_words

def test_BOW(test_images_filenames, test_labels, dense, SIFTdetector, kpt, k_codebook, pyramid_level, codebook, clf):
    
    # Create visual words for testing data    
    visual_words_test = np.zeros((len(test_images_filenames),k_codebook),
                               dtype=np.float32)
    test_descriptors = []    
    for filename in test_images_filenames:    
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)       
        if dense:
            (_, des) = SIFTdetector.compute(gray, kpt)
        else:
            (_, des) = SIFTdetector.detectAndCompute(gray,None)  

        test_descriptors.append(des)
   
    # Pyramid Representation of n Levels            
    pyramid_descriptors = []
    
    while(pyramid_level >= 0):
        pyramid_descriptors.append( compute_des_pyramid(test_descriptors, pyramid_level) )
        pyramid_level -= 1

    # Create visual words with normalized bins for each image and subimage
    # After individually normalized, bins are concatenated for each image
    visual_words_test = pyramid_visual_word(pyramid_descriptors, codebook, k_codebook, test_descriptors) 
    
    return visual_words_test

def compute_accuracy_labels(test_labels, train_labels, test_data, clf):

    accuracy = 100*clf.score(test_data, y_test)   
    predicted_labels = clf.predict(test_data)  
    unique_labels = list(set(train_labels))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_labels, predicted_labels, labels = unique_labels)
    
    return accuracy, cnf_matrix, unique_labels
    

if __name__ == "__main__":
    
    # Prepare files from DS for training
    train_images_filenames = open_pkl('train_images_filenames.dat')
    train_labels = open_pkl('train_labels.dat')
    test_images_filenames = open_pkl('test_images_filenames.dat')
    test_labels = open_pkl('test_labels.dat')
    
    
    # Define Variables
    (sift_step_size, sift_scale, dense, k_codebook, type_classifier, 
            svm_dict, knn_dict, pyramid_level, intersection) = variables()
    
    # INIT CLASSIFIER
    if type_classifier == "KNN": 
        classifier = init_classifier_knn(knn_dict)
        
    elif type_classifier =="SVM":
        # Retorna llistat de models: Linear, LinearSVC, RBF, Poly, interseccio
        classifier_svm = init_classifier_svm(svm_dict)
    else:
        sys.exit("Invalid Classifier") 
        
        
    # only want the rbf for example
    classifier = classifier_svm[0]

    accuracy_list = []    
    time_list = []
    
    range_value = np.arange(3)
    
    norm_method = "L2"

    train_images = open_pkl('train_images_filenames.dat')
    train_labels = open_pkl('train_labels.dat')   
    number_splits = 3
    X = np.array(train_images)
    y = np.array(train_labels)
    skf = StratifiedKFold(n_splits=number_splits, random_state = 42, shuffle= True)

    for swiping_variable in range_value:
        splits_accuracy = []
        splits_time = []
        split = 0 

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]    

            start = time.time()   
      
            (SIFTdetector, kpt) = compute_detector(sift_step_size, sift_scale) 
      
            
            
            (codebook, visual_words) = compute_BOW(X_train, y_train, dense, 
                                                SIFTdetector, kpt, k_codebook, 
                                                pyramid_level, norm_method)   
            bow_time = time.time()

            # Compute kernel for classifier
            if intersection:
                scld = StandardScaler().fit(visual_words)
                scaled_train = scld.transform(visual_words)
                train_data = histogram_intersection(scaled_train, scaled_train)
            else:
                train_data = visual_words

            classifier.fit(train_data, train_labels) 

            visual_words_test = test_BOW (X_test, y_test, dense, 
                                                    SIFTdetector, kpt, k_codebook, 
                                                    pyramid_level, codebook, classifier)
            
            if intersection:
                test_data = histogram_intersection(scld.transform(visual_words_test), scaled_train)
               
            else:
                test_data = visual_words_test
           
            accuracy, cnf_matrix, unique_labels = compute_accuracy_labels(y_test, 
                                                      y_train, test_data, classifier )

            class_time = time.time()
            ttime = class_time-start
            
            splits_accuracy.append(accuracy)
            splits_time.append(ttime)
            
            print ("\nAccuracy for split",split,":", accuracy, "\nTotal Time: ", class_time-start,
                    "\nBOW Time: ", bow_time-start, "\nClassification Time: ", class_time-bow_time) 
            
            # Plot normalized confusion matrix
            np.set_printoptions(precision=2)  
            plot_confusion_matrix(cnf_matrix, classes=unique_labels, 
                        normalize=True,
                        title='Normalized confusion matrix') 
            
            split += 1
        
        time_list.append(np.average(splits_accuracy))
        accuracy_list.append(np.average(splits_time))

  

        
    # Plot Acurracy
    plot_accuracy_vs_time(range_value, accuracy_list, time_list, 
                    feature_name = 'Number of SIFT scales', title = "DSIFT")
       
      