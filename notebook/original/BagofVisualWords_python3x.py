import cv2
import numpy as np
import pickle
import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import itertools
from sklearn import svm
import sys


from classifier import init_classifier_svm, init_classifier_knn
from visualization import plot_accuracy_vs_time, plot_confusion_matrix

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
    """    
    SIFTdetector = cv2.xfeatures2d.SIFT_create(nfeatures=n_features)
    
    if(isinstance(sift_scale, list) == False):
        sift_scale = [sift_scale]
    
    kpt = [cv2.KeyPoint(x, y, scale) for y in 
           range(0, 256, sift_step_size) for x in 
           range(0, 256, sift_step_size) for scale in sift_scale ]
    
    
    return (SIFTdetector, kpt)    


def create_BOW(dense, SIFTdetector, kpt, k_codebook):
    
    train_images_filenames = open_pkl('train_images_filenames.dat')
    train_labels = open_pkl('train_labels.dat')
      
    Train_descriptors = []
    Train_label_per_descriptor = []
     
    for filename, labels in zip(train_images_filenames,train_labels):
        ima=cv2.imread(filename)
        gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        
        if dense:
            (_, des) = SIFTdetector.compute(gray, kpt)
            #norm here
        else:
            (_, des) = SIFTdetector.detectAndCompute(gray, None)
            #norm here
            
        Train_descriptors.append(des)
        Train_label_per_descriptor.append(labels)
 
    D =  np.vstack(Train_descriptors)
    
    codebook = MiniBatchKMeans(n_clusters=k_codebook, batch_size=k_codebook*20,
                               compute_labels=False, reassignment_ratio=10**-4, 
                               random_state=42)
    codebook.fit(D)
    
    visual_words = np.zeros((len(Train_descriptors),k_codebook),dtype=np.float32)    
    for i in range(len(Train_descriptors)):
        words = codebook.predict(Train_descriptors[i])
        visual_words[i,:] = np.bincount(words, minlength = k_codebook)
        #norm here
    return codebook, visual_words, train_labels 


    
def classify_BOW(dense, k_codebook, visual_words, codebook, train_labels, 
                 clf):
    
    test_images_filenames = open_pkl('test_images_filenames.dat')
    test_labels = open_pkl('test_labels.dat')
    
    # Fit model with training data  
    clf.fit(visual_words, train_labels) 
    
    visual_words_test=np.zeros((len(test_images_filenames),k_codebook),
                               dtype=np.float32)
    
    
    for i in range(len(test_images_filenames)):
    
        filename=test_images_filenames[i]
        ima=cv2.imread(filename)
        gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        
        if dense:
            (kp, des) = SIFTdetector.compute(gray, kpt)
        else:
            (kp, des) = SIFTdetector.detectAndCompute(gray,None)  
            
        words = codebook.predict(des)
        visual_words_test[i,:] = np.bincount(words,minlength=k_codebook)
        

    # Score Results with Test Data
    accuracy = 100*clf.score(visual_words_test, test_labels)
    
    predicted_labels = clf.predict(visual_words_test)
    
    unique_labels = list(set(train_labels))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_labels, 
                                  predicted_labels, labels = unique_labels)

    return accuracy, cnf_matrix, unique_labels
     

if __name__ == "__main__":
    
    # Determines total number of kps in an given image (set composed of 256x256px img)
    sift_step_size = 20
    # List providing scale values to compute at each kp
    sift_scale = [16]#[8,16,32]
    # Dense/Normal Sift 
    dense = True
    
    # Number of clusters in KMeans, size of codebook (words)
    k_codebook = 128
    
    type_classifier = "SVM" # SVM
    ## Values for the classifiers
    knn_dict =	{
      "k_classifier": 5,
      "distance_method": "euclidean",
    }
    svm_dict =	{
      "kernel": "linear",
      "C": 1,
    }
    
    # INIT CLASSIFIER
    if type_classifier == "KNN": 
        classifier = init_classifier_knn(knn_dict)
    elif type_classifier =="SVM":
        # Retorna llistat de models: Linear, LinearSVC, RBF, Poly
        classifier_svm = init_classifier_svm(svm_dict)
        
    else:
        sys.exit("Invalid Classifier")    
        
    accuracy_list = []    
    time_list = []
    
    #only want the rbf
    classifier = classifier_svm[2]
    
    #for classifier in classifier_svm:
    start = time.time()   
    
    (SIFTdetector, kpt) = compute_detector(sift_step_size, sift_scale)
    print(len(kpt))
    
    codebook, visual_words, labels = create_BOW(dense, SIFTdetector, 
                                                kpt, k_codebook)   
    bow_time = time.time()
    
    accuracy, cnf_matrix, unique_labels = classify_BOW(dense, k_codebook, 
                                                       visual_words, codebook, 
                                                       labels, classifier)
    accuracy_list.append(accuracy)
    
    class_time = time.time()
    
    ttime = class_time-start
    
    time_list.append(ttime)
    
    print ("Accuracy:",accuracy,"Total Time: ", class_time-start,
           ". BOW Time: ", bow_time-start,
           ". Classification Time: ", class_time-bow_time) 
 
    # Plot normalized confusion matrix
    np.set_printoptions(precision=2)  
    plot_confusion_matrix(cnf_matrix, classes=unique_labels, 
                          normalize=True,
                          title='Normalized confusion matrix')   
    
    # Plots
#    range_value = list(range(len(classifier_svm)))
#    plot_accuracy_vs_time(range_value, accuracy_list, time_list, 
#                       feature_name = 'Number of SIFT scales', title = "DSIFT")
       
   
     
 
    