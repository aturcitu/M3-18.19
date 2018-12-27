import cv2
import numpy as np
import pickle
import time
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import itertools

def open_pkl(pkl_file):
    """
    This function opens pkl files providing file name on WD.
    """    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)    
    return data

def plot_accuracy_vs_time(x, y1, y2, feature_name, title):
    """
    This function plots a doble axis figure.
    Feature name and title can be modified to be plot
    """    
    fig, ax1 =plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel(feature_name)
    ax1.set_ylabel('accuracy', color=color)
    ax1.plot(x, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('time (s)', color=color)  
    ax2.plot(x,y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(title)
    plt.show()     

def plot_confusion_matrix(cm, classes, normalize=False, 
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def compute_detector(sift_step_size, sift_scale, n_features = 300):
    """
    Computes Sift detector objest.
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

def create_BOW(dense, SIFTdetector, kpt, k_codebook, pyramid_level = 0):
    # Prepare files from DS for training
    train_images_filenames = open_pkl('train_images_filenames.dat')
    train_labels = open_pkl('train_labels.dat')
    
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
    visual_words = []            
    for pyramid_level in pyramid_descriptors:
        for im_pyramid, j in zip(pyramid_level, np.arange(len(pyramid_level))):
            words_hist = np.array([])
            for sub_im in im_pyramid:
                sub_words = codebook.predict(sub_im)
                sub_words_hist = np.bincount(sub_words, minlength = k_codebook)
                sub_words_hist = normalize(sub_words_hist.reshape(-1,1), norm= 'l2', axis=0).reshape(1,-1)
                words_hist = np.append(words_hist, sub_words_hist) 
            if(len(visual_words)<len(train_descriptors)):
               visual_words.append(words_hist)
            else:
               visual_words[j] = np.append(visual_words[j], words_hist)              
    visual_words = np.array(visual_words, dtype='f')   
   
    return codebook, visual_words, train_labels 

def classify_BOW(dense, SIFTdetector, kpt, k_codebook, pyramid_level, visual_words, 
                 codebook, train_labels, k_classifier, distance_method):
    # Prepare files from DS for testing   
    test_images_filenames = open_pkl('test_images_filenames.dat')
    test_labels = open_pkl('test_labels.dat')
    
    # Train KNN with visual words and K neighbours    
    knn = KNeighborsClassifier(n_neighbors=k_classifier, n_jobs=-1, 
                               metric=distance_method)
    knn.fit(visual_words, train_labels) 

    # Create visual words for testing data    
    visual_words_test=np.zeros((len(test_images_filenames),k_codebook),
                               dtype=np.float32)
    test_descriptors = []    
    for filename in test_images_filenames:    
        ima=cv2.imread(filename)
        gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)       
        if dense:
            (_, des) = SIFTdetector.compute(gray, kpt)
        else:
            (_, des) = SIFTdetector.detectAndCompute(gray,None)  

        test_descriptors.append(des)
   
    # Pyramid Representation of n Levels            
    pyramid_descriptors = []
    while(pyramid_level >= 0):
        pyramid_descriptors.append(compute_des_pyramid(test_descriptors, pyramid_level))
        pyramid_level -= 1

    # Create visual words with normalized bins for each image and subimage
    # After individually normalized, bins are concatenated for each image
    visual_words_test = []            
    for pyramid_level in pyramid_descriptors:
        for im_pyramid, j in zip(pyramid_level, np.arange(len(pyramid_level))):
            words_hist = np.array([])
            for sub_im in im_pyramid:
                sub_words = codebook.predict(sub_im)
                sub_words_hist = np.bincount(sub_words, minlength = k_codebook)
                sub_words_hist = normalize(sub_words_hist.reshape(-1,1), norm= 'l2', axis=0).reshape(1,-1)
                words_hist = np.append(words_hist, sub_words_hist) 
            if(len(visual_words_test)<len(test_descriptors)):
               visual_words_test.append(words_hist)
            else:
               visual_words_test[j] = np.append(visual_words_test[j], words_hist)
    visual_words_test = np.array(visual_words_test, dtype='f')   
    
    # Evaluate model with test data
    accuracy = 100*knn.score(visual_words_test, test_labels)
    predicted_labels = knn.predict(visual_words_test)
    unique_labels = list(set(train_labels))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_labels, 
                                  predicted_labels, labels = unique_labels)

    return accuracy, cnf_matrix, unique_labels
     

if __name__ == "__main__":
    
    # Determines total number of kps in an given image (set composed of 256x256px img)
    sift_step_size = int(2**(3))
    # List providing scale values to compute at each kp
    sift_scale = [int(2**(3)),int(2**(4))]
    # Dense/Normal Sift 
    dense = True
    # Number of clusters in KMeans, size of codebook (words)
    k_codebook = 128
    # number of neightbours taken into account for the classifier
    k_classifier =  5        
    # Distance metric use to match 
    distance_method = "euclidean"
    pyramid_level = 1
    
    accuracy_list = []    
    time_list = []
    
    range_value = np.arange(2)
    
    for pyramid_level in range_value:
        start = time.time()   
        (SIFTdetector, kpt) = compute_detector(sift_step_size, sift_scale) 
        
        (codebook, visual_words, labels) = create_BOW(dense, SIFTdetector, 
                                                    kpt, k_codebook, pyramid_level)   
        bow_time = time.time()
        (accuracy, cnf_matrix, unique_labels) = classify_BOW(dense, SIFTdetector, 
                                                            kpt, k_codebook, pyramid_level, 
                                                            visual_words, codebook, 
                                                            labels, k_classifier, 
                                                            distance_method)
        accuracy_list.append(accuracy)
        class_time = time.time()
        ttime = class_time-start
        time_list.append(ttime)
        print ("Accuracy:",accuracy,"Total Time: ", class_time-start,
               ". BOW Time: ", bow_time-start,
               ". Classification Time: ", class_time-bow_time) 
 
    plot_accuracy_vs_time(range_value, accuracy_list, time_list, 
                       feature_name = 'Pyramid Level', title = "DSIFT")
       
   
    # Plot normalized confusion matrix
    np.set_printoptions(precision=2)  
    plot_confusion_matrix(cnf_matrix, classes=unique_labels, normalize=True,
                          title='Normalized confusion matrix')        
 
    