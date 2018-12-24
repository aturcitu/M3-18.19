import cv2
import numpy as np
import pickle
import time
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
                 k_classifier, distance_method):
    
    test_images_filenames = open_pkl('test_images_filenames.dat')
    test_labels = open_pkl('test_labels.dat')
          
    knn = KNeighborsClassifier(n_neighbors=k_classifier, n_jobs=-1, 
                               metric=distance_method)

    knn.fit(visual_words, train_labels) 
    
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
            
        words=codebook.predict(des)
        visual_words_test[i,:]=np.bincount(words,minlength=k_codebook)
        


    accuracy = 100*knn.score(visual_words_test, test_labels)
    predicted_labels = knn.predict(visual_words_test)
    unique_labels = list(set(train_labels))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_labels, 
                                  predicted_labels, labels = unique_labels)

    return accuracy, cnf_matrix, unique_labels
     

if __name__ == "__main__":
    
    # Determines total number of kps in an given image (set composed of 256x256px img)
    sift_step_size = 20
    # List providing scale values to compute at each kp
    sift_scale = [8,16,32]
    # Dense/Normal Sift 
    dense = True
    # Number of clusters in KMeans, size of codebook (words)
    k_codebook = 128
    # number of neightbours taken into account for the classifier
    k_classifier =  5        
    # Distance metric use to match 
    distance_method = "euclidean"

    accuracy_list = []    
    time_list = []
    
    
    #range_value = [int(2**(e)) for e in range(3,6)]
    range_value = [[16],[16,32],[8,16,32],[8,16,32,64]]
    
    for sift_scale in range_value:
        start = time.time()   
        (SIFTdetector, kpt) = compute_detector(sift_step_size, sift_scale)
        print(len(kpt))
        codebook, visual_words, labels = create_BOW(dense, SIFTdetector, 
                                                    kpt, k_codebook)   
        bow_time = time.time()
        accuracy, cnf_matrix, unique_labels = classify_BOW(dense, k_codebook, 
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
 
    range_value = [1,2,3,4]
    plot_accuracy_vs_time(range_value, accuracy_list, time_list, 
                       feature_name = 'Number of SIFT scales', title = "DSIFT")
       
   
    # Plot normalized confusion matrix
    np.set_printoptions(precision=2)  
    plot_confusion_matrix(cnf_matrix, classes=unique_labels, normalize=True,
                          title='Normalized confusion matrix')        
 
    