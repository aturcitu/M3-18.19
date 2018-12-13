import cv2
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier

def create_BOW(dense, sift_step_size, k_codebook):
    train_images_filenames = open_pkl('train_images_filenames.dat')
    train_labels = open_pkl('train_labels.dat')

    SIFTdetector = cv2.xfeatures2d.SIFT_create()
    kpt = [cv2.KeyPoint(x, y, sift_step_size) for y in range(0, 256, sift_step_size) for x in range(0, 256, sift_step_size)]
    
    Train_descriptors = []
    Train_label_per_descriptor = []
     
    for filename, labels in zip(train_images_filenames,train_labels):
        ima=cv2.imread(filename)
        gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        if dense:
            (kp, des) = SIFTdetector.compute(gray,kpt)
        else:
            (kp, des) = SIFTdetector.detectAndCompute(gray,None)            
        Train_descriptors.append(des)
        Train_label_per_descriptor.append(labels)
    
    D=np.vstack(Train_descriptors)
    codebook = MiniBatchKMeans(n_clusters=k_codebook, verbose=False, batch_size=k_codebook * 20,compute_labels=False,reassignment_ratio=10**-4,random_state=42)
    codebook.fit(D)
    
    visual_words=np.zeros((len(Train_descriptors),k_codebook),dtype=np.float32)
    for i in range(len(Train_descriptors)):
        words=codebook.predict(Train_descriptors[i])
        visual_words[i,:]=np.bincount(words,minlength=k_codebook)
        
    return codebook, visual_words
    
def classify_BOW(dense, sift_step_size, k_codebook, visual_words, codebook, k_classifier, distance_method):
    test_images_filenames = open_pkl('test_images_filenames.dat')
    test_labels = open_pkl('test_labels.dat')
    train_labels = open_pkl('train_labels.dat')
    
    SIFTdetector = cv2.xfeatures2d.SIFT_create()
    kpt = [cv2.KeyPoint(x, y, sift_step_size) for y in range(0, 256, sift_step_size) for x in range(0, 256, sift_step_size)]
    
    knn = KNeighborsClassifier(n_neighbors=k_classifier,n_jobs=-1,metric=distance_method)
    knn.fit(visual_words, train_labels) 
    
    visual_words_test=np.zeros((len(test_images_filenames),k_codebook),dtype=np.float32)
    for i in range(len(test_images_filenames)):
        filename=test_images_filenames[i]
        ima=cv2.imread(filename)
        gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        if dense:
            (kp, des) = SIFTdetector.compute(gray,kpt)
        else:
            (kp, des) = SIFTdetector.detectAndCompute(gray,None)  
        words=codebook.predict(des)
        visual_words_test[i,:]=np.bincount(words,minlength=k_codebook)
        
    accuracy = 100*knn.score(visual_words_test, test_labels)
    print(accuracy)      
    
    return accuracy
    
def open_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)    
    return data

if __name__ == "__main__":
    dense = True
    
    # determines total number of kps in an given image (set composed of 256x256px img)
    sift_step_size = 20
    # number of clusters in KMeans, size of codebook (words)
    k_codebook = 128   
    # CREATE AND TRAIN THE MODEL
    codebook, visual_words = create_BOW(dense, sift_step_size, k_codebook)
    
    # number of neightbours taken into account for the classifier
    k_classifier =  5
    # Distance metric use to match 
    distance_method = 'euclidean'
    # TEST THE MODEL
    classify_BOW(dense, sift_step_size, k_codebook, visual_words, codebook, k_classifier, distance_method)
