import os
import numpy as np

from sklearn import svm
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier

from PIL import Image
from scipy.misc import imresize

def get_features(model_layer, directory, IMG_SIZE, type_set = 'train'):
    
    descriptors = []
    desc_labels = []
    if type_set == 'test':
      directory = directory+'/test'
    elif type_set == 'train':
      directory = directory+'/train'
    else:
      print('Incorrect imageset type')
    
    for class_im in os.listdir(directory):
        for im in os.listdir(directory+'/'+class_im): 
            x = np.asarray(Image.open(directory+'/'+class_im+'/'+im))
            x = np.expand_dims(imresize(x, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
            features = model_layer.predict(x/255.0)
            descriptors.append(features[0])
            desc_labels.append(class_im)
    
    return np.array(descriptors, dtype='f'), desc_labels     


def get_patch_features(model_layer, directory, PATCH_SIZE, type_set = 'train'):
    
    descriptors = []
    desc_labels = []
    if type_set == 'test':
      directory = directory+'/test'
    elif type_set == 'train':
      directory = directory+'/train'
    else:
      print('Incorrect imageset type')
    
    print('Computing features for',type_set,'dataset')
    
    im_desc = []
    for class_im in os.listdir(directory):
        im_or = os.listdir(directory+'/'+class_im)[0].split('.')[0]
        for im_patch in os.listdir(directory+'/'+class_im):                 
            x = np.asarray(Image.open(directory+'/'+class_im+'/'+im_patch))
            x = np.expand_dims(imresize(x, (PATCH_SIZE, PATCH_SIZE, 3)), axis=0)
            features = model_layer.predict(x/255.0)
    
            if im_or == im_patch.split('.')[0]:
                im_desc.append(features[0])
            else:
                descriptors.append(im_desc)
                desc_labels.append(class_im)
                im_desc = [features[0]]
                im_or = im_patch.split('.')[0]
    
    print('Done')
               
    return descriptors, desc_labels     


def compute_VW(descriptors, codebook, k_codebook = 128):

    print('Computing Visual words')
   
    visual_words = []
    for im_desc in descriptors:
        words = codebook.predict(im_desc)
        words_hist = np.bincount(words, minlength=k_codebook)
        #words_hist = normalize(words_hist.reshape(-1,1), norm= 'l2', axis=0).reshape(1,-1)
        visual_words.append(words_hist) 
        
    print('Done')
   
    return visual_words
    

def classify_model_layer_features(model_layer, directory, PATCH_SIZE, classifier = 'KNN', k_codebook = 128 ):

    descriptors, desc_labels = get_patch_features(model_layer, directory, 
                                                  PATCH_SIZE, type_set = 'train')
    D=np.vstack(descriptors)
    
    codebook = MiniBatchKMeans(n_clusters=k_codebook, batch_size=k_codebook * 20,
                                   compute_labels=False, reassignment_ratio=10 ** -4,
                                   random_state=42)
    
    print('Fitting dataset into a codebook of size', k_codebook)
    codebook.fit(D)
    print('Done')
    
    visual_words = compute_VW(descriptors, codebook, k_codebook)
                  
    if classifier == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=5,n_jobs=-1,metric='euclidean')
    elif classifier == 'SVM':
        clf = svm.SVC(gamma= 0.1, C = 10, kernel = 'rbf', decision_function_shape =  'ovr')
    
    clf.fit(visual_words, desc_labels)
    
    descriptors_test, desc_labels_test = get_patch_features(model_layer, directory, 
                                                            PATCH_SIZE, type_set = 'test')
    visual_words_test = compute_VW(descriptors_test, codebook, k_codebook)
    
    accuracy = 100*clf.score(visual_words_test, desc_labels_test)
    print(classifier,'Classifier accuracy:',accuracy)
       
        
    
    
