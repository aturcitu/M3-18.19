import keras
from model_definition import create_model
import numpy as np
from scipy.misc import imresize
from PIL import Image
import os
import sys
from sklearn import svm
from keras.models import Sequential, Model
from sklearn.model_selection import GridSearchCV


def read_model(IMG_SIZE, PATH_TO_MODEL):
    # Init model with right architecture
    model = create_model(IMG_SIZE, 'sgd', 'dense3')
    model.load_weights(PATH_TO_MODEL)
    model.compile(loss='categorical_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])
    return model

def compute_features(directory, layer, IMG_SIZE):

    features_images = []
    for img_class in  os.listdir(directory):
        class_path = os.path.join(directory, img_class)
        for img in os.listdir(class_path):
            img_path = os.path.join(class_path, img)
            image = Image.open(img_path)
            image = np.expand_dims(imresize(image, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
                
            # compute features    
            features = layer.predict(image/255.0)
            features_images.append([features[0], img_class])

    return np.array(features_images)

def train_parameters(train, descriptors, desc_labels, kernel_type = 'rbf', function_shape = 'ovr'):
    if train:
        # Grid Search
        # Parameter Grid
        param_grid = {'C': [0.1, 0.55, 0.65, 0.75, 0.8, 0.9 ,1, 10, 100], 
                    'gamma': [0.1, 0.55, 0.65, 0.75, 0.8, 0.9 ,1, 10, 100]}


        # Make grid search classifier
        clf_grid = GridSearchCV(svm.SVC(kernel=kernel_type, 
                                        decision_function_shape=function_shape), 
                                        param_grid, verbose=1, n_jobs=4)

        # Train the classifier
        clf_grid.fit(descriptors, desc_labels)

        # clf = grid.best_estimator_()
        print("Best Parameters:\n", clf_grid.best_params_)
        print("Best Estimators:\n", clf_grid.best_estimator_)

        return clf_grid.best_estimator_

    else:
        gamma_val = 0.001
        C_val = 1
        clf = svm.SVC(gamma= gamma_val, C = C_val, kernel = kernel_type, decision_function_shape =  function_shape)
        clf = svm.SVC(gamma= gamma_val, C = C_val, kernel = kernel_type, decision_function_shape =  function_shape)
        return clf

def compute_svm(features_train, features_test, clf):
    # Get data
    data_train = features_train[:,0].tolist()
    labels_train = features_train[:,1].tolist()
    data_test = features_test[:,0].tolist()
    labels_test = features_test[:,1].tolist()

    # Create and Fit SVM
    clf.fit(data_train, labels_train) 
    print ("Fit done!")

    # Evaluate model with test data
    accuracy = 100*clf.score(data_test, labels_test)
    predicted_labels = clf.predict(data_test)

    return accuracy, predicted_labels


PATH_TO_MODEL = sys.argv[1]
IMG_SIZE = 64
DATASET_DIR = '/home/mcv/datasets/MIT_split'
SEARCH_VALUES = False

# import model
model = read_model(IMG_SIZE, PATH_TO_MODEL)

# extract last layer
layer_output = model.get_layer(name='third')
model_layer = Model(inputs=model.input, outputs=layer_output.output)

# get features from images in TRAIN directory
directory = DATASET_DIR+'/train/'
features_train = compute_features(directory, model_layer, IMG_SIZE)

# get features from images in TEST directory
directory = DATASET_DIR+'/test/'
features_test = compute_features(directory, model_layer, IMG_SIZE)
print('Done Computing features for last layer!')

# Apply SVM
classifier = train_parameters(SEARCH_VALUES, features_train[:,0].tolist(), features_train[:,1].tolist())

## Fit and Predict
accuracy, predicted_labels = compute_svm(features_train, features_test, classifier)

print ("Accuracy with SVM: ", accuracy)


