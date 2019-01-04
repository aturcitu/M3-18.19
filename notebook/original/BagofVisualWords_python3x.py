import cv2
import numpy as np
import pickle
import time

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import sys

from pyramid_words import pyramid_visual_word
from classifier import init_classifier_svm, init_classifier_knn, compute_intersection_kernel, compute_regular_kernel
from visualization import plot_accuracy_vs_time, plot_confusion_matrix
from config import variables



def open_pkl(pkl_file):
    """
    This function opens pkl files providing file name on WD.
    """
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data


def compute_detector(sift_step_size, sift_scale, n_features=300):
    """
    Computes Sift detector object.
    Computes mesh of KPs using a custom step size and scale value(s).
    Points are shifted by sift_step_size/2 in order to avoid points on 
    image borders
    """
    SIFTdetector = cv2.xfeatures2d.SIFT_create(nfeatures=n_features)

    if not isinstance(sift_scale, list):
        sift_scale = [sift_scale]

    kpt = [cv2.KeyPoint(x, y, scale) for y in
           range(int(sift_step_size / 2) - 1, 256 - int(sift_step_size / 2), sift_step_size) for x in
           range(int(sift_step_size / 2) - 1, 256 - int(sift_step_size / 2), sift_step_size) for scale in sift_scale]

    return (SIFTdetector, kpt)


def compute_des_pyramid(dataset_desc, pyramid_level, img_px=256):
    """
    Computes Pyramid divison of the kp descriptors dataset
    It uses KPs values to descriminate to which level each descriptor belongs
    """
    div_level = int(2 ** (pyramid_level))
    pyramid_res = img_px / div_level
    pyramid_desc = []

    for image_desc in dataset_desc:
        im_pyramid_desc = []
        # axis 0 divisions
        for n in range(1, div_level + 1):
            # axis 1 divisions
            for m in range(1, div_level + 1):
                sub_desc = []
                for kp_desc, kp in zip(image_desc, kpt):
                    x, y = kp.pt
                    # sub resolution area
                    if (((n - 1) * pyramid_res <= x < n * pyramid_res) and
                            ((m - 1) * pyramid_res <= y < m * pyramid_res)):
                        sub_desc.append(kp_desc)

                im_pyramid_desc.append(np.array(sub_desc, dtype='f'))

        pyramid_desc.append(im_pyramid_desc)

    return pyramid_desc


def compute_BOW(train_images_filenames, dense, SIFTdetector, kpt,
                k_codebook, pyramid_level, norm_method):
    train_descriptors = []
    # Compute SIFT descriptors for whole DS 
    for filename in train_images_filenames:
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        if dense:
            (_, des) = SIFTdetector.compute(gray, kpt)
        else:
            (_, des) = SIFTdetector.detectAndCompute(gray, None)
            # Creates a list with all the descriptors
        train_descriptors.append(des)

    # Descriptors are clustered with KMeans (whole image, e.g pyramid_level = 0)
    descriptors = np.vstack(train_descriptors)

    codebook = MiniBatchKMeans(n_clusters=k_codebook, batch_size=k_codebook * 20,
                               compute_labels=False, reassignment_ratio=10 ** -4,
                               random_state=42)
    codebook.fit(descriptors)

    # Pyramid Representation of n Levels
    pyramid_descriptors = []

    while pyramid_level >= 0:
        pyramid_descriptors.append(compute_des_pyramid(train_descriptors, pyramid_level))
        pyramid_level -= 1

    # Create visual words with normalized bins for each image and subimage
    # After individually normalized, bins are concatenated for each image

    visual_words = pyramid_visual_word(pyramid_descriptors, codebook, k_codebook, train_descriptors)

    return codebook, visual_words


def test_BOW(test_images_filenames, dense, SIFTdetector, kpt, k_codebook, pyramid_level, codebook):
    test_descriptors = []

    for filename in test_images_filenames:
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        if dense:
            (_, des) = SIFTdetector.compute(gray, kpt)
        else:
            (_, des) = SIFTdetector.detectAndCompute(gray, None)

        test_descriptors.append(des)

    # Pyramid Representation of n Levels            
    pyramid_descriptors = []

    while pyramid_level >= 0:
        pyramid_descriptors.append(compute_des_pyramid(test_descriptors, pyramid_level))
        pyramid_level -= 1

    # Create visual words with normalized bins for each image and subimage
    # After individually normalized, bins are concatenated for each image
    visual_words_test = pyramid_visual_word(pyramid_descriptors, codebook, k_codebook, test_descriptors)

    return visual_words_test


def compute_accuracy_labels(test_labels, train_labels, test_data, clf):

    accuracy = 100 * clf.score(test_data, test_labels)
    predicted_labels = clf.predict(test_data)
    unique_labels = list(set(train_labels))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_labels, predicted_labels, labels=unique_labels)

    return accuracy, cnf_matrix, unique_labels


def cross_validation(skf, X, y, sift_scale, sift_step_size, k_codebook, dense, pyramid_level, norm_method, compute_kernel):
    splits_accuracy = []
    splits_time = []

    for number, train_index, test_index in enumerate(skf.split(X, y)):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        start = time.time()

        (SIFTdetector, kpt) = compute_detector(sift_step_size, sift_scale)

        (codebook, visual_words) = compute_BOW(x_train, dense,
                                               SIFTdetector, kpt, k_codebook,
                                               pyramid_level, norm_method)
        bow_time = time.time()

        # Compute kernel for classifier
        # If not intersection, kernelMatrix = visual_words
        kernel_matrix = compute_kernel(visual_words, visual_words)

        classifier.fit(kernel_matrix, y_train)

        visual_words_test = test_BOW(x_test, dense,
                                     SIFTdetector, kpt, k_codebook,
                                     pyramid_level, codebook)

        # Compute kernel for classifier
        # If not intersection, kernelMatrix = visual_words
        kernel_matrix_test = compute_kernel(visual_words_test, visual_words)

        accuracy, cnf_matrix, unique_labels = compute_accuracy_labels(y_test, y_train, kernel_matrix_test, classifier)

        class_time = time.time()
        ttime = class_time - start

        # Add accuracy for each validation step
        splits_accuracy.append(accuracy)
        splits_time.append(ttime)

        print("\nAccuracy for split", number, ":", accuracy, "\nTotal Time: ", class_time - start,
              "\nBOW Time: ", bow_time - start, "\nClassification Time: ", class_time - bow_time)

        # Plot normalized confusion matrix
        np.set_printoptions(precision=2)
        plot_confusion_matrix(cnf_matrix, classes=unique_labels,
                              normalize=True,
                              title='Normalized confusion matrix')

    return splits_accuracy, splits_time


if __name__ == "__main__":

    # Prepare files from DS for training
    train_images = open_pkl('train_images_filenames.dat')
    train_labels = open_pkl('train_labels.dat')
    test_images = open_pkl('test_images_filenames.dat')
    test_labels = open_pkl('test_labels.dat')

    # Define Variables
    (sift_step_size, sift_scale, dense, k_codebook, type_classifier,
     svm_dict, knn_dict, pyramid_level, intersection) = variables()

    # INIT CLASSIFIER
    if type_classifier == "KNN":
        classifier = init_classifier_knn(knn_dict)

    elif type_classifier == "SVM":
        classifier_svm = init_classifier_svm(svm_dict)

        # only want the rbf for example
        classifier = classifier_svm[0][0]
        classifier_name = classifier_svm[0][1]
        intersection = (classifier_name == "inter")

    else:
        sys.exit("Invalid Classifier")

    accuracy_list = []
    time_list = []

    range_value = np.arange(3)

    norm_method = "L2"

    number_splits = 3
    X = np.array(train_images)
    y = np.array(train_labels)
    skf = StratifiedKFold(n_splits=number_splits, random_state=42, shuffle=True)

    for swiping_variable in range_value:

        if intersection:
            accuracy_validation, time_validation = cross_validation(skf, X, y, sift_scale, sift_step_size, k_codebook,
                                                                dense, pyramid_level, norm_method, compute_intersection_kernel)
        else:
            accuracy_validation, time_validation = cross_validation(skf, X, y, sift_scale, sift_step_size, k_codebook,
                                                                dense, pyramid_level, norm_method, compute_regular_kernel)

        # Append for the different testing values
        time_list.append(np.average(accuracy_validation))
        accuracy_list.append(np.average(time_validation))

    # Plot Accuracy
    plot_accuracy_vs_time(range_value, accuracy_list, time_list,
                          feature_name='Number of SIFT scales', title="DSIFT")
