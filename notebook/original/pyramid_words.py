#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import normalize


def pyramid_visual_word(pyramid_descriptors, codebook, k_codebook, test_descriptors):
    visual_words_test = []
    
    for pyramid_level in pyramid_descriptors:
        
        for im_pyramid, j in zip(pyramid_level, np.arange(len(pyramid_level))):
            words_hist = np.array([])
            
            for sub_im in im_pyramid:

                sub_words = codebook.predict(sub_im)
                sub_words_hist = np.bincount(sub_words, minlength=k_codebook)
                sub_words_hist = normalize(sub_words_hist.reshape(-1,1), norm= 'l2', axis=0).reshape(1,-1)
                words_hist = np.append(words_hist, sub_words_hist) 
                
            if(len(visual_words_test)<len(test_descriptors)):
               visual_words_test.append(words_hist)
               
            else:
               visual_words_test[j] = np.append(visual_words_test[j], words_hist)
    
    return np.array(visual_words_test, dtype='f')             
            