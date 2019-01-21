import os
import getpass
from model_definition import *
from utils import *
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.misc import imresize
import time
import configparser
import sys

RESULTS_DIR = sys.argv[1]+'/'
config = configparser.ConfigParser()
config.read(RESULTS_DIR+'config.ini')

IMG_SIZE    = int(config.get('DEFAULT','IMG_SIZE'))
BATCH_SIZE  = int(config.get('DEFAULT','BATCH_SIZE'))
DATASET_DIR = '/home/mcv/datasets/MIT_split'

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

MODEL_FNAME = RESULTS_DIR+'model_mlp.h5'

if not os.path.exists(DATASET_DIR):
  print(Color.RED, 'ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
  quit()

# Network Param
ACTIVATION_FUNCTION1 = config.get('DEFAULT','ACTIVATION_FUNCTION1')
ACTIVATION_FUNCTION2 = config.get('DEFAULT','ACTIVATION_FUNCTION2')
OPTIMIZER = config.get('DEFAULT','OPTIMIZER')
DENSITY = config.get('DEFAULT','DENSITY')

##### LOAD DATA ########
train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)

# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        DATASET_DIR+'/train',  # this is the target directory
        target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
        batch_size=BATCH_SIZE,
        classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
        class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels


validation_generator = test_datagen.flow_from_directory(
        DATASET_DIR+'/test',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
        class_mode='categorical')


##### BUILD MODEL ######
print('Building MLP model...\n')

model = create_model(IMG_SIZE, optimizer_param=OPTIMIZER, depth=DENSITY)

print(model.summary())
plot_model(model, to_file=RESULTS_DIR+'modelMLP.png', show_shapes=True, show_layer_names=True)

print('Done!\n')

if os.path.exists(MODEL_FNAME):
  print('WARNING: model file '+MODEL_FNAME+' exists and will be overwritten!\n')


###### TRAINING ######
print('Start training...\n')

history = model.fit_generator(
        train_generator,
        steps_per_epoch=1881 // BATCH_SIZE,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=807 // BATCH_SIZE)

print('Done!\n')
print('Saving the model into '+MODEL_FNAME+' \n')
model.save_weights(MODEL_FNAME)  # always save your weights after training or during training
print('Done!\n')

  # summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(RESULTS_DIR+'accuracy.jpg')
plt.close()
  # summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(RESULTS_DIR+'loss.jpg')

#to get the output of a given layer
 #crop the model up to a certain layer
model_layer = Model(inputs=model.input, outputs=model.get_layer('second').output)

#get the features from images
directory = DATASET_DIR+'/test/coast'
x = np.asarray(Image.open(os.path.join(directory, os.listdir(directory)[0] )))
x = np.expand_dims(imresize(x, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
print('prediction for image ' + os.path.join(directory, os.listdir(directory)[0] ))
features = model_layer.predict(x/255.0)
print(features)
print('Done!')
