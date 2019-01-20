from __future__ import print_function
from utils import *
from keras.models import Sequential
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from model_definition import create_model
import configparser

RESULTS_DIR = sys.argv[1]+'/' # /home/grupo1/workT
config = configparser.ConfigParser()
config.read(RESULTS_DIR+'config.ini')

######## INIT VARIABLES ######

BATCH_SIZE  = 16
DATASET_DIR = '/home/mcv/datasets/MIT_split'
PATCHES_DIR = RESULTS_DIR+'data/MIT_split_patches'
MODEL_FNAME = RESULTS_DIR+'patch_based_mlp.h5'
PATCH_SIZE  = int(config.get('DEFAULT','PATCH_SIZE'))
EPOCH  = int(config.get('DEFAULT','EPOCH'))


def build_mlp(input_size=PATCH_SIZE, phase='TRAIN'):
  optimizer_param='sgd'
  # Create Model
  model = create_model(input_size, optimizer_param='sgd', depth='patches')

  if phase=='TEST':
    model.add(Dense(units=8, activation='linear')) # In test phase we softmax the average output over the image patches
  else:
    model.add(Dense(units=8, activation='softmax'))

  model.compile(loss='categorical_crossentropy',optimizer=optimizer_param,metrics=['accuracy'])
  return model


if not os.path.exists(DATASET_DIR):
  colorprint(Color.RED, 'ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
  quit()

##### Create image Patches ######
# Per l'script que tenim, no importa, ja que qualsevol canvi de parametre crea un nou directori
if not os.path.exists(PATCHES_DIR):
  colorprint(Color.YELLOW, 'WARNING: patches dataset directory '+PATCHES_DIR+' do not exists!\n')
  colorprint(Color.BLUE, 'Creating image patches dataset into '+PATCHES_DIR+'\n')
  
  generate_image_patches_db(DATASET_DIR,PATCHES_DIR,patch_size=PATCH_SIZE)
  colorprint(Color.BLUE, 'Done!\n')

###### BUILD MODEL #########
colorprint(Color.BLUE, 'Building MLP model...\n')

model = build_mlp(input_size=PATCH_SIZE)
print(model.summary())
colorprint(Color.BLUE, 'Done!\n')

####### TRAIN PHASE ############
if not os.path.exists(MODEL_FNAME):
  colorprint(Color.YELLOW, 'WARNING: model file '+MODEL_FNAME+' do not exists!\n')
  colorprint(Color.BLUE, 'Start training...\n')
  
  train_datagen = ImageDataGenerator(
          rescale=1./255,
          horizontal_flip=True)
 
  test_datagen = ImageDataGenerator(rescale=1./255)
  
  train_generator = train_datagen.flow_from_directory(
          PATCHES_DIR+'/train',  # this is the target directory
          target_size=(PATCH_SIZE, PATCH_SIZE),  
          batch_size=BATCH_SIZE,
          classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
          class_mode='categorical')  
  
  validation_generator = test_datagen.flow_from_directory(
          PATCHES_DIR+'/test',
          target_size=(PATCH_SIZE, PATCH_SIZE),
          batch_size=BATCH_SIZE,
          classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
          class_mode='categorical')
  
  model.fit_generator(
          train_generator,
          steps_per_epoch=18810 // BATCH_SIZE,
          epochs=EPOCH,
          validation_data=validation_generator,
          validation_steps=8070 // BATCH_SIZE)
  
  colorprint(Color.BLUE, 'Done!\n')
  colorprint(Color.BLUE, 'Saving the model into '+MODEL_FNAME+' \n')
  model.save_weights(MODEL_FNAME)  # always save your weights after training or during training
  colorprint(Color.BLUE, 'Done!\n')