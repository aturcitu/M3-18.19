from __future__ import print_function
from utils_custom import classify_model_layer_features
from utils import *
from keras.models import Sequential,Model
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from model_definition import create_model
import configparser
import sys

RESULTS_DIR = sys.argv[1]+'/' # /home/grupo1/workT
config = configparser.ConfigParser()
config.read(RESULTS_DIR+'config.ini')

######## INIT VARIABLES ######

BATCH_SIZE  = 16
DATASET_DIR = '/home/mcv/datasets/MIT_split'
MODEL_FNAME = RESULTS_DIR+'patch_based_mlp.h5'
PATCH_SIZE  = int(config.get('DEFAULT','PATCH_SIZE'))
BOW  = config.getboolean('DEFAULT','BOW')
BOW=False
PATCHES_DIR = RESULTS_DIR+'data/MIT_split_patches'


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

####### TEST PHASE #########
colorprint(Color.BLUE, 'Building MLP model for testing...\n')

# define model structure
model = build_mlp(input_size=PATCH_SIZE, phase='TEST')
print(model.summary())

colorprint(Color.BLUE, 'Done!\n')

colorprint(Color.BLUE, 'Loading weights from '+MODEL_FNAME+' ...\n')
print ('\n')
# load model
model.load_weights(MODEL_FNAME)
colorprint(Color.BLUE, 'Done!\n')


if BOW:
    layer_output = model.get_layer(name='third')
    model = Model(inputs=model.input, outputs=layer_output.output)
    #classify_model_layer_features(model_layer, PATCHES_DIR, PATCH_SIZE, classifier = 'SVM')

else:
    colorprint(Color.BLUE, 'Start evaluation ...\n')

    # Pel dataset complete de test, sense patch size
    directory = DATASET_DIR+'/test'
    classes = {'coast':0,'forest':1,'highway':2,'inside_city':3,'mountain':4,'Opencountry':5,'street':6,'tallbuilding':7}
    correct = 0.
    total   = 807
    count   = 0

    for class_dir in os.listdir(directory):
        cls = classes[class_dir]
        for imname in os.listdir(os.path.join(directory,class_dir)):
            im = Image.open(os.path.join(directory,class_dir,imname))
            patches = image.extract_patches_2d(np.array(im), (PATCH_SIZE, PATCH_SIZE),  max_patches = int((256*256)/(PATCH_SIZE*PATCH_SIZE)) )
            out = model.predict(patches/255.)
            predicted_cls = np.argmax( softmax(np.mean(out,axis=0)) )
            if predicted_cls == cls:
                correct+=1
            count += 1
        print('Evaluated images: '+str(count)+' / '+str(total), end='\r')
        
    colorprint(Color.BLUE, 'Done!\n')
    colorprint(Color.GREEN, 'Test Acc. = '+str(correct/total)+'\n')
