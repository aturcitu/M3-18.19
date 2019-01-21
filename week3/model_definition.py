from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape, Dropout, BatchNormalization
import keras

def create_model(IMG_SIZE, units1=2048, units2=1024, drop=0, optimizer_param='sgd', depth = 'shallow', init='glorot_uniform'):

    if optimizer_param == "adam":
        optimizer_param = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    elif optimizer_param == "adadelta":
        optimizer_param = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

    #Build the Multi Layer Perceptron model
    model = Sequential()
    #output to a certain shape: (im_size * im_size * 3)
    model.add(Reshape( (IMG_SIZE*IMG_SIZE*3,), input_shape=(IMG_SIZE, IMG_SIZE, 3), name='first') )

    if depth == 'shallow':
        #implements the operation with the activation function
        model.add( Dense ( units=2048, activation='relu', name='second') )
        model.add( Dense ( units=8, activation='softmax') )
    
    if depth == 'dense':
        model.add(Dense(units=64, activation='relu', name='second'))
        model.add(Dropout(0.5))
        model.add(Dense(units=64, activation='relu', name='third'))
        model.add(Dropout(0.5))
        model.add(Dense(units=8, activation='softmax'))

    if depth == 'dense2':
        model.add(Dense(units=2048, activation='relu', name='second'))
        model.add(Dropout(0.5))
        model.add(Dense(units=1024, activation='relu', name='third'))
        model.add(Dropout(0.5))
        model.add(Dense(units=8, activation='softmax'))        

    if depth == 'dense3':
        model.add(Dense(units=2048, activation='relu', name='second'))
        model.add(Dense(units=1024, activation='relu', name='third'))
        model.add(Dense(units=8, activation='softmax'))        

    if depth == 'dense4':
        model.add(Dense(units=units1, activation='relu', name='second'))
        model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None))
        model.add(Dropout(drop))
        model.add(Dense(units=units2, activation='relu', name='third'))
        model.add(Dense(units=8, activation='softmax'))        


    if depth == 'patches':
        model.add(Dense(units=2048, activation='relu', name='second'))
        model.add(Dense(units=1024, activation='relu', name='third'))


    #model configuration 
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer_param,
                metrics=['accuracy'])    

    return model
