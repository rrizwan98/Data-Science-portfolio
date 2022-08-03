import argparse, os
import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.python.keras import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from keras.optimizers import SGD

if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    
    args, _ = parser.parse_known_args()
    epochs     = args.epochs
    batch_size = args.batch_size
    model_dir  = args.model_dir
    training_dir   = args.training


    train = pd.read_csv(os.path.join(training_dir,"training", 'train.csv'))
    test = pd.read_csv(os.path.join(training_dir,"testing" ,'test.csv'))
    train_image = os.path.join(training_dir,"training" )
    test_image = os.path.join(training_dir, "testing")
    print(train.head())
    
    from keras_preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(rescale=1./255., validation_split=0.1)

    train_generator = datagen.flow_from_dataframe(train,
                                                  directory=train_image,
                                                  x_col='image_path',
                                                  y_col='mask',
                                                  subset='training',
                                                  class_mode='categorical',
                                                  classes=["Brain_tumor", "Normal"],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  target_size=(256,256)                                              )
    valid_generator = datagen.flow_from_dataframe(train,
                                                  directory=train_image,
                                                  x_col='image_path',
                                                  y_col='mask',
                                                  subset='validation',
                                                  class_mode='categorical',
                                                  classes=["Brain_tumor", "Normal"],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  target_size=(256,256)
                                                 )
    test_datagen = ImageDataGenerator(rescale=1./255.)
    test_generator = test_datagen.flow_from_dataframe(test,
                                                      directory=test_image,
                                                      x_col='image_path',
                                                      y_col='mask',
                                                      class_mode='categorical',
                                                      classes=["Brain_tumor", "Normal"],
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      target_size=(256,256)
                                                     )
    input_shape=(256,256,3)
    from tensorflow.keras.applications.resnet50 import ResNet50
    clf_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # before this i tried with trainable layer but the accuracy was less as compared
    layers=clf_model.layers
    for layer in layers:
        layer.trainable = False
    
    head = clf_model.output
    head = AveragePooling2D(pool_size=(4,4))(head)
    head = Flatten(name='Flatten')(head)
    head = Dense(256, activation='relu')(head)
    head = Dropout(0.3)(head)
    head = Dense(256, activation='relu')(head)
    head = Dropout(0.3)(head)
    head = Dense(2, activation='softmax')(head)

    model = Model(clf_model.input, head)
    model.compile(optimizer="adam",
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

#     h = model.fit(train_generator, 
#                   steps_per_epoch= train_generator.n // train_generator.batch_size, 
#                   epochs = epochs, 
#                   validation_data= valid_generator, 
#                   validation_steps= valid_generator.n // valid_generator.batch_size)
    h = model.fit(train_generator, 
                  steps_per_epoch= len(train_generator), 
                  epochs = epochs, 
                  validation_data= valid_generator, 
                  validation_steps= len(valid_generator))
    
    # save Keras model for Tensorflow Serving
    
    model.save(os.path.join(model_dir,"my_model.h5"))
        # save Keras model for Tensorflow Serving
    model.save(os.path.join(model_dir,"my_model"))

