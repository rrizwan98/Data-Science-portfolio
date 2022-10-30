from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.models import Sequential
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import argparse, os
import pandas as pd
import numpy as np
import keras


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    
    args, _ = parser.parse_known_args()
    epochs     = args.epochs
    batch_size = args.batch_size
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation
    
    train = training_dir
    val = validation_dir
    
#     print("Train Data Shape",train.shape)
#     print("Test Data Shape",val.shape)
    
#    Data Augmentation
    from keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import optimizers
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    val_datagen = ImageDataGenerator(rescale = 1./255.,)

    train_generator = train_datagen.flow_from_directory(train, batch_size=batch_size, class_mode='binary', target_size = (220, 220))
    validation_generator = val_datagen.flow_from_directory(val, batch_size=batch_size, class_mode = 'binary', target_size=(220, 220))
    
    print(train_generator.class_indices)
    
    input_shape = (220, 220, 3)
    
#     used ResNet50V2 model
    base_model = tf.keras.applications.ResNet50V2(weights='imagenet', input_shape=input_shape, include_top=False)

    for layer in base_model.layers:
        layer.trainable = False
#     base_model.summary()
    
#    Define model hyperperameters
    
    from keras import models, regularizers
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512, activation = 'relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation = 'relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation = 'relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    #compile your model
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy"])
    
    #start training
    history = model.fit(train_generator,
                        validation_data=validation_generator,
                        steps_per_epoch = len(train_generator),
                        epochs=epochs,
                        validation_steps=len(validation_generator))
    
    # save model
    model.save(os.path.join(args.model_dir, '000000001'), 'my_model.h5')
#     model.save(f'{args.model_dir}/1')