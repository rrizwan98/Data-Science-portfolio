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
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    
    args, _ = parser.parse_known_args()
    epochs     = args.epochs
    batch_size = args.batch_size
    model_dir  = args.model_dir
    training_dir   = args.training
    
    train = pd.read_csv(os.path.join(training_dir,"train", 'train-classes.csv'))
    test = pd.read_csv(os.path.join(training_dir,"test" ,'test-classes.csv'))
    train_image = os.path.join(training_dir,"train" )
    test_image = os.path.join(training_dir, "test")
    
    print("Train Data Shape",train.shape)
    print("Test Data Shape",test.shape)
    print("Column Name",train.columns)
    print("s3 image path",train_image)
    
    #prepare train images
    train_image = []
    for i in tqdm(range(train.shape[0])):
        img = tf.keras.utils.load_img("/opt/ml/input/data/training/train/" + train['filename'][i],target_size=(400,400,3))
        img = tf.keras.utils.img_to_array(img)
        img = img/255
        train_image.append(img)
    train_images = np.array(train_image)
    print(train_images.shape)
    
    #prepare test images
    test_image = []
    for i in tqdm(range(test.shape[0])):
        img = tf.keras.utils.load_img("/opt/ml/input/data/training/test/" + test['filename'][i],target_size=(400,400,3))
        img = tf.keras.utils.img_to_array(img)
        img = img/255
        test_image.append(img)
    test_images = np.array(test_image)
    print(test_images.shape)
    
    #prepare train labels
    train_labels = np.array(train.drop(['filename'],axis=1))
    print(train_labels.shape)
    
    #prepare test labels
    test_labels = np.array(test.drop(['filename'],axis=1))
    print(test_labels.shape)
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

    # seperate train test images and labels.
    X_train = train_images
    y_train = train_labels
    
    X_test = test_images
    y_test = test_labels
    
    # print shapes
#     print("X_train shape",X_train)
#     print("y_train shape",y_train)
#     print("X_test shape",X_test)
#     print("y_test shape",y_test)
    
    #apply FIne tune methord for resnet50 model
    input_shape = (400 , 400 , 3)
    base_model = tf.keras.applications.ResNet50V2(weights='imagenet', input_shape=input_shape, include_top=False)

    for layer in base_model.layers:
        layer.trainable = False
        base_model.summary()
        
    for layer in base_model.layers[:]:
        layer.trainable = False
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name, layer.trainable)
        
    # Make last block of the conv_base trainable:

    for layer in base_model.layers[:177]:
        layer.trainable = False
    for layer in base_model.layers[177:]:
        layer.trainable = True
    
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name, layer.trainable)
    
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
    model.add(Dense(16, activation='sigmoid'))
#     model.summary()

    
    #compile your model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    #start training
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=batch_size)
    
    # save model
    model.save(f'{args.model_dir}/1')