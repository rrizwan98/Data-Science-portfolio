import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse, os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout, Activation
from keras.utils import np_utils

print("TensorFlow version", tf.__version__)

# Process command-line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning-rate', type=float, default=0.01)
parser.add_argument('--batch-size', type=int, default=128)

parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

args, _ = parser.parse_known_args()

epochs     = args.epochs
lr         = args.learning_rate
batch_size = args.batch_size

gpu_count  = args.gpu_count
model_dir  = args.model_dir
training_dir   = args.training
validation_dir = args.validation

# Load data set
X_train = np.load(os.path.join(training_dir, 'training.npz'))['image']
y_train = np.load(os.path.join(training_dir, 'training.npz'))['label']
X_test  = np.load(os.path.join(validation_dir, 'validation.npz'))['image']
y_test  = np.load(os.path.join(validation_dir, 'validation.npz'))['label']

num_classes = 10

# Converts a class vector (integers) to binary class matrix.
def y_cate_encode(y_train,y_test):
    print(y_train.shape)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print(y_train.shape)
    return y_train, y_test

y_train, y_test = y_cate_encode(y_train,y_test)

# As before, let's make everything float and scale
def float_scale(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # normalizer
    x_train /= 255
    x_test /= 255
    return x_train, x_test

x_train, x_test=float_scale(X_train,X_test)

# the size of single image
print(x_train.shape[1:])

# building a linear stack of layers with the sequential model

# build a sequential model
model_1 = Sequential()
## 5x5 convolution with 1x1 stride and 32 filters
model_1.add(Conv2D(32, (5, 5), padding='same',
                 input_shape=x_train.shape[1:]))
model_1.add(Activation('relu'))

## Another 5x5 convolution with 1x1 stride and 64 filters
model_1.add(Conv2D(64, (5, 5), padding='same'))
model_1.add(Activation('relu'))

## 2x2 max pooling reduces to 3 x 3 x 32
model_1.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model_1.add(Dropout(0.25))

## Flatten turns 3x3x32 into 288x1
model_1.add(Flatten())
model_1.add(Dense(512))
model_1.add(Activation('relu'))
model_1.add(Dropout(0.5))
model_1.add(Dense(num_classes))
model_1.add(Activation('softmax'))

model_1.summary()

# compile model
batch_size = 32

# initiate Adam optimizer
opt = keras.optimizers.Adam(learning_rate=0.002)

# Let's train the model using Adam
model_1.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history=model_1.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

model_1.save(os.path.join(model_dir, '1'))