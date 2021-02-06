# Classifiers

# This notebook contains the implementation of our three deep neural classifiers with correct hyperparameters.

import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint

from keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, Embedding, Flatten, MaxPooling1D, Dropout, Conv1D, Input, LSTM, SpatialDropout1D, Bidirectional

train_features = pd.read_csv('train_features.csv', delimiter=',') # load the features after creating them
test_feautres = pd.read_csv('test_features.csv', delimiter=',') # load the features after creating them

train_labels = pd.read_csv("../data/training-set.csv")["is_suicide"]
test_labels = pd.read_csv("../data/testing-set.csv")["is_suicide"]

# training hyperparameters

epochs = 80
batch_size = 32

# Convolutional Neural Network

cnn = Sequential()

cnn_path = "cnn"

filters = 3
kernal = 2

cnn.add(Input(shape=(512,768)))
cnn.add(Conv1D(filters= filters, kernel_size = kernal, activation='relu'))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
cnn.add(Dense(1, activation='sigmoid'))

cnn.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

mc = ModelCheckpoint(cnn_path + ".h5", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

cnn.summary()

# Fully Dense Network
dense = Sequential()

dense_path = "dense"

dense = Sequential()
dense.add(Input(shape=(512,)))
dense.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
dense.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
dense.add(Dense(1, activation='sigmoid'))

dense.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

mc = ModelCheckpoint(dense_path + ".h5", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

dense.summary()

# Bi-LSTM
bilstm = Sequential()

bilstm_path = "bilstm"

pool_size = 2

bilstm.add(Input(shape=(512,768)))
bilstm.add(Bidirectional(LSTM(20, return_sequences=True, dropout=0.25, recurrent_dropout=0.2)))
bilstm.add(MaxPooling1D(pool_size = pool_size))
bilstm.add(Flatten())
bilstm.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
bilstm.add(Dense(1, activation='sigmoid'))


bilstm.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

mc = ModelCheckpoint(bilstm_path + ".h5", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

bilstm.summary()
