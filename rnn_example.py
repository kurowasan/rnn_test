import random
import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

# open and read text file
file = open('C://Users//kurowasan//Desktop//analysis//poe_preprocessed.txt', encoding="utf8")
data = file.read()
file.close()

# convert to onehot
alphabet = list(set(data))
char_to_int = dict((c, i) for i, c in enumerate(alphabet))

# prepare the dataset of input to output pairs encoded as integers
seq_length = 5 #how many letters before making a prediction

dataX = []
dataY = []
for i in range(0, len(data) - seq_length, 1):
    seq_in = data[i:i + seq_length]
    seq_out = data[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (len(dataX), seq_length, 1))
# normalize
X = X / float(len(alphabet))
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(alphabet)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, nb_epoch=20, batch_size=128,callbacks=callbacks_list)