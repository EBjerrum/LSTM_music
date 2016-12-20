# LSTM music from piano rolls 
import numpy as np

#TODO, check what is used
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense #, Activation, Dropout
from keras.layers import LSTM, Convolution1D
from keras.optimizers import Adam, RMSprop
#from keras.utils.data_utils import get_file
import numpy as np
#import random
#import sys

from NNmusiclib import dataset_load

dataset = dataset_load('data/Nottingham/train/*')

notes= dataset.shape[1] #Number of notes in the dataset


debug = True

def return_batch(dataset, startindex, step=20, maxlen=100):
	assert startindex < step, 'Startindex must be smaller than step'
	if debug: print('Vectorizing segment %s using %s steps and maxlen %s'%(startindex,step,maxlen))
	sentences = []
	next_notes = []
	#Build up sentences
	for i in range(startindex, dataset.shape[0]-step-maxlen, step):
	    sentences.append(dataset[i: i + maxlen]) #from i to i+maxlen (not including)
	    next_notes.append(dataset[i + maxlen]) #next notes
	return np.array(sentences), np.array(next_notes)

maxlen=100
#Build a simple LSTM engine in Keras
print('Build model...')
model = Sequential()
#TODO add a convolutional layer spanning an Octave (12 tones) #Or should it wrap around in octaves, in a 2D timestep, and then have a 4(dur and mol)x2(1 octave) convolutional network.
model.add(Convolution1D(notes, 12, border_mode='same', input_shape=(maxlen, notes)))
model.add(LSTM(128, return_sequences=True))#, input_shape=(maxlen, notes))) #return_sequences if stacking
#model.add(LSTM(128, input_shape=(maxlen, notes)))
model.add(LSTM(128))
model.add(Dense(notes)) #Output layer of what keys to press.

lr = 0.005
decayfactor = 0.5
#decayevery = 3
batchsize = 128

optimizer = Adam(lr=lr) #TODO test other optimizers
loss = 'mse'#'categorical_crossentropy' # alternatives 'mse'
#optimizer = RMSprop(lr=lr)
model.compile(loss=loss, optimizer=optimizer)


segments = 20
X_test, Y_test = return_batch(dataset, segments-1, step=segments, maxlen=100)

def fitmodel(model, epochs):
	for ep in range(epochs):
		for i in range(segments-2): #Last segment is test set
			X,Y = return_batch(dataset, i, step=segments, maxlen=100)
			model.fit(X, Y, batch_size=batchsize, nb_epoch=1, validation_data=(X_test, Y_test) ) #TODO grap history object, but how to concatenate?

fitmodel(model,3)

#Lower learning rate
for i in range(7):
	lr=lr*decayfactor#, validation_data=(Xv,yv))#TODO add some test set to
	print(lr)
	optimizer = Adam(lr=lr) #TODO test other optimizers
	model.compile(loss=loss, optimizer=optimizer) #Can lr be set without recompiling??
	firmodel(model,3)

model.save('models/LSTM_conv_2x128.h5')
