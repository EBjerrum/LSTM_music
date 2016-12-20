from midi.utils import midiread
import glob
import numpy as np


def dataset_load(path,
r=(21, 109), #Key span
dt=0.3):     #Timestep for conversion into piano roll
	#Read files
	files = glob.glob('data/Nottingham/train/*')
	assert len(files) > 0, 'Training set is empty!' \
		                       ' (did you download the data files?)'
	pianorolls = [midiread(f, r, dt).piano_roll for f in files]
	#Concatenate 
	dataset = np.concatenate(pianorolls, axis = 0) #TODO add some zeros, so that one music piece is clearly different from another.
	return dataset



