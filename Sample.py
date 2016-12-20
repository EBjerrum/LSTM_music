import matplotlib.pylab as plt
import time
from midi.utils import midiwrite

#Import model
from keras.models import load_model

model = load_model('LSTM_2x128.h5')


for temperature in [0.3,0.5,0.8,1.0,1.2]:

	probroll = []
	sampleroll = []
	pressedroll = []

	#Start with silence
	x = np.zeros((1, maxlen, notes))
	#Start with something
	#start = 12900
	#x = X[start:start+1] #start with the first segment

	#plt.matshow(x[0])
	#plt.title('starter seq')
	#x = x.copy()
	for i in range(200):
		predi = model.predict(x, verbose=0)[0]
		#Add some randomness to the propabilities
		predi[predi<0] = 1E-20 #Avoid negative props
		#predi[predi>0.999] = 0.999  #Avoid props over 1
		preds = np.log(predi) / temperature
		exp_preds = np.exp(preds)
		#rescale so that lower temperature does not get complete squashed
		#exp_preds = max(predi)*exp_preds/(max(exp_preds)) #Rescale so that max prop is the same.
		#exp_preds = exp_preds*(predi.sum()/exp_preds.sum()) #Rescale so that the collected propability stays the same.
		#exp_preds = exp_preds*( (1. + predi.sum()/exp_preds.sum())/2.) #Rescale so that the collected propability is attenuated towards the previous.
		exp_preds = exp_preds*( (1.**2 + (predi.sum()/exp_preds.sum())**3 )**0.5) #root mean square normalization 


		exp_preds[exp_preds > 1] = 0.99 #Ensure no propability over 1.
		#rand = np.random.randn(predi.shape[0])*2
		#probas = exp_preds*rand
		#preds = exp_preds / np.sum(exp_preds)
		#probas = np.random.multinomial(10, preds, 1)
		#keypressed = probas > 0.50
		#use a binomial distribution with propability as exp_preds
		keypressed =np.random.binomial(1,exp_preds)
		sampleroll.append(predi)
		probroll.append(exp_preds)
		pressedroll.append(keypressed)
		#print(keypressed)
		#Take x one forward
		x[0,:-1] = x[0,1:] #Roll one forward
		x[0,-1] = keypressed #insert new prediction


	#plt.matshow(sampleroll)
	#plt.title('Raw Predicted')
	#plt.matshow(probroll)
	#plt.title('Propability of keypress')
	#plt.matshow(pressedroll)
	#plt.title('New Keypresses')
	##plt.matshow(np.array(pianoroll) > 0.25)
	#plt.show()

	#Go from the Pianoroll to a midi file which can be played.
	midiwrite('2x128_temp%s.mid'%temperature, pressedroll , r, dt)
