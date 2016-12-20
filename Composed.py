#Sample
#def sample(preds, temperature=1.0):
#    # helper function to sample an index from a probability array
#    preds = np.asarray(preds).astype('float64')
#    preds = np.log(preds) / temperature
#    exp_preds = np.exp(preds)
#    preds = exp_preds / np.sum(exp_preds)
#    probas = np.random.multinomial(1, preds, 1)
#    return np.argmax(probas)
#How to convert propability into keypresses across array???





#Composed
probroll = []
sampleroll = []
pressedroll = []

#Start with silence
x = np.zeros((1, maxlen, notes))

def sample(temperature = 1.0, iterations=25):
	for i in range(iterations):
		predi = model.predict(x, verbose=0)[0]
		#Add some randomness to the propabilities
		predi[predi<0] = 1E-20 #Avoid negative props
		preds = np.log(predi) / temperature
		exp_preds = np.exp(preds)
		exp_preds = exp_preds*( (1.**2 + (predi.sum()/exp_preds.sum())**3 )**0.5) #root mean square normalization 
		exp_preds[exp_preds > 1] = 0.99 #Ensure no propability over 1.
		#sample Binomial distribution for each keypress
		keypressed =np.random.binomial(1,exp_preds)
		sampleroll.append(predi)
		probroll.append(exp_preds)
		pressedroll.append(keypressed)
		#Take x one forward
		x[0,:-1] = x[0,1:] #Roll one forward
		x[0,-1] = keypressed #insert new prediction

sample(0.6,30)
sample(0.9,30)
sample(1.2,30)
sample(0.6,30)
sample(1.2,30)

midiwrite('2x128_composed.mid', pressedroll , r, dt)
