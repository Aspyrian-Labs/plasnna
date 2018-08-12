import plasnna as pnn
# import pickle
import random
import numpy as np
#load processed data and labels with pickle

#Generate mock data
data = np.random.rand(100,28,28)         # 100 arrays of 28x28 random numbers 0-1 
labels = np.random.randint(0,1,(100,10))	  # 100 arrays of 10 ints, one of which is 1; the rest are 0
for i in range(len(labels)):
	j = random.randint(0,9)
	labels[i][j] = 1

nn = pnn.Plasma(data[0].shape, (labels[0].shape[0],1))
acc = nn.evolve(xData=data, yData=labels)

print(acc)
