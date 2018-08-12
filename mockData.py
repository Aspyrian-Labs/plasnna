import numpy as np
import random

xData = np.random.rand(100,28,28)

print(xData.shape)
print(xData[0][1])

# generate 100 lists of 10 zeroes
yData = np.random.randint(0,1,(100,10))
for i in range(len(yData)):
	j = random.randint(0,9)
	yData[i][j] = 1
	print (j, yData[i])

print(yData.shape)
print(yData)