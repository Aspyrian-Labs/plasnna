#mnist data files from https://pjreddie.com/projects/mnist-in-csv/
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

for file in ['mnist_train.csv', 'mnist_test.csv']:
	xData, yData = [], []
	df = pd.read_csv(file)

	for row in tqdm(range(len(df)), desc=file):
		x = []
		for i in range(1, 784, 28):
			x.append([df.iloc[row][i+j]/255 for j in range(28)])
		
		y = [0 for i in range(10)]
		y[df.iloc[row][0]] = 1

		assert np.asarray(x).shape == (28, 28)

		xData.append(np.asarray(x))
		yData.append(np.asarray(y))

	xData = np.asarray(xData)
	yData = np.asarray(yData)

	pickle.dump(xData, open('%s_X.p' % file[:-4], 'wb'))
	pickle.dump(yData, open('%s_Y.p' % file[:-4], 'wb'))
	