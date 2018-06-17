import plasnna as pnn
import pickle

#load processed data and labels with pickle

nn = pnn.Plasma(data[0].shape, labels[0].shape)
acc = nn.evolve(xData=data, yData=labels)

print(acc)
