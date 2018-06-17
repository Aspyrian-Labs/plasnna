import plasnna as pnn
import pickle

#load processed data and labels with pickle

nn = pnn.Plasma(data, labels)
acc = nn.evolve()

print(acc)
