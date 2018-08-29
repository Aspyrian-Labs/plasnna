'''
Plastic neural network architecture (plasnna - pronounced plasma).

Basic documentation:
	Classes:
		Plasma
		Neuron
		Synapse
	
	Variables:
		ngf: Nerve growth factor. Consumed for a neuron to survive and produced when a neuron fires.
	
'''

import numpy as np
import random 
from tqdm import tqdm

evolveParameters = {
  'neurotransmitter_binding_chance' : 0.5,
  'activation_threshold' : 1.0,
  'ngf_per_fire' : 0.1,
  'ngf_decay_factor' : 0.05,
  'weight_factor' : 0.25,
  'plasticity_threshold' : 1.5,
  'plasticity_floor' : 0.2,
  'neighbour_range' : 2,
  'synapse_kill_threshold' : 0.2,
  'spill_threshold' : 2.0
  }

class Plasma():
	def __init__(self, inputs, outputs, gridSize=(10, 10, 10)):
		self.accuracy = 0.0
		self.inputs = inputs
		self.outputs = outputs
		self.gridSize = gridSize
		self.plasmaGrid = {}
		self.inputCoords = []
		self.outputCoords = []
		for x in range(gridSize[0]):
			for y in range(gridSize[1]):
				for z in range(gridSize[2]):
					coords = (x, y, z)
					self.plasmaGrid[coords] = Neuron(coords=coords)
		for x in range(inputs[0]):
			for y in range(inputs[1]):
				coords = (x, y, -1)
				self.plasmaGrid[coords] = Neuron(coords=coords, inputNeuron=True)
				self.inputCoords.append(coords)
		for x in range(outputs[0]):
			for y in range(outputs[1]):
				coords = (x, y, gridSize[2])
				self.plasmaGrid[coords] = Neuron(coords=coords, outputNeuron=True)
				self.outputCoords.append(coords)
		self.vis = Visualiser()
        
	def evolve(self, xData, yData, evolveParameters=evolveParameters, epochs=1, timeSteps=1000, rewardObservationLength=0.1):
		for e in range(epochs):
			#Iterate over batch
			# for i,dat in enumerate(tqdm(xData, postfix='acc: %f' % self.accuracy, ascii=True, ncols=100, desc='Epoch %i' % (e+1))):
			for i,dat in enumerate(xData):
				#Convert data to fireRate for input layer (assume normalised data)
				assert xData[0].shape == self.inputs, 'Input data shape must match input layer shape.'
				for c in self.inputCoords:
					self.plasmaGrid[c].fireRate = int(1.0/xData[i][c[0]][c[1]]) #! currently cutoff scales with timeSteps
																			   #! requires 2D data
				#Set up output recording (!NB: for classifiers only for now)
				# assert yData[0].shape == self.outputs, 'Output data shape must match output layer shape.'
				outputRecord = {}
				for c in self.outputCoords:
					outputRecord[c] = [yData[i], 0] #increment counter for each correct classification

				#Start time evolution
				currentAccuracy = 0
				accuracyRecord = []
				numSynapses = 0
				for t in tqdm(range(timeSteps), postfix='acc: %f, syn: %d' % (currentAccuracy, numSynapses), ascii=True, ncols=100):
					#Initial growth promotion: output layer requests connections
					if currentAccuracy == 0.0:
						for x in range(self.outputs[0]):
							for y in range(self.outputs[1]):
								coords = (x, y, self.gridSize[2])
								self.plasmaGrid[coords].ngf = 5.0 #enough to spill into preceeding layers

					#Fire neurons
					for coord in self.plasmaGrid:
						# print("coord = ", coord)
						neuron = self.plasmaGrid[coord]
						neuron.update(evolveParameters['ngf_per_fire'],
							evolveParameters['activation_threshold'],
							evolveParameters['plasticity_threshold'],
							evolveParameters['plasticity_floor'],
							evolveParameters['ngf_decay_factor'],
							t,
							currentAccuracy)
						if (coord[2] >= 0): # make sure we aren't in the input layer
							self.vis.update(coord, neuron.fired)
				  
				    #Propagate signal through synapses
					numSynapses = 0
					for neuron in self.plasmaGrid:
						neuron = self.plasmaGrid[neuron]
						numSynapses += len(neuron.synapses)
						for synapse in neuron.synapses:
							synapse.update(evolveParameters['neurotransmitter_binding_chance'],
								currentAccuracy, #euphamine probability
								evolveParameters['weight_factor'],
								evolveParameters['synapse_kill_threshold'])
					print(numSynapses)

					# Plasticity: form new connections between compatible neurons, distribute excess NGF
					neighbourRange = evolveParameters['neighbour_range']
					for neuron in self.plasmaGrid:
						neuronObj = self.plasmaGrid[neuron]
						if neuronObj.plasticity == 'Forwards' or neuronObj.ngf > evolveParameters['spill_threshold']:
							#Collect nearest neighbours
							synapseCandidates = []
							spillCandidates = []
							for x in range(neuron[0] - neighbourRange, neuron[0] + neighbourRange):
								for y in range(neuron[1] - neighbourRange, neuron[1] + neighbourRange):
									for z in range(neuron[2] - neighbourRange, neuron[2] + neighbourRange):
										if (x,y,z) in self.plasmaGrid: 
											neighbour = self.plasmaGrid[(x,y,z)]
											if neighbour.plasticity == 'Backwards':
												synapseCandidates.append((x,y,z))
											if neighbour.ngf < neuronObj.ngf:
												spillCandidates.append((x,y,z))
							
							if neuronObj.plasticity == 'Forwards' and len(synapseCandidates) > 0:
								candidate = random.choice(synapseCandidates)
								#Make new synapse (owned by recieving (candidate) neuron)
								self.plasmaGrid[candidate].synapses.append(Synapse(inputNeuron=self.plasmaGrid[neuron], outputNeuron=self.plasmaGrid[candidate]))

							if neuronObj.ngf > evolveParameters['spill_threshold']:
								availableNgf = neuronObj.ngf - evolveParameters['spill_threshold']
								
								#Caclulate total ngf of neighbours
								totalNgf = 0
								for candidate in spillCandidates:
									candidate = self.plasmaGrid[candidate]
									totalNgf += candidate.ngf

								#Distribute it to the most needy neurons
								for candidate in spillCandidates:
									candidate = self.plasmaGrid[candidate]
									if totalNgf > 0.0:
										candidate.ngf += availableNgf * (candidate.ngf / totalNgf)
									else : #distribute evenly if they all have nothing
										candidate.ngf += availableNgf / len(spillCandidates)

					#Verify output
					totalCorrect = 0
					for i, c in enumerate(self.outputCoords):
						# self.vis.grid[10,c[1]] += 1
						if self.plasmaGrid[c].fired and outputRecord[c][0]:
							outputRecord[c][1] += 1
							totalCorrect += 1
					currentAccuracy = float(totalCorrect)/len(self.outputCoords)
					accuracyRecord.append(currentAccuracy)

					self.vis.show()
					self.vis.reset()

			#After time evolution:
			self.accuracy = sum(accuracyRecord)/len(accuracyRecord)	
		return self.accuracy

class Visualiser():
	def __init__(self):
		self.grid = np.zeros((11,10))

	def update(self, neuronCoord, fired):
		x = neuronCoord[0]
		z = neuronCoord[2]
		if fired:
			self.grid[z,x] += 1

	def show(self):
		print("\n")
		print("0 | %3d | %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d | %3d |" % (-1, self.grid[0,0], self.grid[1,0], self.grid[2,0], self.grid[3,0], self.grid[4,0], self.grid[5,0], self.grid[6,0], self.grid[7,0], self.grid[8,0], self.grid[9,0], self.grid[10,0]))
		print("1 | %3d | %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d | %3d |" % (-1, self.grid[0,1], self.grid[1,1], self.grid[2,1], self.grid[3,1], self.grid[4,1], self.grid[5,1], self.grid[6,1], self.grid[7,1], self.grid[8,1], self.grid[9,1], self.grid[10,1]))
		print("2 | %3d | %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d | %3d |" % (-1, self.grid[0,2], self.grid[1,2], self.grid[2,2], self.grid[3,2], self.grid[4,2], self.grid[5,2], self.grid[6,2], self.grid[7,2], self.grid[8,2], self.grid[9,2], self.grid[10,2]))
		print("3 | %3d | %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d | %3d |" % (-1, self.grid[0,3], self.grid[1,3], self.grid[2,3], self.grid[3,3], self.grid[4,3], self.grid[5,3], self.grid[6,3], self.grid[7,3], self.grid[8,3], self.grid[9,3], self.grid[10,3]))
		print("4 | %3d | %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d | %3d |" % (-1, self.grid[0,4], self.grid[1,4], self.grid[2,4], self.grid[3,4], self.grid[4,4], self.grid[5,4], self.grid[6,4], self.grid[7,4], self.grid[8,4], self.grid[9,4], self.grid[10,4]))
		print("5 | %3d | %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d | %3d |" % (-1, self.grid[0,5], self.grid[1,5], self.grid[2,5], self.grid[3,5], self.grid[4,5], self.grid[5,5], self.grid[6,5], self.grid[7,5], self.grid[8,5], self.grid[9,5], self.grid[10,5]))
		print("6 | %3d | %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d | %3d |" % (-1, self.grid[0,6], self.grid[1,6], self.grid[2,6], self.grid[3,6], self.grid[4,6], self.grid[5,6], self.grid[6,6], self.grid[7,6], self.grid[8,6], self.grid[9,6], self.grid[10,6]))
		print("7 | %3d | %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d | %3d |" % (-1, self.grid[0,7], self.grid[1,7], self.grid[2,7], self.grid[3,7], self.grid[4,7], self.grid[5,7], self.grid[6,7], self.grid[7,7], self.grid[8,7], self.grid[9,7], self.grid[10,7]))
		print("8 | %3d | %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d | %3d |" % (-1, self.grid[0,8], self.grid[1,8], self.grid[2,8], self.grid[3,8], self.grid[4,8], self.grid[5,8], self.grid[6,8], self.grid[7,8], self.grid[8,8], self.grid[9,8], self.grid[10,8]))
		print("9 | %3d | %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d | %3d |" % (-1, self.grid[0,9], self.grid[1,9], self.grid[2,9], self.grid[3,9], self.grid[4,9], self.grid[5,9], self.grid[6,9], self.grid[7,9], self.grid[8,9], self.grid[9,9], self.grid[10,9]))
		print("\n")

	def reset(self):		
		self.grid = np.zeros((11,10))


class Neuron():
	def __init__(self, coords, inputNeuron=False, outputNeuron=False, fireRate=None):
		self.coords = coords
		self.activationScore = 0.0
		if inputNeuron:
			self.animate()
			self.plasticity = 'Forwards'
		elif outputNeuron:
			self.animate()
			self.plasticity = 'Backwards'
		else:
			# self.alive = random.choice([True, False])
			self.animate()
			self.plasticity = 'Backwards'
		self.ngf = 1.0 if self.alive else 0.0
		self.synapses = []
		self.fired = False
		self.inputNeuron = inputNeuron
		self.fireRate = fireRate # input neuron only
		self.outputNeuron = outputNeuron

	def animate(self):
		self.alive = True

	def apoptosis(self):
		self.alive = False
		self.ngf = 0.0

	def update(self, ngfPerFire, activationThreshold, plasticityThreshold, plasticityFloor, ngfDecayFactor, time, accuracy):
		#Input neurons fire at their fire rate and are not subject to any thresholds
		if self.inputNeuron:
			self.fired = False
			if time % self.fireRate == 0:
				self.fired = True
				self.ngf += ngfPerFire
		else:
			if not self.alive:
				if self.ngf > 1.0:
					self.animate()
				elif accuracy == 0.0 and random.random() < 0.5: # 
					self.ngf = 2.0
					self.animate()
				else:
					return

			self.fired = False
			for synapse in self.synapses:
				if synapse.fired:
					self.activationScore += synapse.weight

			#Fire if excited
			if self.activationScore > activationThreshold:
				self.fired = True
				self.activationScore = 0.0
				self.ngf += ngfPerFire

				if self.ngf >= plasticityThreshold and not self.outputNeuron:
					#Seek new outputs
					self.plasticity = 'Forwards'
			#Suicide if negative ngf
			elif self.ngf <= 0.0 and not self.outputNeuron:
				self.apoptosis()
				return
			#Seek new inputs
			elif self.ngf < plasticityFloor:
				self.plasticity = 'Backwards'

			#ngf decay 
			if not self.outputNeuron:
				self.ngf -= ngfDecayFactor


class Synapse():
	def __init__(self, inputNeuron=None, outputNeuron=None):
		self.inputNeuron = inputNeuron
		self.outputNeuron = outputNeuron
		self.weight = random.random()
		self.fired = False

	def update(self, chanceToBind, euphamineProbability, weightFactor, synapseKillThreshold):
		self.fired = False
		if self.inputNeuron.fired:
			self.fired = True

			if random.random() > chanceToBind:
				if random.random() < euphamineProbability: # accuracy 
					self.weight += (1.0 - self.weight)*weightFactor
				else:
					self.weight -= (1.0 - self.weight)*weightFactor
		elif self.weight < synapseKillThreshold:
			self.outputNeuron.synapses.remove(self) #should be garbage collected

