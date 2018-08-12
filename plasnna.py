'''
Plastic neural network architecture (plasnna - pronounced plasma).

Basic documentation:
	Classes:
	
	Variables:
		ngf: Nerve growth factor. Consumed for a neuron to survive and produced when a neuron fires.
	
'''

import numpy as np
import random 

evolveParameters = {
  'neurotransmitter_binding_chance' = 0.5,
  'activation_threshold' = 1.0,
  'ngf_per_fire' = 0.1,
  'weight_factor' = 0.25,
  'plasticity_threshold' = 1.5,
  'plasticity_floor' = 0.2,
  'neighbour_range' = 2,
  'synapse_kill_threshold' = 0.2,
  'spill_threshold' = 2.0
  }

class Plasma():
	def __init__(self, inputs, outputs, gridSize=(100, 100, 100)):
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
				coords = (x, y, gridSize[2]+1)
				self.plasmaGrid[coords] = Neuron(coords=coords, outputNeuron=True)
				self.outputCoords.append(coords)
        
	def evolve(self, timeSteps=1000, rewardObservationLength=0.1, xData, yData, evolveParameters):
		#Iterate over batch
		for i, dat in enumerate(xData):
			#Convert data to fireRate for input layer (assume normalised data)
			assert xData[0].shape == self.inputs, 'Input data shape must match input layer shape.'
			for c in self.inputCoords:
				self.plasmaGrid[c].rate = int(1.0/xData[i][c[0]][c[1]]) #! currently cutoff scales with timeSteps
																		   #! requires 2D data
			#Set up output recording (!NB: for classifiers only for now)
			assert yData[0].shape == self.outputs, 'Output data shape must match output layer shape.'
			outputRecord = {}
			for c in self.outputCoords:
				outputRecord[c] = [yData[i], 0] #increment counter for each correct classification

			#Start time evolution
			currentAccuracy = 0
			accuracyRecord = []
			for t in range(timeSteps):
				#Initial growth promotion: output layer requests connections
				if currentAccuracy == 0.0:
					for x in range(self.outputs[0]):
						for y in range(self.outputs[1]):
							coords = (x, y, self.gridSize[2]+1)
							self.plasmaGrid[coords].ngf = 5.0 #enough to spill into preceeding layers

				#Fire neurons
				for neuron in self.plasmaGrid:
					neuron = self.plasmaGrid[neuron]
					neuron.update(evolveParameters['activation_threshold'],
						evolveParameters['plasticity_threshold'],
						evolveParameters['plasticity_floor'],
						t)
			  
			    #Propagate signal through synapses
				for neuron in self.plasmaGrid:
					neuron = self.plasmaGrid[neuron]
					for synapse in neuron.synapses:
						synapse.update(evolveParameters['neurotransmitter_binding_chance'],
							currentAccuracy, #euphamine probability
							evolveParameters['weight_factor'],
							evolveParameters['synapse_kill_threshold'])

				neighbourRange = evolveParameters['neighbour_range']
				for neuron in self.plasmaGrid:
					neuronObj = self.plasmaGrid[neuron]
					if neuronObj.plasticity == 'Forwards' or neuronObj.ngf > evolveParameters['spill_threshold']:
						#Collect nearest neighbours
						synapseCandidates = []
						spillCandidates = []
						for x in range(neuron[0] - neighbourRange, neuron[0] + neighbourRange):
							for y in range(neuron[1] - neighbourRange, nweuron[1] + neighbourRange):
								for z in range(neuron[2] - neighbourRange, neuron[2] + neighbourRange):
									neighbour = self.plasmaGrid[(x,y,z)]
									if neighbour.plasticity == 'Backwards':
										synapseCandidates.append((x,y,z))
									if neighbour.ngf < neuronObj.ngf:
										spillCandidates.append((x,y,z))
						
						if neuronObj.plasticity == 'Forwards':
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
								candidate.ngf += availableNgf * (candidate.ngf / totalNgf)

				#Verify output
				totalCorrect = 0
				for i, c in enumerate(self.outputCoords):
					if self.plasmaGrid[c].fired and outputRecord[c][0]:
						outputRecord[c][1] += 1
						totalCorrect += 1
				currentAccuracy = float(totalCorrect)/len(self.outputCoords)
				accuracyRecord.append(currentAccuracy)

			#After time evolution:
			self.accuracy = sum(accuracyRecord)/len(accuracyRecord)	
		return self.accuracy


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
  			self.alive = random.choice([True, False])
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
    
	def update(self, activationThreshold, plasticityThreshold, plasticityFloor, time):
		#Input neurons fire at their fire rate and are not subject to any thresholds
		if self.inputNeuron:
			self.fired = False
			if time % self.fireRate == 0:
				self.fired = True
				self.ngf += ngf_per_fire
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
	  			self.ngf += ngf_per_fire

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

