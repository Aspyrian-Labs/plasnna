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
  'activation_threshold' = 1.0
  'chance_to_bind' = 0.5
  'weight_factor' = 0.25
  }


class Plasma():
	def __init__(self, inputs, outputs, gridSize=(100, 100, 100)):
		self.plasmaGrid = {}
		for x in range(gridSize[0]):
			for y in range(gridSize[1]):
				for z in range(gridSize[2]):
					coords = (x, y, z)
					self.plasmaGrid[coords] = Neuron(coords=coords)
		for x in range(inputs[0]):
			for y in range(inputs[1]):
				coords = (x, y, -1)
				self.plasmaGrid[coords] = Neuron(coords=coords, inputNeuron=True)
		for x in range(outputs[0]):
			for y in range(outputs[1]):
				coords = (x, y, gridSize[2]+1)
				self.plasmaGrid[coords] = Neuron(coords=coords, outputNeuron=True)
        
	def evolve(self, timeSteps=1000, rewardObservationLength=0.1, xData, yData, evolveParameters):
		# Fire neurons
		for neuron in self.plasmaGrid:
			neuron = self.plasmaGrid[neuron]
			neuron.update(evolveParameters['activation_threshold'])
	  
	        # Propagate signal through synapses
		for neuron in self.plasmaGrid:
			neuron = self.plasmaGrid[neuron]
	  	for synapse in neuron.synapses:
	  		synapse.update(evolveParameters['change_to_bind'],
				accuracy, #euphamine probability
				evolveParameters['weight_factor'])
		return accuracy


class Neuron():
	def __init__(self, coords, inputNeuron=False, outputNeuron=False):
		self.coords = coords
		self.alive = random.choice([True, False])
  		self.ngf = 1.0 if self.alive else 0.0
  		self.activationScore = 0.0
  		self.plasticity = None
  		self.synapses = []
  		self.fired = False
    
	def animate(self):
		self.alive = True
		self.ngf = 1.0
  
	def apoptosis(self):
		self.alive = False
		self.ngf = 0.0
    
	def update(self, activationThreshold):
		self.fired = False
		for synapse in self.synapses:
			if synapse.fired:
  			self.activationScore += synapse.weight
    
		if self.activationScore > activationThreshold:
  			self.fired = True
  			self.activationScore = 0.0
      

class Synapse():
	def __init__(self, inputNeuron=None, outputNeuron=None):
		self.inputNeuron = inputNeuron
		self.outputNeuron = outputNeuron
		self.weight = random.random()
		self.fired = False
  
	def update(self, chanceToBind, euphamineProbability, weightFactor):
		self.fired = False
		if self.inputNeuron.fired:
			self.fired = True
      
			if random.random() > chanceToBind:
				if random.random() < euphamineProbability:
					weight += (1.0 - weight)*weightFactor
          
    
    
    
    
