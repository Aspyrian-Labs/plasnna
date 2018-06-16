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
		#Fire neurons
		for neuron in self.plasmaGrid:
			neuron = self.plasmaGrid[neuron]
			neuron.update(evolveParameters['activation_threshold'],
				evolveParameters['plasticity_threshold'],
				evolveParameters['plasticity_floor'])
	  
	    #Propagate signal through synapses
		for neuron in self.plasmaGrid:
			neuron = self.plasmaGrid[neuron]
			for synapse in neuron.synapses:
				synapse.update(evolveParameters['neurotransmitter_binding_chance'],
					accuracy, #euphamine probability
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
  
	def apoptosis(self):
		self.alive = False
		self.ngf = 0.0
    
	def update(self, activationThreshold, plasticityThreshold, plasticityFloor):
		if not self.alive:
			if self.ngf > 1.0:
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

  			if self.ngf >= plasticityThreshold:
  				#Seek new outputs
  				self.plasticity = 'Forwards'
  		#Suicide if negative ngf
  		elif self.ngf <= 0.0:
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
				if random.random() < euphamineProbability:
					self.weight += (1.0 - self.weight)*weightFactor
				else:
					self.weight -= (1.0 - self.weight)*weightFactor
		elif self.weight < synapseKillThreshold:
			self.outputNeuron.synapses.remove(self) #should be garbage collected

