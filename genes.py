import random

from managers import IdManager, InnovationManager
from config import Config

config = Config("config.ini", "DEFAULT")

class NeuronGene:
    def __init__(self, layer, neuron_id=None):
        self.id = neuron_id if neuron_id is not None else IdManager.get_new_id()
        self.layer = layer
        if layer == "output":
            self.activation = config.default_output_activation
            self.bias = 0
        elif layer == "hidden":
            self.activation = config.default_hidden_activation
            self.bias = random.uniform(*config.bias_init_range)
        else:
            self.activation = None
            self.bias = None
        self.enabled = True

    def copy(self):
        new_gene = NeuronGene(self.layer, self.id)
        new_gene.activation = self.activation
        new_gene.bias = self.bias
        new_gene.enabled = self.enabled

        return new_gene

class ConnectionGene:
    def __init__(self, from_neuron, to_neuron, connection_id=None):
        self.id = connection_id if connection_id is not None else IdManager.get_new_id()
        self.innovation_number = InnovationManager.get_new_innovation_number()
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = random.uniform(*config.weight_init_range)
        self.enabled = True

    def copy(self, retain_innovation_number=True):
        new_gene = ConnectionGene(self.from_neuron, self.to_neuron, self.id)
        new_gene.weight = self.weight
        new_gene.enabled = self.enabled
        new_gene.innovation_number = self.innovation_number if retain_innovation_number else InnovationManager.get_new_innovation_number()

        return new_gene
