import random

from torch_activation_functions import ActivationFunctions as activation_functions
from managers import IdManager, InnovationManager
from config import Config

config = Config("config.ini", "DEFAULT")

class NeuronGene:
    def __init__(self, layer, neuron_id=None):
        self.id = neuron_id if neuron_id is not None else IdManager.get_new_id()
        self.layer = layer
        if layer == "output":
            self.activation = config.default_output_activation
            self.bias = random.uniform(*config.bias_init_range)
            self.bias = 0
        elif layer == "hidden":
            self.activation = random.choice(activation_functions.get_activation_functions())
            self.bias = random.uniform(*config.bias_init_range)
        else:
            self.activation = "identity"
            self.bias = 0
        self.enabled = True

    def copy(self, keep_id=True):
        new_gene = NeuronGene(self.layer)
        new_gene.activation = self.activation
        new_gene.bias = self.bias
        new_gene.enabled = self.enabled
        new_gene.id = (self.id if keep_id else new_gene.id)

        return new_gene
class ConnectionGene:
    def __init__(self, from_neuron, to_neuron, connection_innovation=None):
        self.innovation = connection_innovation if connection_innovation is not None else InnovationManager.get_new_innovation()
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = random.uniform(*config.weight_init_range)
        self.enabled = True

    def copy(self, keep_innovation=True):
        new_gene = ConnectionGene(self.from_neuron, self.to_neuron)
        new_gene.weight = self.weight
        new_gene.enabled = self.enabled
        new_gene.innovation = (self.innovation if keep_innovation else new_gene.innovation)

        return new_gene
