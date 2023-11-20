import random
from neural_network import NeuralNetwork

from torch_activation_functions import ActivationFunctions as activation_functions
from managers import IdManager
from genes import ConnectionGene, NeuronGene
from config import Config

config = Config("config.ini", "DEFAULT")

class Genome:
    def __init__(self):
        self.id = IdManager.get_new_id()
        self.neuron_genes = {}
        self.connection_genes = {}
        self.fitness = None
        self.network = None
        self.network_needs_rebuild = True

    def create(self, input_ids, output_ids, hidden_ids):
        self.add_neurons("input", count=len(input_ids), neuron_ids=input_ids)
        self.add_neurons("output", count=len(output_ids), neuron_ids=output_ids)
        self.add_neurons("hidden", count=len(hidden_ids), neuron_ids=hidden_ids)
        # Attempt to create initial connections
        max_possible_conn = config.hidden_neurons * (config.input_neurons + config.hidden_neurons + config.output_neurons)
        attempts = min(config.initial_conn_attempts, max_possible_conn * config.attempts_to_max_factor)
        self.attempt_connections(from_layer=None, to_layer=None, attempts=attempts)

        return self

    def add_neurons(self, layer, count, neuron_ids=None):
        if neuron_ids is None:
            neuron_ids = [IdManager.get_new_id() for _ in range(count)]

        for neuron_id in neuron_ids:
            new_neuron = NeuronGene(layer, neuron_id)
            self.neuron_genes[neuron_id] = new_neuron
            #print(f"Added neuron {neuron_id} to genome {self.id} in layer {layer}")

    def attempt_connections(self, from_layer=None, to_layer=None, attempts=1):
        #print(f"Attempting {attempts} connections for genome {self.id}...")
        for _ in range(attempts):
            from_neurons = []
            to_neurons = []

            if from_layer and to_layer:
                # Use the provided layers
                from_neurons = [neuron for neuron in self.neuron_genes.values() if neuron.layer == from_layer]
                to_neurons = [neuron for neuron in self.neuron_genes.values() if neuron.layer == to_layer]
            else:
                # Randomly select layers based on architecture rules
                from_layers = ["input", "hidden"]
                attempting_from_layer = random.choice(from_layers)
                if attempting_from_layer == "input":
                    attempting_to_layer = "hidden"
                else:
                    attempting_to_layer = random.choice(["hidden", "output"])
                from_neurons = [neuron for neuron in self.neuron_genes.values() if neuron.layer == attempting_from_layer]
                to_neurons = [neuron for neuron in self.neuron_genes.values() if neuron.layer == attempting_to_layer]

            # Check if there are neurons available for connection
            if not from_neurons or not to_neurons:
                continue

            # Randomly select from neuron and to neuron
            from_neuron = random.choice(from_neurons)
            to_neuron = random.choice(to_neurons)

            # Check if the connection already exists
            existing_connection = any(
                conn.from_neuron == from_neuron.id and conn.to_neuron == to_neuron.id
                for conn in self.connection_genes.values()
            )

            # Before creating a new connection, log the neurons being connected
            #print(f"Attempting to connect: From Neuron ID {from_neuron.id} (Layer: {from_neuron.layer}) to To Neuron ID {to_neuron.id} (Layer: {to_neuron.layer})")
            # Create connection if it doesn"t exist
            if not existing_connection:
                new_connection = ConnectionGene(from_neuron.id, to_neuron.id)
                self.connection_genes[new_connection.id] = new_connection
                #print(f"Added connection in genome {self.id} from {from_neuron.layer} to {to_neuron.layer}")

    def crossover(self, other_genome):
        offspring = Genome()

        # Inherit all input neurons from one parent (e.g., self)
        input_neuron_ids = [neuron_id for neuron_id, neuron in self.neuron_genes.items() if neuron.layer == "input"]
        for neuron_id in input_neuron_ids:
            offspring.neuron_genes[neuron_id] = self.neuron_genes[neuron_id].copy()

        # Inherit output neurons, combining properties from both parents
        output_neuron_ids = [neuron_id for neuron_id, neuron in self.neuron_genes.items() if neuron.layer == "output"]
        for neuron_id in output_neuron_ids:
            if random.random() < 0.5:
                offspring.neuron_genes[neuron_id] = self.neuron_genes[neuron_id].copy()
            else:
                offspring.neuron_genes[neuron_id] = other_genome.neuron_genes[neuron_id].copy()

        # Handle connection genes (both matching and disjoint/excess genes)
        genes1 = {gene.innovation_number: gene for gene in self.connection_genes.values()}
        genes2 = {gene.innovation_number: gene for gene in other_genome.connection_genes.values()}
        all_innovations = set(genes1.keys()) | set(genes2.keys())

        if self.fitness is None and other_genome.fitness is None:
            more_fit_parent = None
        elif self.fitness is None:
            more_fit_parent = other_genome
        elif other_genome.fitness is None:
            more_fit_parent = self
        else:
            more_fit_parent = self if self.fitness > other_genome.fitness else other_genome if self.fitness < other_genome.fitness else None

        for innovation_number in all_innovations:
            gene1 = genes1.get(innovation_number)
            gene2 = genes2.get(innovation_number)

            offspring_gene = None
            if gene1 and gene2:  # Matching genes
                # Determine the gene state based on the more fit parent
                if more_fit_parent:
                    # Inherit the gene state from the more fit parent
                    parent_gene = gene1 if more_fit_parent == self else gene2
                    offspring_gene = parent_gene.copy(retain_innovation_number=True)
                else:
                    # If fitness is equal, optionally handle disabled state
                    if not gene1.enabled or not gene2.enabled:
                        probability_to_disable = config.matching_disabled_connection_chance
                        offspring_gene = random.choice([gene1, gene2]).copy(retain_innovation_number=True)
                        offspring_gene.enabled = False if random.random() < probability_to_disable else offspring_gene.enabled
                    else:
                        offspring_gene = random.choice([gene1, gene2]).copy(retain_innovation_number=True)

            elif gene1 or gene2:  # Disjoint or excess genes
                if more_fit_parent:
                    # Choose the gene from the more fit parent
                    parent_gene = gene1 if more_fit_parent == self else gene2
                    offspring_gene = parent_gene.copy(retain_innovation_number=True) if parent_gene else None
                else:
                    # For equal fitness, randomly choose between gene1, gene2 (which might already be None)
                    offspring_gene = random.choice([gene1, gene2])
                    if offspring_gene:
                        offspring_gene = offspring_gene.copy(retain_innovation_number=True)

            if offspring_gene:
                offspring.connection_genes[offspring_gene.id] = offspring_gene

        # Inherit hidden neurons referenced in connection genes
        hidden_neuron_ids = set()
        for conn_gene in offspring.connection_genes.values():
            from_neuron_id, to_neuron_id = conn_gene.from_neuron, conn_gene.to_neuron

            # Check if the neuron IDs are hidden neurons in either parent
            if (self.neuron_genes.get(from_neuron_id) and self.neuron_genes[from_neuron_id].layer == "hidden") or \
            (other_genome.neuron_genes.get(from_neuron_id) and other_genome.neuron_genes[from_neuron_id].layer == "hidden"):
                hidden_neuron_ids.add(from_neuron_id)

            if (self.neuron_genes.get(to_neuron_id) and self.neuron_genes[to_neuron_id].layer == "hidden") or \
            (other_genome.neuron_genes.get(to_neuron_id) and other_genome.neuron_genes[to_neuron_id].layer == "hidden"):
                hidden_neuron_ids.add(to_neuron_id)

        for neuron_id in hidden_neuron_ids:
            # Randomly choose the parent from which to inherit each hidden neuron
            if neuron_id in self.neuron_genes and neuron_id in other_genome.neuron_genes:
                chosen_parent = self if random.random() < 0.5 else other_genome
                offspring.neuron_genes[neuron_id] = chosen_parent.neuron_genes[neuron_id].copy()
            elif neuron_id in self.neuron_genes:
                offspring.neuron_genes[neuron_id] = self.neuron_genes[neuron_id].copy()
            elif neuron_id in other_genome.neuron_genes:
                offspring.neuron_genes[neuron_id] = other_genome.neuron_genes[neuron_id].copy()

        return offspring

    def mutate(self):
        if random.random() < config.connection_add_chance:
            self.mutate_add_connection()

        if random.random() < config.neuron_add_chance:
            self.mutate_add_neuron()

        if random.random() < config.weight_mutate_chance:
            self.mutate_weight()

        if random.random() < config.bias_mutate_chance:
            self.mutate_bias()

        if random.random() < config.activation_mutate_chance:
            self.mutate_activation_function()

        if random.random() < config.connection_toggle_chance:
            self.mutate_connection_toggle()

        if random.random() < config.neuron_toggle_chance:
            self.mutate_neuron_toggle()

        self.network_needs_rebuild = True

    def mutate_add_connection(self):
            connection_attempts = [
                lambda: self.attempt_connections(from_layer="input", to_layer="hidden", attempts=1),
                lambda: self.attempt_connections(from_layer="hidden", to_layer="hidden", attempts=1),
                lambda: self.attempt_connections(from_layer="hidden", to_layer="output", attempts=1)
            ]
            random.choice(connection_attempts)()

    def mutate_add_neuron(self):
        # Ensure there are enabled connection genes to split
        enabled_genes = [gene for gene in self.connection_genes.values() if gene.enabled]
        if not enabled_genes:
            print("No enabled connections to split for genome", self.id)
            return

        # Choose a random enabled connection gene to split
        gene_to_split = random.choice(enabled_genes)
        
        # Disable the chosen connection gene
        gene_to_split.enabled = False

        # Create a new neuron gene in the hidden layer
        new_neuron = NeuronGene("hidden")
        self.neuron_genes[new_neuron.id] = new_neuron

        # Create two new connection genes
        # Connection from the start of the original connection to the new neuron
        new_connection1 = ConnectionGene(gene_to_split.from_neuron, new_neuron.id)
        self.connection_genes[new_connection1.id] = new_connection1

        # Connection from the new neuron to the end of the original connection
        new_connection2 = ConnectionGene(new_neuron.id, gene_to_split.to_neuron)
        self.connection_genes[new_connection2.id] = new_connection2

        # Set weight of the new connections
        new_connection1.weight = 1  # or some other desired initialization
        new_connection2.weight = gene_to_split.weight  # inherit weight from the split gene

        print(f"Added neuron {new_neuron.id} to genome {self.id} and split connection {gene_to_split.id} into connections {new_connection1.id} and {new_connection2.id}")

    def mutate_weight(self):
        # first choose a random connection gene to mutate
        gene_to_mutate = random.choice(list(self.connection_genes.values()))
        # then choose whether to perturb or set the weight
        if random.random() < config.weight_perturb_vs_set_chance:
            gene_to_mutate.weight = random.uniform(-1, 1) * config.weight_mutate_factor * gene_to_mutate.weight
            #print(f"Perturbed weight for genome {self.id} by factor {config.weight_mutate_factor} to {gene_to_mutate.weight}")
        else:
            gene_to_mutate.weight = random.uniform(*config.weight_init_range)
            #print(f"Set weight of neuron {gene_to_mutate.id} in genome {self.id} to {gene_to_mutate.weight}")

    def mutate_bias(self):
        # first choose a random not input layer neuron gene to mutate
        gene_to_mutate = random.choice([gene for gene in self.neuron_genes.values() if gene.layer != "input"])
        # then choose whether to perturb or set the bias
        if random.random() < config.bias_perturb_vs_set_chance:
            gene_to_mutate.bias = random.uniform(-1, 1) * config.bias_mutate_factor * gene_to_mutate.bias
            #print(f"Perturbed bias of neuron {gene_to_mutate.id} in genome {self.id} by factor {config.bias_mutate_factor} to {gene_to_mutate.bias}")
        else:
            gene_to_mutate.bias = random.uniform(*config.bias_init_range)
            #print(f"Set bias of neuron {gene_to_mutate.id} in genome {self.id} to {gene_to_mutate.bias}")

    def mutate_activation_function(self):
        available_functions = activation_functions.get_activation_functions()
        gene_to_mutate = random.choice([gene for gene in self.neuron_genes.values() if gene.layer == "hidden"])
        gene_to_mutate.activation = random.choice(available_functions)
        print(f"Set activation function of neuron {gene_to_mutate.id} in genome {self.id} to {gene_to_mutate.activation}")

    def mutate_connection_toggle(self):
        gene_to_mutate = random.choice([gene for gene in self.connection_genes.values()])
        gene_to_mutate.enabled = not gene_to_mutate.enabled
        print(f"Toggled connection {gene_to_mutate.id} from neuron {gene_to_mutate.from_neuron} to neuron {gene_to_mutate.to_neuron} in genome {self.id} to {gene_to_mutate.enabled}")
    
    def mutate_neuron_toggle(self):
        # Find all enabled hidden neurons
        enabled_hidden_neurons = [gene for gene in self.neuron_genes.values() if gene.layer == "hidden" and gene.enabled]
        
        # Check if there is more than one enabled hidden neuron
        if len(enabled_hidden_neurons) > 1:
            # Choose a random enabled hidden neuron to mutate
            gene_to_mutate = random.choice(enabled_hidden_neurons)
            gene_to_mutate.enabled = not gene_to_mutate.enabled
            print(f"Toggled neuron {gene_to_mutate.id} in genome {self.id} to {gene_to_mutate.enabled}")
        else:
            print(f"No other enabled hidden neurons to toggle in genome {self.id}")

    def copy(self):
        # Creates a new genome with a new ID
        new_genome = Genome()
        # Copying all attributes except id to new_genome
        new_genome.neuron_genes = self.neuron_genes
        new_genome.connection_genes = self.connection_genes
        new_genome.fitness = self.fitness

        return new_genome

    def build_network(self):
        print(f"Building network for genome {self.id}...")
        if self.network_needs_rebuild:
            self.network = NeuralNetwork(self)
            self.network_needs_rebuild = False
        #print(f"Network built for genome {self.id}")
        return self.network

    def calculate_genetic_distance(self, other_genome):
        # Mapping innovation numbers to connection genes
        inno_to_conn_gene1 = {gene.innovation_number: gene for gene in self.connection_genes.values()}
        inno_to_conn_gene2 = {gene.innovation_number: gene for gene in other_genome.connection_genes.values()}

        # Highest innovation numbers in each genome
        max_inno1 = max(inno_to_conn_gene1.keys(), default=0)
        max_inno2 = max(inno_to_conn_gene2.keys(), default=0)

        disjoint_genes = excess_genes = matching_genes = weight_diff = activation_diff = 0

        # Calculate genetic distance based on connection genes
        for inno_num in set(inno_to_conn_gene1.keys()).union(inno_to_conn_gene2.keys()):
            in_gene1 = inno_num in inno_to_conn_gene1
            in_gene2 = inno_num in inno_to_conn_gene2

            if in_gene1 and in_gene2:
                # Count as matching gene
                matching_genes += 1
                weight_diff += abs(inno_to_conn_gene1[inno_num].weight - inno_to_conn_gene2[inno_num].weight)
            elif in_gene1:
                if inno_num <= max_inno2:
                    disjoint_genes += 1
                else:
                    excess_genes += 1
            elif in_gene2:
                if inno_num <= max_inno1:
                    disjoint_genes += 1
                else:
                    excess_genes += 1

        # Calculate genetic distance based on neuron activation functions
        for neuron_id in set(self.neuron_genes.keys()).union(other_genome.neuron_genes.keys()):
            neuron1 = self.neuron_genes.get(neuron_id)
            neuron2 = other_genome.neuron_genes.get(neuron_id)

            if neuron1 and neuron2:
                activation_diff += neuron1.activation != neuron2.activation

        # Normalize weight and activation differences
        if matching_genes > 0:
            weight_diff /= matching_genes
            activation_diff /= matching_genes

        N = max(len(inno_to_conn_gene1), len(inno_to_conn_gene2))
        distance = ((config.disjoint_coefficient * disjoint_genes) + \
                   (config.excess_coefficient * excess_genes) + \
                   (config.activation_diff_coefficient * activation_diff)) / N + \
                   (config.weight_diff_coefficient * weight_diff)

        #print(f"Genome {self.id} vs {other_genome.id} - Distance: {distance}")

        return distance
