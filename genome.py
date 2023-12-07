import random
import pickle
import sys

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
        self.species_id = None
        matching_connections = None

    def create(self, input_ids, output_ids):
        self.add_neurons("input", count=len(input_ids), neuron_ids=input_ids)
        self.add_neurons("output", count=len(output_ids), neuron_ids=output_ids)
        self.add_neurons("hidden", count=config.hidden_neurons, neuron_ids=None)

        initial_connections = int(self.max_total_connections() * config.initial_connections_quota)
        self.add_connections(count=initial_connections)

        return self

    def add_neurons(self, layer, count, neuron_ids=None):
        if neuron_ids is None:
            neuron_ids = [IdManager.get_new_id() for _ in range(count)]

        for neuron_id in neuron_ids:
            new_neuron = NeuronGene(layer, neuron_id)
            self.neuron_genes[neuron_id] = new_neuron

    def max_total_connections(self):
        enabled_hidden_neurons = len([n for n in self.neuron_genes.values() if n.layer == "hidden" and n.enabled])
        max_total_connections = (enabled_hidden_neurons * (config.input_neurons + enabled_hidden_neurons + config.output_neurons))

        return max_total_connections

    def max_attempts(self):
        max_attempts = int(self.max_total_connections() * config.max_to_attempts_factor)
        #print(f"Max attempts: {max_attempts}")

        return max_attempts

    def add_connections(self, from_layer=None, to_layer=None, count=1):
        attempts = 0
        added_connections = 0
        max_attempts = self.max_attempts()

        # Maintain a set of all existing connections
        existing_connections = {(conn.from_neuron, conn.to_neuron) for conn in self.connection_genes.values()}

        while added_connections != count and attempts < max_attempts:

            from_neurons = []
            to_neurons = []

            if from_layer and to_layer:
                from_neurons = [neuron for neuron in self.neuron_genes.values() if neuron.layer == from_layer and neuron.enabled]
                to_neurons = [neuron for neuron in self.neuron_genes.values() if neuron.layer == to_layer and neuron.enabled]

            else:
                from_layers = ["input", "hidden"]
                attempting_from_layer = random.choice(from_layers)
                if attempting_from_layer == "input":
                    attempting_to_layer = "hidden"
                else:
                    attempting_to_layer = random.choice(["hidden", "output"])

                from_neurons = [neuron for neuron in self.neuron_genes.values() if neuron.layer == attempting_from_layer and neuron.enabled]
                to_neurons = [neuron for neuron in self.neuron_genes.values() if neuron.layer == attempting_to_layer and neuron.enabled]

            from_neuron = random.choice(from_neurons)
            to_neuron = random.choice(to_neurons)

            if (from_neuron.id, to_neuron.id) not in existing_connections:
                new_connection = ConnectionGene(from_neuron.id, to_neuron.id)
                self.connection_genes[new_connection.innovation] = new_connection
                added_connections += 1

                # Add the new connection to the set of existing connections
                existing_connections.add((from_neuron.id, to_neuron.id))

            attempts += 1

    def crossover(self, other_genome):
        offspring = Genome()

        input_neuron_ids = [neuron_id for neuron_id, neuron in self.neuron_genes.items() if neuron.layer == "input"]
        for neuron_id in input_neuron_ids:
            offspring.neuron_genes[neuron_id] = self.neuron_genes[neuron_id].copy(keep_id=True)

        output_neuron_ids = [neuron_id for neuron_id, neuron in self.neuron_genes.items() if neuron.layer == "output"]
        for neuron_id in output_neuron_ids:
            if random.random() < 0.5:
                offspring.neuron_genes[neuron_id] = self.neuron_genes[neuron_id].copy(keep_id=True)
            else:
                offspring.neuron_genes[neuron_id] = other_genome.neuron_genes[neuron_id].copy(keep_id=True)

        genes1 = {gene.innovation: gene for gene in self.connection_genes.values()}
        genes2 = {gene.innovation: gene for gene in other_genome.connection_genes.values()}
        all_innovations = set(genes1.keys()) | set(genes2.keys())

        if self.fitness is None and other_genome.fitness is None:
            more_fit_parent = None
        elif self.fitness is None:
            more_fit_parent = other_genome
        elif other_genome.fitness is None:
            more_fit_parent = self
        else:
            more_fit_parent = (self if self.fitness > other_genome.fitness else other_genome if self.fitness < other_genome.fitness else None)

        for innovation in all_innovations:
            gene1 = genes1.get(innovation)
            gene2 = genes2.get(innovation)

            offspring_gene = None

            if gene1 and gene2:

                if gene1.from_neuron == gene2.from_neuron and gene1.to_neuron == gene2.to_neuron:
                    if more_fit_parent:
                        parent_gene = gene1 if more_fit_parent == self else gene2
                        offspring_gene = parent_gene.copy(keep_innovation=True)
                    else:
                        offspring_gene = random.choice([gene1, gene2]).copy(keep_innovation=True)
                else:
                    if more_fit_parent:
                        parent_gene = gene1 if more_fit_parent == self else gene2
                        offspring_gene = parent_gene.copy(keep_innovation=True)
                    else:
                        if not gene1.enabled or not gene2.enabled:
                            offspring_gene = random.choice([gene1, gene2]).copy(keep_innovation=True)
                            offspring_gene.enabled = (False if random.random() < config.matching_disabled_connection_chance else offspring_gene.enabled)
                        else:
                            offspring_gene = random.choice([gene1, gene2]).copy(keep_innovation=True)

            elif gene1 or gene2:
                # If only one parent has the gene (disjoint or excess)
                parent_gene = gene1 if gene1 else gene2
                parent_with_gene = self if gene1 else other_genome

                # Inherit the gene only if the parent with the gene is the fittest
                if more_fit_parent is None or parent_with_gene == more_fit_parent:
                    offspring_gene = parent_gene.copy(keep_innovation=True)
                    offspring.connection_genes[offspring_gene.innovation] = offspring_gene

            if offspring_gene:
                # Add the gene to the offspring
                offspring.connection_genes[offspring_gene.innovation] = offspring_gene

        hidden_neuron_ids = set()

        for conn_gene in offspring.connection_genes.values():
            from_neuron_id, to_neuron_id = conn_gene.from_neuron, conn_gene.to_neuron

            if (
                self.neuron_genes.get(from_neuron_id)
                and self.neuron_genes[from_neuron_id].layer == "hidden"
            ) or (
                other_genome.neuron_genes.get(from_neuron_id)
                and other_genome.neuron_genes[from_neuron_id].layer == "hidden"
            ):
                hidden_neuron_ids.add(from_neuron_id)

            if (
                self.neuron_genes.get(to_neuron_id)
                and self.neuron_genes[to_neuron_id].layer == "hidden"
            ) or (
                other_genome.neuron_genes.get(to_neuron_id)
                and other_genome.neuron_genes[to_neuron_id].layer == "hidden"
            ):
                hidden_neuron_ids.add(to_neuron_id)

        for neuron_id in hidden_neuron_ids:
            if neuron_id in self.neuron_genes and neuron_id in other_genome.neuron_genes:
                if more_fit_parent:
                    chosen_parent = self if more_fit_parent == self else other_genome
                else:
                    chosen_parent = self if random.random() < 0.5 else other_genome
                offspring.neuron_genes[neuron_id] = chosen_parent.neuron_genes[neuron_id].copy(keep_id=True)
            
            elif neuron_id in self.neuron_genes:
                offspring.neuron_genes[neuron_id] = self.neuron_genes[neuron_id].copy(keep_id=True)
            
            elif neuron_id in other_genome.neuron_genes:
                offspring.neuron_genes[neuron_id] = other_genome.neuron_genes[neuron_id].copy(keep_id=True)

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

    def mutate_add_connection(self):

        self.add_connections(count=1)

    def mutate_add_neuron(self):

        enabled_genes = [
            gene for gene in self.connection_genes.values() if gene.enabled
        ]
        if not enabled_genes:
            print("No enabled connections to split for genome", self.id)
            return

        gene_to_split = random.choice(enabled_genes)

        gene_to_split.enabled = False

        new_neuron = NeuronGene("hidden")
        self.neuron_genes[new_neuron.id] = new_neuron

        new_connection1 = ConnectionGene(gene_to_split.from_neuron, new_neuron.id)
        self.connection_genes[new_connection1.innovation] = new_connection1

        new_connection2 = ConnectionGene(new_neuron.id, gene_to_split.to_neuron)
        self.connection_genes[new_connection2.innovation] = new_connection2

        new_connection1.weight = 1
        new_connection2.weight = gene_to_split.weight

    def mutate_weight(self):

        gene_to_mutate = random.choice(list(self.connection_genes.values()))

        if random.random() < config.weight_perturb_vs_set_chance:
            gene_to_mutate.weight = (
                random.uniform(-1, 1)
                * config.max_abs_weight_mutate_factor
                * gene_to_mutate.weight
            )

        else:
            gene_to_mutate.weight = random.uniform(*config.weight_init_range)

    def mutate_bias(self):

        gene_to_mutate = random.choice(
            [gene for gene in self.neuron_genes.values() if gene.layer == "hidden"]
        )

        if random.random() < config.bias_perturb_vs_set_chance:
            gene_to_mutate.bias = (
                random.uniform(-1, 1) * config.max_abs_bias_mutate_factor * gene_to_mutate.bias
            )

        else:
            gene_to_mutate.bias = random.uniform(*config.bias_init_range)

    def mutate_activation_function(self):
        available_functions = activation_functions.get_activation_functions()
        gene_to_mutate = random.choice(
            [gene for gene in self.neuron_genes.values() if gene.layer == "hidden"]
        )
        gene_to_mutate.activation = random.choice(available_functions)
        #print(
        #    f"Set activation function of neuron {gene_to_mutate.id} in genome {self.id} to {gene_to_mutate.activation}"
        #)

    def mutate_connection_toggle(self):
        gene_to_mutate = random.choice(
            [gene for gene in self.connection_genes.values()]
        )
        gene_to_mutate.enabled = not gene_to_mutate.enabled
        #print(
        #    f"Toggled connection {gene_to_mutate.innovation} from neuron {gene_to_mutate.from_neuron} to neuron {gene_to_mutate.to_neuron} in genome {self.id} to {gene_to_mutate.enabled}"
        #)

    def mutate_neuron_toggle(self):

        enabled_hidden_neurons = [
            gene
            for gene in self.neuron_genes.values()
            if gene.layer == "hidden" and gene.enabled
        ]

        if len(enabled_hidden_neurons) > 1:

            gene_to_mutate = random.choice(enabled_hidden_neurons)
            gene_to_mutate.enabled = not gene_to_mutate.enabled
            #print(
            #    f"Toggled neuron {gene_to_mutate.id} in genome {self.id} to {gene_to_mutate.enabled}"
            #)
        else:
            print(f"No other enabled hidden neurons to toggle in genome {self.id}")

    def copy(self, keep_id=True, keep_innovation=True):
        new_genome = Genome()

        new_genome.neuron_genes = {neuron_id: gene.copy(keep_id=keep_id) for neuron_id, gene in self.neuron_genes.items()}
        new_genome.connection_genes = {conn_id: gene.copy(keep_innovation=keep_innovation) for conn_id, gene in self.connection_genes.items()}
        new_genome.fitness = self.fitness
        new_genome.species_id = self.species_id

        return new_genome

    def calculate_genetic_distance(self, other_genome):
        disjoint_connections = excess_connections = matching_connections = common_hidden_neurons = weight_diff = bias_diff = activation_diff = 0
        
        self_connections = self.connection_genes.values()
        other_connections = other_genome.connection_genes.values()
        
        # create sets of innovation numbers
        self_inno = {gene.innovation for gene in self_connections}
        other_inno = {gene.innovation for gene in other_connections}
        
        # find the highest innovation numbers
        max_inno1 = max(self_inno)
        max_inno2 = max(other_inno)

        # create set of matching innovation numbers
        matching_inno = self_inno & other_inno

        # create set of excess innovation numbers
        excess_self_inno = {inno for inno in self_inno if inno > max_inno2}
        excess_other_inno = {inno for inno in other_inno if inno > max_inno1}

        # create sets of disjoint innovation numbers
        disjoint_self_inno = self_inno - other_inno - excess_self_inno
        disjoint_other_inno = other_inno - self_inno - excess_other_inno
        disjoint_inno = disjoint_self_inno | disjoint_other_inno

        # create set of hidden neurons ids
        self_hidden_neurons_id = {neuron_id for neuron_id, neuron in self.neuron_genes.items() if neuron.layer == "hidden"}
        other_hidden_neurons_id = {neuron_id for neuron_id, neuron in other_genome.neuron_genes.items() if neuron.layer == "hidden"}

        # create set of common hidden neurons ids
        common_hidden_neurons_id = self_hidden_neurons_id & other_hidden_neurons_id

        # count size of matching, disjoint and excess genes
        disjoint_connections = len(disjoint_inno)
        excess_connections = len(excess_self_inno) + len(excess_other_inno)
        matching_connections = len(matching_inno)
        
        # count size of common hidden neurons
        common_hidden_neurons = len(common_hidden_neurons_id)

        # calculate total absolute weight differences of matching genes
        for gene in self_connections:
            if gene.innovation in matching_inno:
                other_gene = other_genome.connection_genes[gene.innovation]
                weight_diff += abs(gene.weight - other_gene.weight)
        
        # Compute average weight difference
        if matching_connections > 0:
            weight_diff /= matching_connections

        # calculate total absolute bias and activation differences of common hidden neurons
        for neuron_id in common_hidden_neurons_id:
            self_neuron = self.neuron_genes[neuron_id]
            other_neuron = other_genome.neuron_genes[neuron_id]
            bias_diff += abs(self_neuron.bias - other_neuron.bias)
            # activation diff is 1 if the activation functions are different, 0 otherwise
            activation_diff += 1 if self_neuron.activation != other_neuron.activation else 0
        
        # Compute average bias and activation difference
        if common_hidden_neurons > 0:
            bias_diff /= common_hidden_neurons
            activation_diff /= common_hidden_neurons

        # Calculate the genetic distance
        N = max(len(self_connections), len(other_connections), 1)
        distance = (config.disjoint_coefficient * disjoint_connections + config.excess_coefficient * excess_connections) / N + config.activation_diff_coefficient * activation_diff + config.weight_diff_coefficient * weight_diff + config.bias_diff_coefficient * bias_diff

        return distance, matching_connections

    def save_to_file(genome, filename):
        with open(filename, "wb") as file:
            pickle.dump(genome, file)

    def load_from_file(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)
