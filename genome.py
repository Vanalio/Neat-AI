import random
import pickle

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
        #print(f"Enabled hidden neurons: {enabled_hidden_neurons}")
        # Count current connections
        #current_connections = len(self.connection_genes)
        #print(f"Current connections: {current_connections}")
        max_total_connections = (enabled_hidden_neurons * (config.input_neurons + enabled_hidden_neurons + config.output_neurons))
        #print(f"Max total connections: {max_total_connections}")

        return max_total_connections

    def max_attempts(self):
        max_attempts = int(self.max_total_connections() * config.max_to_attempts_factor)
        #print(f"Max attempts: {max_attempts}")

        return max_attempts

    def add_connections(self, from_layer=None, to_layer=None, count=1):
        #print(f"Adding connection to genome {self.id} from layer {from_layer} to layer {to_layer}")
        attempts = 0
        added_connections = 0
        max_attempts = self.max_attempts()

        while added_connections != count and attempts < max_attempts:

            from_neurons = []
            to_neurons = []

            if from_layer and to_layer:
                from_neurons = [neuron for neuron in self.neuron_genes.values() if neuron.layer == from_layer]
                to_neurons = [neuron for neuron in self.neuron_genes.values() if neuron.layer == to_layer]
            else:
                from_layers = ["input", "hidden"]
                attempting_from_layer = random.choice(from_layers)
                if attempting_from_layer == "input":
                    attempting_to_layer = "hidden"
                else:
                    attempting_to_layer = random.choice(["hidden", "output"])

                from_neurons = [neuron for neuron in self.neuron_genes.values() if neuron.layer == attempting_from_layer]
                to_neurons = [neuron for neuron in self.neuron_genes.values() if neuron.layer == attempting_to_layer]

            from_neuron = random.choice(from_neurons)
            to_neuron = random.choice(to_neurons)

            existing_connection = any(conn.from_neuron == from_neuron.id and conn.to_neuron == to_neuron.id for conn in self.connection_genes.values())

            if not existing_connection:
                new_connection = ConnectionGene(from_neuron.id, to_neuron.id)
                self.connection_genes[new_connection.id] = new_connection
                added_connections += 1
                #print(f"Added connections: {added_connections} of {count}")
            #else:
                #print(f"Connection already exists between {from_neuron.id} and {to_neuron.id}")

            attempts += 1
            #print(f"Attempt {attempts} of {max_attempts}")

    def crossover(self, other_genome):
        offspring = Genome()

        input_neuron_ids = [
            neuron_id
            for neuron_id, neuron in self.neuron_genes.items()
            if neuron.layer == "input"
        ]
        for neuron_id in input_neuron_ids:
            offspring.neuron_genes[neuron_id] = self.neuron_genes[neuron_id].copy()

        output_neuron_ids = [
            neuron_id
            for neuron_id, neuron in self.neuron_genes.items()
            if neuron.layer == "output"
        ]
        for neuron_id in output_neuron_ids:
            if random.random() < 0.5:
                offspring.neuron_genes[neuron_id] = self.neuron_genes[neuron_id].copy()
            else:
                offspring.neuron_genes[neuron_id] = other_genome.neuron_genes[
                    neuron_id
                ].copy()

        genes1 = {
            gene.innovation_number: gene for gene in self.connection_genes.values()
        }
        genes2 = {
            gene.innovation_number: gene
            for gene in other_genome.connection_genes.values()
        }
        all_innovations = set(genes1.keys()) | set(genes2.keys())

        if self.fitness is None and other_genome.fitness is None:
            more_fit_parent = None
        elif self.fitness is None:
            more_fit_parent = other_genome
        elif other_genome.fitness is None:
            more_fit_parent = self
        else:
            more_fit_parent = (
                self
                if self.fitness > other_genome.fitness
                else other_genome
                if self.fitness < other_genome.fitness
                else None
            )

        for innovation_number in all_innovations:
            gene1 = genes1.get(innovation_number)
            gene2 = genes2.get(innovation_number)

            offspring_gene = None
            if gene1 and gene2:

                if more_fit_parent:

                    parent_gene = gene1 if more_fit_parent == self else gene2
                    offspring_gene = parent_gene.copy(retain_innovation_number=True)
                else:

                    if not gene1.enabled or not gene2.enabled:
                        probability_to_disable = (
                            config.matching_disabled_connection_chance
                        )
                        offspring_gene = random.choice([gene1, gene2]).copy(
                            retain_innovation_number=True
                        )
                        offspring_gene.enabled = (
                            False
                            if random.random() < probability_to_disable
                            else offspring_gene.enabled
                        )
                    else:
                        offspring_gene = random.choice([gene1, gene2]).copy(
                            retain_innovation_number=True
                        )

            elif gene1 or gene2:
                # If only one parent has the gene (disjoint or excess)
                parent_gene = gene1 if gene1 else gene2
                parent_with_gene = self if gene1 else other_genome

                # Inherit the gene only if the parent with the gene is the fittest
                if more_fit_parent is None or parent_with_gene == more_fit_parent:
                    offspring_gene = parent_gene.copy(retain_innovation_number=True)
                    offspring.connection_genes[offspring_gene.id] = offspring_gene

            if offspring_gene:
                offspring.connection_genes[offspring_gene.id] = offspring_gene

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

            if (
                neuron_id in self.neuron_genes
                and neuron_id in other_genome.neuron_genes
            ):
                chosen_parent = self if random.random() < 0.5 else other_genome
                offspring.neuron_genes[neuron_id] = chosen_parent.neuron_genes[
                    neuron_id
                ].copy()
            elif neuron_id in self.neuron_genes:
                offspring.neuron_genes[neuron_id] = self.neuron_genes[neuron_id].copy()
            elif neuron_id in other_genome.neuron_genes:
                offspring.neuron_genes[neuron_id] = other_genome.neuron_genes[
                    neuron_id
                ].copy()

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

        connection_types = [
            lambda: self.add_connections(from_layer="input", to_layer="hidden", count=1),
            lambda: self.add_connections(from_layer="hidden", to_layer="hidden", count=1),
            lambda: self.add_connections(from_layer="hidden", to_layer="output", count=1)
        ]
        random.choice(connection_types)()

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
        self.connection_genes[new_connection1.id] = new_connection1

        new_connection2 = ConnectionGene(new_neuron.id, gene_to_split.to_neuron)
        self.connection_genes[new_connection2.id] = new_connection2

        new_connection1.weight = 1
        new_connection2.weight = gene_to_split.weight

        #print(
        #    f"Added neuron {new_neuron.id} to genome {self.id} and split connection {gene_to_split.id} into connections {new_connection1.id} and {new_connection2.id}"
        #)

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
            [gene for gene in self.neuron_genes.values() if gene.layer != "input"]
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
        #    f"Toggled connection {gene_to_mutate.id} from neuron {gene_to_mutate.from_neuron} to neuron {gene_to_mutate.to_neuron} in genome {self.id} to {gene_to_mutate.enabled}"
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

    def copy(self):
        new_genome = Genome()

        new_genome.neuron_genes = {neuron_id: gene.copy() for neuron_id, gene in self.neuron_genes.items()}
        new_genome.connection_genes = {conn_id: gene.copy() for conn_id, gene in self.connection_genes.items()}
        new_genome.fitness = self.fitness
        new_genome.species_id = self.species_id

        return new_genome

    def calculate_genetic_distance(self, other_genome):
        # Extract the highest innovation numbers
        self_connection_genes = self.connection_genes.values()
        other_connection_genes = other_genome.connection_genes.values()
        max_inno1 = max((gene.innovation_number for gene in self_connection_genes), default=0)
        max_inno2 = max((gene.innovation_number for gene in other_connection_genes), default=0)
        #print(f"Max innovation numbers: {max_inno1}, {max_inno2}")

        disjoint_genes = excess_genes = matching_genes = weight_diff = activation_diff = 0

        # Create sets of innovation numbers for efficient comparison
        self_inno_set = {gene.innovation_number for gene in self_connection_genes}
        other_inno_set = {gene.innovation_number for gene in other_connection_genes}
        #print(f"Self innovation numbers: {self_inno_set}")
        #print(f"Other innovation numbers: {other_inno_set}")

        # Calculate excess genes
        excess_self = {inno for inno in self_inno_set if inno > max_inno2}
        excess_other = {inno for inno in other_inno_set if inno > max_inno1}
        #print(f"Excess self: {excess_self}")
        #print(f"Excess other: {excess_other}")


        # Calculate disjoint genes (genes not in excess and not in the other genome)
        disjoint_self = self_inno_set - other_inno_set - excess_self
        #print(f"Disjoint self: {disjoint_self}")
        disjoint_other = other_inno_set - self_inno_set - excess_other
        #print(f"Disjoint other: {disjoint_other}")
        disjoint = disjoint_self | disjoint_other
        #print(f"Disjoint: {disjoint}")

        # Process matching genes
        for gene in self_connection_genes:
            inno_num = gene.innovation_number
            if inno_num in other_inno_set:
                matching_genes += 1
                other_gene = other_genome.connection_genes.get(inno_num)
                if other_gene:
                    weight_diff += abs(gene.weight - other_gene.weight)

        # Update counts for disjoint and excess genes
        disjoint_genes = len(disjoint)
        excess_genes = len(excess_self) + len(excess_other)
        #print(f"Disjoint genes: {disjoint_genes} - Excess genes: {excess_genes}")

        # Neuron genes comparison
        self_neuron_genes = self.neuron_genes
        #print(f"Self neuron genes: {self_neuron_genes}")
        other_neuron_genes = other_genome.neuron_genes
        #print(f"Other neuron genes: {other_neuron_genes}")
        common_neurons = self_neuron_genes.keys() & other_neuron_genes.keys()
        #print(f"Common neurons: {common_neurons}")
        activation_diff = sum(self_neuron_genes[neuron_id].activation != other_neuron_genes[neuron_id].activation for neuron_id in common_neurons)

        # Compute average differences if there are matching genes
        if matching_genes > 0:
            weight_diff /= matching_genes
            activation_diff /= matching_genes

        # Calculate the genetic distance
        N = max(len(self_connection_genes), len(other_connection_genes))
        N = max(N, 1)  # Prevent division by zero
        distance = (config.disjoint_coefficient * disjoint_genes + config.excess_coefficient * excess_genes) / N + config.activation_diff_coefficient * activation_diff + config.weight_diff_coefficient * weight_diff
        #print(f"Distance: {distance}")

        return distance

    def save_to_file(genome, filename):
        with open(filename, 'wb') as file:
            pickle.dump(genome, file)

    def load_from_file(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
