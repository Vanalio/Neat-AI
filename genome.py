import random

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

    def create(self, input_ids, output_ids, hidden_ids):
        self.add_neurons("input", count=len(input_ids), neuron_ids=input_ids)
        self.add_neurons("output", count=len(output_ids), neuron_ids=output_ids)
        self.add_neurons("hidden", count=len(hidden_ids), neuron_ids=hidden_ids)

        self.add_connections(from_layer=None, to_layer=None, count=config.initial_connections)

        return self

    def add_neurons(self, layer, count, neuron_ids=None):
        if neuron_ids is None:
            neuron_ids = [IdManager.get_new_id() for _ in range(count)]

        for neuron_id in neuron_ids:
            new_neuron = NeuronGene(layer, neuron_id)
            self.neuron_genes[neuron_id] = new_neuron

    def add_connections(self, from_layer=None, to_layer=None, count=1):
        
        max_possible_conn = config.hidden_neurons * (
            config.input_neurons + config.hidden_neurons + config.output_neurons
        )

        #max_possible_conn = number of enabled hidden neurons * (config.input_neurons + number of enabled hidden neurons + config.output_neurons)
        enabled_hidden_neurons = len([n for n in self.neuron_genes.values() if n.layer == "hidden" and n.enabled])
        max_possible_conn = enabled_hidden_neurons * (config.input_neurons + enabled_hidden_neurons + config.output_neurons)

        attempts = min(
            config.connection_attempts,
            max_possible_conn * config.max_to_attempts_factor,
        )

        for _ in range(attempts):
            from_neurons = []
            to_neurons = []

            if from_layer and to_layer:

                from_neurons = [
                    neuron
                    for neuron in self.neuron_genes.values()
                    if neuron.layer == from_layer
                ]
                to_neurons = [
                    neuron
                    for neuron in self.neuron_genes.values()
                    if neuron.layer == to_layer
                ]
            else:

                from_layers = ["input", "hidden"]
                attempting_from_layer = random.choice(from_layers)
                if attempting_from_layer == "input":
                    attempting_to_layer = "hidden"
                else:
                    attempting_to_layer = random.choice(["hidden", "output"])
                from_neurons = [
                    neuron
                    for neuron in self.neuron_genes.values()
                    if neuron.layer == attempting_from_layer
                ]
                to_neurons = [
                    neuron
                    for neuron in self.neuron_genes.values()
                    if neuron.layer == attempting_to_layer
                ]

            if not from_neurons or not to_neurons:
                continue

            from_neuron = random.choice(from_neurons)
            to_neuron = random.choice(to_neurons)

            existing_connection = any(
                conn.from_neuron == from_neuron.id and conn.to_neuron == to_neuron.id
                for conn in self.connection_genes.values()
            )

            if not existing_connection:
                new_connection = ConnectionGene(from_neuron.id, to_neuron.id)
                self.connection_genes[new_connection.id] = new_connection

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
                if more_fit_parent:

                    parent_gene = gene1 if more_fit_parent == self else gene2
                    offspring_gene = (
                        parent_gene.copy(retain_innovation_number=True)
                        if parent_gene
                        else None
                    )
                else:

                    offspring_gene = random.choice([gene1, gene2])
                    if offspring_gene:
                        offspring_gene = offspring_gene.copy(
                            retain_innovation_number=True
                        )

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
            lambda: self.add_connections(from_layer="input", to_layer="hidden"),
            lambda: self.add_connections(from_layer="hidden", to_layer="hidden"),
            lambda: self.add_connections(from_layer="hidden", to_layer="output")
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

        print(
            f"Added neuron {new_neuron.id} to genome {self.id} and split connection {gene_to_split.id} into connections {new_connection1.id} and {new_connection2.id}"
        )

    def mutate_weight(self):

        gene_to_mutate = random.choice(list(self.connection_genes.values()))

        if random.random() < config.weight_perturb_vs_set_chance:
            gene_to_mutate.weight = (
                random.uniform(-1, 1)
                * config.weight_mutate_factor
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
                random.uniform(-1, 1) * config.bias_mutate_factor * gene_to_mutate.bias
            )

        else:
            gene_to_mutate.bias = random.uniform(*config.bias_init_range)

    def mutate_activation_function(self):
        available_functions = activation_functions.get_activation_functions()
        gene_to_mutate = random.choice(
            [gene for gene in self.neuron_genes.values() if gene.layer == "hidden"]
        )
        gene_to_mutate.activation = random.choice(available_functions)
        print(
            f"Set activation function of neuron {gene_to_mutate.id} in genome {self.id} to {gene_to_mutate.activation}"
        )

    def mutate_connection_toggle(self):
        gene_to_mutate = random.choice(
            [gene for gene in self.connection_genes.values()]
        )
        gene_to_mutate.enabled = not gene_to_mutate.enabled
        print(
            f"Toggled connection {gene_to_mutate.id} from neuron {gene_to_mutate.from_neuron} to neuron {gene_to_mutate.to_neuron} in genome {self.id} to {gene_to_mutate.enabled}"
        )

    def mutate_neuron_toggle(self):

        enabled_hidden_neurons = [
            gene
            for gene in self.neuron_genes.values()
            if gene.layer == "hidden" and gene.enabled
        ]

        if len(enabled_hidden_neurons) > 1:

            gene_to_mutate = random.choice(enabled_hidden_neurons)
            gene_to_mutate.enabled = not gene_to_mutate.enabled
            print(
                f"Toggled neuron {gene_to_mutate.id} in genome {self.id} to {gene_to_mutate.enabled}"
            )
        else:
            print(f"No other enabled hidden neurons to toggle in genome {self.id}")

    def copy(self):

        new_genome = Genome()

        new_genome.neuron_genes = self.neuron_genes
        new_genome.connection_genes = self.connection_genes
        new_genome.fitness = self.fitness

        return new_genome

    def calculate_genetic_distance(self, other_genome):

        inno_to_conn_gene1 = {
            gene.innovation_number: gene for gene in self.connection_genes.values()
        }
        inno_to_conn_gene2 = {
            gene.innovation_number: gene
            for gene in other_genome.connection_genes.values()
        }

        max_inno1 = max(inno_to_conn_gene1.keys(), default=0)
        max_inno2 = max(inno_to_conn_gene2.keys(), default=0)

        disjoint_genes = (
            excess_genes
        ) = matching_genes = weight_diff = activation_diff = 0

        for inno_num in set(inno_to_conn_gene1.keys()).union(inno_to_conn_gene2.keys()):
            in_gene1 = inno_num in inno_to_conn_gene1
            in_gene2 = inno_num in inno_to_conn_gene2

            if in_gene1 and in_gene2:

                matching_genes += 1
                weight_diff += abs(
                    inno_to_conn_gene1[inno_num].weight
                    - inno_to_conn_gene2[inno_num].weight
                )
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

        for neuron_id in set(self.neuron_genes.keys()).union(
            other_genome.neuron_genes.keys()
        ):
            neuron1 = self.neuron_genes.get(neuron_id)
            neuron2 = other_genome.neuron_genes.get(neuron_id)

            if neuron1 and neuron2:
                activation_diff += neuron1.activation != neuron2.activation

        if matching_genes > 0:
            weight_diff /= matching_genes
            activation_diff /= matching_genes

        N = max(len(inno_to_conn_gene1), len(inno_to_conn_gene2))
        distance = ((config.disjoint_coefficient * disjoint_genes) + (config.excess_coefficient * excess_genes)) / N + (config.activation_diff_coefficient * activation_diff) + (config.weight_diff_coefficient * weight_diff)

        #print(
            #f"Genetic distance between genome {self.id} and genome {other_genome.id} is {distance}"
        #)

        return distance
