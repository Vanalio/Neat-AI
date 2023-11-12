# implement and use a random selector functions for each class except population, like random_neuron, random_connection, random_gene, random_genome, random_species, random_network
# check if at each new generation we are keeping somewhere any unneeded objects, like previous instance of genomes, species, networks, neurons, connections, genes, etc.

import multiprocessing
import random
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import pickle

config = {

    "input_neurons": 24,
    "output_neurons": 4,
    "initial_conn_attempts": 50,
    "refractory_factor": 0.25,

    "TESTING_EVOLUTION_INPUT": (
        (1.0, 2.0, 3.0, 4.0),
        (-5.2, -4.2, -3.2, -2.2),
        (50000.0, 50001.0, 50002, 50003.0),
        (-1724.0, -1723.0, -1722.0, -1721.0),
        (101.0, 102.0, 103.0, 104.0),
        (1200000.0, 1200001.0, 1200002, 1200003.0, 1200004.0, 1200005.0, 1200006, 1200007.0)
        ),
    "TESTING_EVOLUTION_OUTPUT": (
        (5.0, 6.0),
        (-1.2, -0.2),
        (50004.0, 50005.0),
        (-1720.0, -1719.0),
        (105.0, 106.0),
        (1200008.0, 1200009.0)
        ),
    
    "TESTING_REAL_INPUT": (31200, 31201),
    "TESTING_REAL_OUTPUT": (31202, 31203),

    "generations": 60,
    "population_size": 5000,

    "elites_per_species": 2,
    "max_stagnation": 20,
    "target_species": 50,

    "compatibility_threshold": 10,
    "distance_adj_factor": 0.2,
    "disjoint_coefficient": 1,
    "excess_coefficient": 1,
    "weight_diff_coefficient": 1,
    "activation_diff_coefficient": 1,

    "allow_interspecies_mating": True,
    "interspecies_mating_count": 10,

    "keep_best_percentage": 66,

    "neuron_add_chance": 0.01,
    "neuron_toggle_chance": 0.0075,
    "bias_mutate_chance": 0.1,
    "bias_mutate_factor": 0.5,
    "bias_init_range": (-2, 2),

    "activation_mutate_chance": 0.1,

    "gene_add_chance": 0.02,
    "gene_toggle_chance": 0.001,
    "weight_mutate_chance": 0.1,
    "weight_mutate_factor": 0.5,
    "weight_init_range": (-2, 2),

    "default_activation": "clipped_relu",
    "relu_clip_at": 1,
    "samples_per_batch": 10,
    "sample_size_range": (120, 1200),
    "parallelization": 6,
    "global_mutation_enable": False,
    "global_mutation_chance": 0.5,
    "population_save_interval": 10
}

class ActivationFunctions:
    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def leaky_relu(x):
        return np.where(x > 0, x, x * 0.01)

    @staticmethod
    def clipped_relu(x):
        return np.clip(x, 0, config["relu_clip_at"])

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softplus(x):
        return np.log1p(np.exp(x))

    @staticmethod
    def abs(x):
        return np.abs(x)

class IdManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(IdManager, cls).__new__(cls, *args, **kwargs)
            cls._instance.current_id = 0
        return cls._instance

    @staticmethod
    def get_new_id():
        instance = IdManager()
        instance.current_id += 1
        return instance.current_id

class InnovationManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(InnovationManager, cls).__new__(cls, *args, **kwargs)
            cls._instance.current_innovation = 0
        return cls._instance

    @staticmethod
    def get_new_innovation_number():
        instance = InnovationManager()
        instance.current_innovation += 1
        return instance.current_innovation

class Population:
    def __init__(self, first=False):
        self.id = IdManager.get_new_id()
        self.genomes = {}
        self.species = {}
        self.average_fitness = 0
        self.max_fitness = 0
        self.best_genome = 0
        if first:
            self._first_population()

    def _first_population(self):
        for _ in range(config["population_size"]):
            genome = Genome().create()
            self.genomes[genome.id] = genome

    def evaluate_single_genome(self, genome):
        environment = gym.make("BipedalWalker-v3", hardcore=True)
        environment = gym.wrappers.TimeLimit(environment, max_episode_steps=2000)

        neural_network = genome.build_network()
        observation = environment.reset()
        done = False
        total_reward = 0

        while not done:
            outputs = neural_network.forward_pass(observation)
            observation, reward, terminated, truncated, info = environment.step(outputs)
            total_reward += reward
            done = terminated or truncated

        environment.close()
        return total_reward

    def evaluate(self):
        # Create a multiprocessing pool
        #with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:

        with multiprocessing.Pool(config["parallelization"]) as pool:
            # Parallelize the evaluation of genomes
            fitness_scores = pool.map(self.evaluate_single_genome, self.genomes.values())


        # Assign fitness scores back to genomes
        for genome, fitness in zip(self.genomes.values(), fitness_scores):
            genome.fitness = fitness
    ####################################################################

    def speciate(self):  # add id to species # FIXME #

        self.species = {}
        for genome in self.genomes:
            placed_in_species = False                         # before trying to add to a specied, consider not in any species
            for species_instance in self.species:             # for each Species instance
                if species_instance.is_same_species(genome):  # if genome fit the species
                    species_instance.add_genome(genome)       # add genome to species
                    placed_in_species = True                  # and set is as assigned
                    break
            if not placed_in_species:
                new_species = Species()               # create a new species for the genome
                new_species.add_genome(genome)        # add genome to it
                self.species.append(new_species)      # add species to the list of species

        species_ratio = len(self.species) / config["target_species"]
        if species_ratio < 1.0:
            config["compatibility_threshold"] *= (1.0 - config["distance_adj_factor"])
        elif species_ratio > 1.0:
            config["compatibility_threshold"] *= (1.0 + config["distance_adj_factor"])

    def remove_species(self, removal_condition, message):
        initial_species_count = len(self.species)
        self.species = [spec for spec in self.species if not removal_condition(spec)]
        removed_count = initial_species_count - len(self.species)
        if removed_count:
            print(f"Removed {removed_count} {message}")

    def prune_species(self):
        stale_threshold = config["max_stagnation"]
        is_stale = lambda spec: spec.generations_without_improvement > stale_threshold
        self.remove_species(is_stale, "stale species")

        total_avg_fitness = sum(spec.average_fitness for spec in self.species)
        threshold = config["target_species"]
        is_weak = lambda spec: (spec.average_fitness / total_avg_fitness * len(self.genomes)) < threshold
        self.remove_species(is_weak, "weak species")

    def assess(self):
        total_fitness = 0
        self.max_fitness = 0
        self.best_genome = None

        for genome in self.genomes:
            total_fitness += genome.fitness
            if genome.fitness > self.max_fitness:
                self.max_fitness = genome.fitness
                self.best_genome = genome

        # Update best and average fitness in the population
        self.average_fitness = total_fitness / len(self.genomes) if self.genomes else 0

        # Assess each species
        for species in self.species:
            # Calculate the average fitness of the species
            species.average_fitness = sum(genome.fitness for genome in species.genomes) / len(species.genomes) if species.genomes else 0

            # Find elites in the species
            species.genomes.sort(key=lambda x: x.fitness, reverse=True)
            species.elites = species.genomes[:config["elites_per_species"]] if species.genomes else []

    def survive_and_reproduce(self):
        next_gen_genomes = []
        for spec in self.species:
            next_gen_genomes.extend(spec.elites[:config["elites_per_species"]]) # add elites to next gen
            spec.cull(keep_best=True) # order by fitness and keep only a quote
            offspring_count = self.get_offspring_count(spec)
            offspring = spec.produce_offspring(offspring_count)
            next_gen_genomes.extend(offspring)
        if config["allow_interspecies_mating"]:
            interspecies_offspring = self.produce_interspecies_offspring()
            next_gen_genomes.extend(interspecies_offspring)
        while len(next_gen_genomes) < config["population_size"]:
            next_gen_genomes.append(self.random_species().produce_offspring(1))
        self.genomes = next_gen_genomes

    def get_offspring_count(self, species):
        total_average_fitness = sum(spec.average_fitness for spec in self.species)
        return int((species.average_fitness / total_average_fitness) * config["population_size"])

    def produce_interspecies_offspring(self):
        offspring = []
        for _ in range(config["interspecies_mating_count"]):
            species_1 = self.random_species()
            species_2 = self.random_species()
            if species_1 != species_2:
                parent_1 = species_1.random_genome()
                parent_2 = species_2.random_genome()
                child = parent_1.crossover(parent_2)
                offspring.append(child)
        return offspring

    def random_species(self):
        return random.choice(self.species)

    def save_genomes_to_file(self, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(self.genomes, file)

    def load_genomes_from_file(self, file_path):
        with open(file_path, "rb") as file:
            self.genomes = pickle.load(file)

    def evolve(self):
        self.evaluate()
        self.speciate()
        self.prune_species()
        self.assess()
        self.survive_and_reproduce()

class Species:
    def __init__(self):
        self.id = IdManager.get_new_id()
        self.genomes = {}
        self.elites = {}
        self.representative = None
        self.max_shared_fitness = float('-inf')  # Highest shared fitness in the species
        self.age = 0
        self.generations_without_improvement = 0

    def add_genome(self, genome):
        self.genomes[genome.id] = genome
        if not self.representative:
            self.representative = genome

    def cull(self, keep_best_percentage=0.5):
        if not 0 < keep_best_percentage <= 1:
            raise ValueError("keep_best_percentage must be between 0 and 1.")

        # Sort genomes by fitness and update genomes dictionary
        sorted_genomes = sorted(self.genomes.values(), key=lambda genome: genome.fitness, reverse=True)
        cutoff = int(len(sorted_genomes) * keep_best_percentage)
        self.genomes = {genome.id: genome for genome in sorted_genomes[:cutoff]}

        # Update elites
        self.elites = {genome.id: genome for genome in sorted_genomes[:min(len(sorted_genomes), 2)]}

    def produce_offspring(self, offspring_count=1):
        offspring = []
        for _ in range(offspring_count):
            parent1, parent2 = random.sample(list(self.genomes.values()), 2)  # Randomly select two different parents
            child = parent1.crossover(parent2)
            child.mutate()
            offspring.append(child)
        return offspring

    def random_genome(self):
        return random.choice(list(self.genomes.values()))

    def is_same_species(self, genome):
        distance = genome.calculate_genetic_distance(self.representative)
        return distance < config["compatibility_threshold"]

class Genome:
    def __init__(self):
        self.id = IdManager.get_new_id()
        self.neuron_genes = {}
        self.connection_genes = {}
        self.network = None
        self.network_needs_rebuild = True
        self.fitness = 0
        self.shared_fitness = 0

    def create(self):
        self.add_neurons("input", config["input_neurons"])
        self.add_neurons("output", config["output_neurons"])
        max_possible_conn = config["input_neurons"] * config["output_neurons"]
        attempts = min(config["initial_conn_attempts"], max_possible_conn)
        self.attempt_connections("input", "output", attempts)
        return self

    def add_neurons(self, layer, count):
        for _ in range(count):
            NeuronGene(layer)

    def attempt_connections(self, from_layer, to_layer, attempts=1):
        for _ in range(attempts):
            # randomly select 2 neurons and look for the couple in connections dict
            # if connection doesn't already exist:
            ConnectionGene(from_layer, to_layer)
    ##########################################        
    def mutate(self):
        if random.random() < config["gene_add_chance"]:
            self.attempt_connections()
        if random.random() < config["neuron_add_chance"]:
            self.add_neurons("hidden", 1)
        if random.random() < config["weight_mutate_chance"]:
            self.mutate_weight()
        if random.random() < config["bias_mutate_chance"]:
            self.mutate_bias()
        if random.random() < config["activation_mutate_chance"]:
            self.mutate_activation_function()
        if random.random() < config["gene_toggle_chance"]:
            self.mutate_gene_toggle()
        if random.random() < config["neuron_toggle_chance"]:
            self.mutate_neuron_toggle()
        self.network_needs_rebuild = True

    def mutate_weight(self):
        pass

    def mutate_bias(self):
        pass

    def mutate_activation_function(self):
        pass

    def mutate_gene_toggle(self):
        pass

    def mutate_neuron_toggle(self):
        pass

    def build_network(self):
        if self.network_needs_rebuild:
            self.network = NeuralNetwork(self)
            self.network_needs_rebuild = False
        return self.network

    def crossover(self, other_genome):
        offspring = Genome()

        genes1 = {gene.innovation_number: gene for gene in self.connection_genes}
        genes2 = {gene.innovation_number: gene for gene in other_genome.connection_genes}

        all_innovations = set(genes1.keys()) | set(genes2.keys())

        for innovation_number in all_innovations:
            gene1 = genes1.get(innovation_number)
            gene2 = genes2.get(innovation_number)

            if gene1 and gene2:

                offspring_gene = random.choice([gene1, gene2]).copy()
            elif gene1:

                offspring_gene = gene1.copy()
            elif gene2:

                offspring_gene = gene2.copy()

            offspring.connection_genes.append(offspring_gene)

        return offspring

    def calculate_genetic_distance(self, other_genome):

        genes1 = sorted(self.connection_genes, key=lambda g: g.innovation_number)
        genes2 = sorted(other_genome.connection_genes, key=lambda g: g.innovation_number)

        i = j = 0
        disjoint_genes = excess_genes = matching_genes = weight_diff = 0

        while i < len(genes1) and j < len(genes2):
            gene1 = genes1[i]
            gene2 = genes2[j]

            if gene1.innovation_number == gene2.innovation_number:
                matching_genes += 1
                weight_diff += abs(gene1.weight - gene2.weight)
                i += 1
                j += 1
            elif gene1.innovation_number < gene2.innovation_number:
                disjoint_genes += 1
                i += 1
            else:
                disjoint_genes += 1
                j += 1

        excess_genes = len(genes1[i:]) + len(genes2[j:])
        weight_diff /= matching_genes if matching_genes else 1

        N = max(len(genes1), len(genes2))
        distance = (config.disjoint_coefficient * disjoint_genes / N) + (config.weight_diff_coefficient * weight_diff)

        return distance

class ConnectionGene:
    def __init__(self, from_neuron, to_neuron):
        self.id = IdManager.get_new_id()
        self.innovation_number = InnovationManager.get_new_innovation_number()
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = 1
        self.enabled = True

    def copy(self):
        pass

class NeuronGene:
    def __init__(self, layer):
        self.id = IdManager.get_new_id()
        self.layer = layer
        self.activation = config["default_activation"]
        self.bias = random.uniform(*config["bias_init_range"])
        self.enabled = True

class NeuralNetwork:
    def __init__(self, genome):
        self.id = IdManager.get_new_id()
        self.genome = genome
        self.neurons = []
        self.input_neurons = []
        self.output_neurons = []
        self.connections = []
        self.initialize_network()

    def initialize_network(self):

        self.neurons = {neuron.innovation_number: 0.0 for neuron in self.genome.neuron_genes}
        self.input_neurons = [neuron for neuron in self.genome.neuron_genes if neuron.layer == "input"]
        self.output_neurons = [neuron for neuron in self.genome.neuron_genes if neuron.layer == "output"]
        self.connections = {gene.from_neuron: [] for gene in self.genome.connection_genes if gene.enabled}

        for gene in self.genome.connection_genes:
            if gene.enabled:
                self.connections[gene.from_neuron].append((gene.to_neuron, gene.weight))

    def forward_pass(self, inputs, num_timesteps):
        # first hidden states to zero
        hidden_states = {neuron.innovation_number: 0.0 for neuron in self.genome.neuron_genes if neuron.layer == "hidden"}

        # Store the previous timestep's hidden states
        prev_hidden_states = {neuron.innovation_number: 0.0 for neuron in self.genome.neuron_genes if neuron.layer == "hidden"}

        for _ in range(num_timesteps):
            # Update input neurons with current inputs
            for i, neuron in enumerate(self.input_neurons):
                hidden_states[neuron.innovation_number] = inputs[i]

            # Compute new hidden states based on current inputs, recurrent connections, and self-connections
            for neuron in self.hidden_neurons:
                total_input = self.bias[neuron.innovation_number]
                for input_neuron, weight in self.connections[neuron.innovation_number]:
                    if input_neuron == neuron.innovation_number:  # Self-connection
                        total_input += prev_hidden_states[neuron.innovation_number] * weight
                    else:
                        total_input += hidden_states[input_neuron] * weight

                # Get the activation function from the NeuronGene
                activation_function = self.get_activation_function(neuron.activation)
                hidden_states[neuron.innovation_number] = activation_function(total_input)

            # Update prev_hidden_states for the next timestep
            prev_hidden_states.update(hidden_states)

        # Compute outputs from output neurons
        outputs = []
        for neuron in self.output_neurons:
            total_input = self.bias[neuron.innovation_number]
            for input_neuron, weight in self.connections[neuron.innovation_number]:
                total_input += hidden_states[input_neuron] * weight

            activation_function = self.get_activation_function(neuron.activation)
            outputs.append(activation_function(total_input))

        return outputs

    def get_activation_function(self, name):
        # Lookup the actual function in the ActivationFunctions class
        return getattr(ActivationFunctions, name)

    def compute_hidden_states(self, prev_hidden_states):
        current_hidden_states = {}
        # Calculate the state for each hidden neuron
        for neuron in self.hidden_neurons:
            # Sum of inputs from input neurons and recurrent connections from hidden neurons
            total_input = sum(
                self.neurons[input_neuron] * weight
                for input_neuron, weight in self.connections[neuron.innovation_number]
            )
            # Add recurrent input from previous time step
            total_input += prev_hidden_states[neuron.innovation_number] * self.get_self_weight(neuron)
            # Apply activation function
            current_hidden_states[neuron.innovation_number] = self.activation_function(total_input)
        return current_hidden_states

    def compute_outputs(self, hidden_states):
        outputs = []
        # Calculate the output for each output neuron
        for neuron in self.output_neurons:
            total_input = sum(
                hidden_states[input_neuron] * weight
                for input_neuron, weight in self.connections[neuron.innovation_number]
            )
            # Apply activation function
            outputs.append(self.activation_function(total_input))
        return outputs

    def get_self_weight(self, neuron):
        # Return the weight of the self-connection for the neuron, if it exists
        self_connection = next((conn for conn in self.connections[neuron.innovation_number] if conn[0] == neuron.innovation_number), None)
        return self_connection[1] if self_connection else 0

class Visualization:
    def __init__(self):
        self.id = IdManager.get_new_id()

    def plot_species_count(self, data):
        plt.plot(data)
        plt.title("Species Count")
        plt.xlabel("Generation")
        plt.ylabel("Number of Species")
        plt.show()

    def plot_max_fitness(self, data):
        plt.plot(data)
        plt.title("Max Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.show()

    def visualize_network(self, genome):
        pass

def NEAT_run():
    id_manager = IdManager()
    innovation_manager = InnovationManager()
    population = Population(first=True)
    visualizer = Visualization()
    species_data = []
    fitness_data = []

    for generation in range(config["generations"]):
    
        population.evolve()
    
        species_data.append(len(population.species))
        fitness_data.append(population.max_fitness)
        visualizer.plot_species_count(species_data)
        visualizer.plot_max_fitness(fitness_data)
    
        if generation % config["population_save_interval"] == 0:
            population.save_genomes_to_file(f"population_gen_{generation}.pkl")
    
    population.save_genomes_to_file("final_population.pkl")
    visualizer.visualize_network(population.best_genome)
    print("Objects created:", InnovationManager.get_new_innovation_number())

if __name__ == "__main__":
    NEAT_run()