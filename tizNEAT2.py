# check code for the NeuralNetwork (RNN)
# check wich object we are keeping between generations, like previous instance of genomes, species, networks, neurons, connections, genes, etc... and which of these must be deleted to free memory

import multiprocessing
import random
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import pickle

config = {

    "input_neurons": 24,
    "hidden_neurons": 1,
    "output_neurons": 4,
    "initial_conn_attempts": 15, # max possible connections = hidden_neurons * (input_neurons + hidden_neurons + output_neurons)
    "refractory_factor": 0.33,

    "generations": 10,
    "population_size": 10,

    "elites_per_species": 2,
    "max_stagnation": 20,
    "target_species": 25,

    "compatibility_threshold": 10,
    "distance_adj_factor": 0.2,
    "disjoint_coefficient": 1,
    "excess_coefficient": 1,
    "weight_diff_coefficient": 1,
    "activation_diff_coefficient": 1,

    "allow_interspecies_mating": True,
    "interspecies_mating_count": 10,

    "keep_best_percentage": 0.5,

    "neuron_add_chance": 0.01,
    "neuron_toggle_chance": 0.0075,
    "bias_mutate_chance": 0.1,
    "bias_mutate_factor": 0.5,
    "bias_init_range": (-2, 2),

    "activation_mutate_chance": 0.1,
    "default_hidden_activation": "clipped_relu",
    "default_output_activation": "identity",
    "relu_clip_at": 1,

    "gene_add_chance": 0.02,
    "gene_toggle_chance": 0.001,
    "weight_mutate_chance": 0.1,
    "weight_mutate_factor": 0.5,
    "weight_init_range": (-2, 2),

    "parallelize": False,
    "parallelization": 6,

    "global_mutation_enable": False,
    "global_mutation_chance": 0.5,
    "population_save_interval": 10
}

class ActivationFunctions:

    @staticmethod
    def get_activation_functions():
        return [
            "identity", "relu", "leaky_relu", "clipped_relu",
            "tanh", "sigmoid", "softplus", "abs"
        ]

    @staticmethod
    def identity(x):
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
        environment = gym.make("BipedalWalker-v3", hardcore=True, render_mode="human")
        environment = gym.wrappers.TimeLimit(environment, max_episode_steps=200)

        neural_network = genome.build_network()
        for neuron_id, neuron in neural_network.neurons.items():
            print(f"Neuron {neuron_id}, Layer: {neuron.layer}, Bias: {neuron.bias}")
            for conn in neuron.input_connections:
                print(f"  - Connection from {conn.from_neuron.id}, Weight: {conn.weight}")
        observation_tuple = environment.reset()
        observation = observation_tuple[0]  # Extract the array part of the tuple
        #print("Observation size:", len(observation))
        done = False
        total_reward = 0

        while not done:
            outputs = neural_network.forward_pass(observation)
            action = np.array(outputs[-1]).flatten()  # Flatten and convert to NumPy array
            #print(action)
            observation, reward, terminated, truncated, info = environment.step(action)  # Use 'action' here
            total_reward += reward
            print("Tot. reward:", total_reward, "- Terminated:", terminated, "- Truncated:", truncated)
            done = terminated or truncated

        environment.close()
        return total_reward

    def evaluate(self):
        if config["parallelize"]:
            self.evaluate_parallel()
        else:
            self.evaluate_serial()
    
    def evaluate_serial(self):
        fitness_scores = []
        for genome in self.genomes.values():
            fitness_scores.append(self.evaluate_single_genome(genome))

    def evaluate_parallel(self):
        fitness_scores = []
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
        if not self.genomes:
            raise ValueError("No genomes in the species to copy from.")

        random_genome = random.choice(list(self.genomes.values()))
        return random_genome.copy()  # This will now use the modified copy method

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
        self.add_neurons("hidden", config["hidden_neurons"])
        max_possible_conn = config["hidden_neurons"] * (config["input_neurons"] + config["hidden_neurons"] + config["output_neurons"])
        attempts = min(config["initial_conn_attempts"], max_possible_conn)
        self.attempt_connections(from_layer=None, to_layer=None, attempts=attempts)
        return self

    def copy(self):
        new_genome = Genome()  # Creates a new genome with a new ID
        new_genome.network_needs_rebuild = self.network_needs_rebuild
        new_genome.fitness = self.fitness
        new_genome.shared_fitness = self.shared_fitness

        # Copying all neuron genes
        for neuron_id, neuron_gene in self.neuron_genes.items():
            new_genome.neuron_genes[neuron_id] = neuron_gene.copy()

        # Copying all connection genes
        for conn_id, conn_gene in self.connection_genes.items():
            new_genome.connection_genes[conn_id] = conn_gene.copy()

        return new_genome

    def add_neurons(self, layer, count):
        for _ in range(count):
            # Create a new neuron and add it to the neuron_genes dictionary
            new_neuron = NeuronGene(layer)
            self.neuron_genes[new_neuron.id] = new_neuron

    def attempt_connections(self, from_layer=None, to_layer=None, attempts=1):
        for _ in range(attempts):
            from_neurons = []
            to_neurons = []

            if from_layer and to_layer:
                # Use the provided layers
                from_neurons = [neuron for neuron in self.neuron_genes.values() if neuron.layer == from_layer]
                to_neurons = [neuron for neuron in self.neuron_genes.values() if neuron.layer == to_layer]
            else:
                # Randomly select layers based on architecture rules
                from_layers = ['input', 'hidden']
                from_layer = random.choice(from_layers)
                if from_layer == 'input':
                    to_layer = 'hidden'
                elif from_layer == 'hidden':
                    to_layer = random.choice(['hidden', 'output'])
                
                from_neurons = [neuron for neuron in self.neuron_genes.values() if neuron.layer == from_layer]
                to_neurons = [neuron for neuron in self.neuron_genes.values() if neuron.layer == to_layer]

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

            # Create connection if it doesn't exist
            if not existing_connection:
                new_connection = ConnectionGene(from_neuron.id, to_neuron.id)
                self.connection_genes[new_connection.id] = new_connection

    def copy(self):
        pass
       
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
        for conn_gene in self.connection_genes.values():
            if random.random() < config["weight_mutate_factor"]:
                perturb = random.uniform(-1, 1) * config["weight_mutate_factor"]
                conn_gene.weight += perturb
            else:
                conn_gene.weight = random.uniform(*config["weight_init_range"])


    def mutate_bias(self):
        for neuron_gene in self.neuron_genes.values():
            if random.random() < config["bias_mutate_chance"]:
                perturb = random.uniform(-1, 1) * config["bias_mutate_factor"]
                neuron_gene.bias += perturb
            else:
                neuron_gene.bias = random.uniform(*config["bias_init_range"])


    def mutate_activation_function(self):
        available_functions = ActivationFunctions.get_activation_functions()

        for neuron_gene in self.neuron_genes.values():
            if random.random() < config["activation_mutate_chance"]:
                neuron_gene.activation = random.choice(available_functions)


    def mutate_gene_toggle(self):
        for conn_gene in self.connection_genes.values():
            if random.random() < config["gene_toggle_chance"]:
                conn_gene.enabled = not conn_gene.enabled


    def mutate_neuron_toggle(self):
        for neuron_gene in self.neuron_genes.values():
            if random.random() < config["neuron_toggle_chance"]:
                neuron_gene.enabled = not neuron_gene.enabled

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

        # Determine the more fit parent, or None if they have equal fitness
        more_fit_parent = None
        if self.fitness > other_genome.fitness:
            more_fit_parent = self
        elif self.fitness < other_genome.fitness:
            more_fit_parent = other_genome

        for innovation_number in all_innovations:
            gene1 = genes1.get(innovation_number)
            gene2 = genes2.get(innovation_number)

            offspring_gene = None
            if gene1 and gene2:  # Matching genes
                offspring_gene = random.choice([gene1, gene2]).copy(retain_innovation_number=True)
            elif gene1 or gene2:  # Disjoint or excess genes
                if more_fit_parent:
                    parent_gene = genes1.get(innovation_number) if more_fit_parent == self else genes2.get(innovation_number)
                    if parent_gene:
                        offspring_gene = parent_gene.copy(retain_innovation_number=True)
                else:  # Fitness is equal, choose randomly
                    parent_gene = gene1 if gene1 else gene2
                    offspring_gene = parent_gene.copy(retain_innovation_number=True)

            if offspring_gene:
                offspring.connection_genes[offspring_gene.id] = offspring_gene

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
        self.weight = random.uniform(*config["weight_init_range"])
        self.enabled = True

    def copy(self, retain_innovation_number=True):
        new_gene = ConnectionGene(self.from_neuron, self.to_neuron)
        new_gene.weight = self.weight
        new_gene.enabled = self.enabled
        new_gene.innovation_number = self.innovation_number if retain_innovation_number else InnovationManager.get_new_innovation_number()
        return new_gene

class NeuronGene:
    def __init__(self, layer):
        self.id = IdManager.get_new_id()
        self.layer = layer
        self.activation = self.activation = config["default_hidden_activation"] if self.layer == "hidden" else (config["default_output_activation"] if self.layer == "output" else "identity")
        self.bias = random.uniform(*config["bias_init_range"])
        self.enabled = True

    def copy(self):
        new_gene = NeuronGene(self.layer)
        new_gene.activation = self.activation
        new_gene.bias = self.bias
        new_gene.enabled = self.enabled
        return new_gene

class NeuralNetwork:
    def __init__(self, genome):
        self.genome = genome
        self.neurons = {}  # key: neuron id, value: Neuron object
        self.input_neurons = []
        self.output_neurons = []
        self.initialize_network()

    def initialize_network(self):
        # Create Neurons from NeuronGenes
        for neuron_id, neuron_gene in self.genome.neuron_genes.items():
            if neuron_gene.enabled:
                activation_func = getattr(ActivationFunctions, neuron_gene.activation)
                neuron = Neuron(neuron_id, neuron_gene.layer, activation_func, neuron_gene.bias)
                self.neurons[neuron_id] = neuron
                if neuron_gene.layer == 'input':
                    self.input_neurons.append(neuron)
                elif neuron_gene.layer == 'output':
                    self.output_neurons.append(neuron)

        # Create Connections from ConnectionGenes
        for _, conn_gene in self.genome.connection_genes.items():
            if conn_gene.enabled:
                from_neuron = self.neurons[conn_gene.from_neuron]
                to_neuron = self.neurons[conn_gene.to_neuron]
                connection = Connection(from_neuron, to_neuron, conn_gene.weight)
                print(f"Creating connection from neuron {conn_gene.from_neuron} to neuron {conn_gene.to_neuron} with weight {conn_gene.weight}")
                to_neuron.add_input_connection(connection)
        
    def forward_pass(self, inputs, num_timesteps=1):
        outputs = []
        saved_state = {neuron_id: 0 for neuron_id in self.neurons}  # Initial state

        for _ in range(num_timesteps):
            # Update input neurons with current inputs
            for i, neuron in enumerate(self.input_neurons):
                neuron.value = inputs[i]
                saved_state[neuron.id] = inputs[i]

            # Propagate through network
            for neuron_id, neuron in self.neurons.items():
                if neuron.layer != 'input':
                    neuron.calculate_activation(saved_state)
                    saved_state[neuron_id] = neuron.value

            # Collect outputs for this timestep
            timestep_output = [neuron.value for neuron in self.output_neurons]
            outputs.append(timestep_output)

        return outputs

class Neuron:
    def __init__(self, id, layer, activation_function, bias):
        self.id = id
        self.layer = layer
        self.activation_function = activation_function
        self.bias = bias
        self.value = 0
        self.input_connections = []  # List of Connection objects

    def add_input_connection(self, connection):
        self.input_connections.append(connection)
        print(f"Added connection from neuron {connection.from_neuron.id} to neuron {self.id}")

    def calculate_activation(self, saved_state):
        refractory_factor = config["refractory_factor"]
        weighted_sum = sum(conn.weight * saved_state[conn.from_neuron.id] for conn in self.input_connections) + self.bias

        # Apply refractory scaling to previous state's value
        if self.layer != 'input':  # Assuming input neurons don't have refractory behavior
            weighted_sum += self.value * refractory_factor

        self.value = self.activation_function(weighted_sum)

class Connection:
    def __init__(self, from_neuron, to_neuron, weight):
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = weight

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