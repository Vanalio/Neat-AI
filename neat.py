import random
import matplotlib.pyplot as plt
import networkx as nx
import gymnasium as gym
import pickle
import torch
import torch.nn as nn

import configparser
import ActivationFunctions
import Visualization

import configparser
import re

class Config:
    def __init__(self, filename, section="DEFAULT"):
        self.parser = configparser.ConfigParser()
        self.parser.read(filename)
        if section not in self.parser:
            raise ValueError(f"Section \"{section}\" not found in the configuration file.")
        self.load_config(section)

    def load_config(self, section):
        for key in self.parser[section]:
            value = self.parser.get(section, key)
            setattr(self, key, self.auto_type(value))

    def auto_type(self, value):
        # Check if value is a numeric range
        if re.match(r"^[\s\d.,-]+$", value) and "," in value:
            parts = value.split(",")
            try:
                return tuple(float(part.strip()) if "." in part else int(part.strip()) for part in parts)
            except ValueError:
                pass

        # Check for comma-separated string values
        if "," in value and not re.match(r"^-?\d+(\.\d+)?$", value):
            return tuple(part.strip() for part in value.split(","))

        # Attempt to convert to integer
        try:
            return int(value)
        except ValueError:
            pass

        # Attempt to convert to float
        try:
            return float(value)
        except ValueError:
            pass

        # Attempt to convert to boolean
        if value.lower() in ["true", "yes", "on"]:
            return True
        if value.lower() in ["false", "no", "off"]:
            return False

        # Default to string
        return value

class Population:
    def __init__(self, first=False):
        self.id = IdManager.get_new_id()
        self.genomes = {}
        self.species = {}
        self.average_fitness = None
        self.max_fitness = None
        self.best_genome = None
        if first:
            self._first_population()

    def _first_population(self):
        print("Creating first population...")
        for _ in range(config.population_size):
            genome = Genome().create()
            self.genomes[genome.id] = genome
        print(f"Genomes created for first population: {len(self.genomes)}")
        print("Genomes composed of:")
        for genome in self.genomes.values():
            print(f"Genome ID: {genome.id}, Neurons: {len(genome.neuron_genes)}, Connections: {len(genome.connection_genes)}")

    def evaluate_single_genome(self, genome):
        #print(f"Evaluating genome {genome.id}...")
        #environment = gym.make("BipedalWalker-v3", hardcore=True, render_mode="human")
        environment = gym.make("BipedalWalker-v3", hardcore=True)
        environment = gym.wrappers.TimeLimit(environment, max_episode_steps=100)
        #print("Environment created")
        #print("Creating neural network...")
        neural_network = NeuralNetwork(genome)
        observation = environment.reset()  # Get the initial observation
        if isinstance(observation, tuple):
            observation = observation[0]  # Extract the array part of the tuple
        observation = torch.from_numpy(observation).float()  # Convert to PyTorch tensor

        done = False
        total_reward = 0

        while not done:
            action = neural_network(observation)
            action = action.detach().cpu().numpy().ravel()
            #print("Action values:", action)  # Debug: Print the action values
            observation, reward, terminated, truncated, info = environment.step(action)
            total_reward += reward

            observation = torch.from_numpy(observation).float()  # Convert to PyTorch tensor

            done = terminated or truncated

        environment.close()
        print(f"Genome ID: {genome.id}, Fitness: {total_reward}")
        return total_reward


    def evaluate(self):
        print("Evaluating population...")
        if config.parallelize:
            exit()
            #self.evaluate_parallel()
        else:
            self.evaluate_serial()
    
    def evaluate_serial(self):
        print("Serial evaluation")
        for genome in self.genomes.values():
            genome.fitness = self.evaluate_single_genome(genome)
            #print(f"FROM SERIAL EVALUATE LOOP - Genome ID: {genome.id}, Fitness: {genome.fitness}")

    #def evaluate_parallel(self):
    #    print("Parallel evaluation")
    #    fitness_scores = []
    #    # Create a multiprocessing pool
    #    #with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    #    with multiprocessing.Pool(config.parallelization) as pool:
    #        # Parallelize the evaluation of genomes
    #        fitness_scores = pool.map(self.evaluate_single_genome, self.genomes.values())
    #    # Assign fitness scores back to genomes
    #    for genome, fitness in zip(self.genomes.values(), fitness_scores):
    #        genome.fitness = fitness
    #        print(f"FROM PARALLEL Genome ID: {genome.id}, Fitness: {genome.fitness}")
    ####################################################################

    def speciate(self):
        print("Speciating population...")
        self.species = {}
        for _, genome in self.genomes.items():  # Iterate over genome objects
            placed_in_species = False
            #print(f"Processing genome ID: {genome.id}")

            for species_id, species_instance in self.species.items():  # Iterate over Species objects
                #print(f"Checking against species ID: {species_id}")
                if species_instance.is_same_species(genome):
                    #print(f"Genome {genome.id} is in the same species as the representative")
                    species_instance.add_genome(genome)
                    placed_in_species = True
                    #print(f"Genome {genome.id} placed in existing species {species_id}")
                    break

            if not placed_in_species:
                #print(f"Genome {genome.id} is not in this species")
                new_species = Species()
                #print(f"New species ID: {new_species.id}")
                new_species.add_genome(genome)
                self.species[new_species.id] = new_species
                #print(f"Added new species {new_species.id} to population...")
            #else:
                #print(f"Skipped genome {genome.id}, already placed in a species.")

        species_ratio = len(self.species) / config.target_species
        if species_ratio < 1.0:
            config.distance_threshold *= (1.0 - config.distance_adj_factor)
        elif species_ratio > 1.0:
            config.distance_threshold *= (1.0 + config.distance_adj_factor)
        print(f"Species count: {len(self.species)}, Adjusted compatibility threshold: {config.distance_threshold}")

        # Additional print for debugging: Display all species and their members
        #for species_id, species_instance in self.species.items():
            #print(f"Species ID: {species_id}, Species representatives: {species_instance.representative.id}, Species members: {len(species_instance.genomes)}")

    def remove_species(self, removal_condition, message):
        initial_species_count = len(self.species)
        self.species = {species_id: spec for species_id, spec in self.species.items() if not removal_condition(spec)}
        removed_count = initial_species_count - len(self.species)
        if removed_count:
            print(f"Removed {removed_count} {message}, Total species: {len(self.species)}")

    def prune_species(self):
        print("Extinction of weak and stale species...")

        # Calculate the stale threshold and weak species threshold
        stale_threshold = config.max_stagnation
        weak_threshold = config.weak_threshold

        # Define conditions for stale and weak species
        is_stale = lambda spec: spec.generations_without_improvement > stale_threshold
        is_weak = lambda spec: (spec.average_fitness / self.average_fitness * len(self.genomes)) < weak_threshold

        # Remove stale species if it does not go below the minimum required species
        if len(self.species) - len([spec for spec in self.species.values() if is_stale(spec)]) >= config.min_species:
            self.remove_species(is_stale, "stale species")

        # Remove weak species if it does not go below the minimum required species
        if len(self.species) - len([spec for spec in self.species.values() if is_weak(spec)]) >= config.min_species:
            self.remove_species(is_weak, "weak species")

    def assess(self):
        print("Assessing population...")
        total_fitness = None
        self.max_fitness = None
        self.best_genome = None

        for _, genome in self.genomes.items():
            #print(f"Genome ID: {genome.id}, Fitness: {genome.fitness}")

            if total_fitness is None:
                total_fitness = 0
            total_fitness += genome.fitness

            if self.max_fitness is None or genome.fitness > self.max_fitness:
                self.max_fitness = genome.fitness
                self.best_genome = genome

        if total_fitness is not None:
            self.average_fitness = total_fitness / len(self.genomes)
        else:
            self.average_fitness = None

        print(f"Species: {len(self.species)}, Total fitness: {total_fitness}, Average fitness: {self.average_fitness}, Max fitness: {self.max_fitness}", "Best genome:", self.best_genome.id if self.best_genome else None)

        # Assess each species
        for _, species in self.species.items():
            # Calculate the average fitness of the species
            species.average_fitness = sum(genome.fitness for _, genome in species.genomes.items()) / len(species.genomes) if species.genomes else 0
            # Sort genomes by fitness and update the genomes dictionary in the species
            sorted_genomes = sorted(species.genomes.values(), key=lambda genome: genome.fitness, reverse=True)
            species.genomes = {genome.id: genome for genome in sorted_genomes}

            # Find elites in the species
            species.elites = {genome.id: genome for genome in sorted_genomes[:config.elites_per_species]}
            print(f"Species ID: {species.id}, Average fitness: {species.average_fitness}, Members: {len(species.genomes)}, Elites: {len(species.elites)}")

    def survive_and_reproduce(self):
        next_gen_genomes = {}
        print("Start of survive_and_reproduce, next_gen_genomes:", next_gen_genomes)
        for species_instance in self.species.values():
            print(f"Taking species {species_instance.id} elites to the next generation...")
            next_gen_genomes.update(species_instance.elites)
            #print("After adding elites, next_gen_genomes:", next_gen_genomes)

        for species_instance in self.species.values():
            print(f"Culling species {species_instance.id}...")
            species_instance.cull(config.keep_best_percentage)  # Keep only a portion of the genomes

        for species_instance in self.species.values():
            offspring_count = self.get_offspring_count(species_instance)
            offspring = species_instance.produce_offspring(offspring_count)
            # Check if offspring is a dictionary of Genome objects
            if not all(isinstance(genome, Genome) for genome in offspring.values()):
                raise TypeError("produce_offspring did not return a dictionary of Genome objects")
            next_gen_genomes.update(offspring)
            #print("After offspring, next_gen_genomes:", next_gen_genomes)

        # Handle interspecies offspring
        if config.allow_interspecies_mating:
            interspecies_offspring = self.produce_interspecies_offspring()
            # Check if interspecies_offspring is a dictionary of Genome objects
            if not all(isinstance(genome, Genome) for genome in interspecies_offspring.values()):
                raise TypeError("produce_interspecies_offspring did not return a dictionary of Genome objects")
            next_gen_genomes.extend(interspecies_offspring)
            #print("After interspecies offspring, next_gen_genomes:", next_gen_genomes)

        # Ensure population size is maintained
        while len(next_gen_genomes) < config.population_size:
            next_gen_genomes.update(self.random_species().produce_offspring(1))
        #print("After random species offspring, next_gen_genomes:", next_gen_genomes)

        # Check if next_gen_genomes is a dictionary of Genome objects
        if not all(isinstance(genome, Genome) for genome in next_gen_genomes.values()):
            raise TypeError("next_gen_genomes does not contain only Genome objects")

        # Update genomes for the next generation
        #print(f"Before updating genomes, next_gen_genomes: {next_gen_genomes}")
        self.genomes = {genome.id: genome for genome in next_gen_genomes.values()}

    def get_offspring_count(self, species):
        #print("Getting offspring count...")
        if self.average_fitness is None or self.average_fitness == 0:
            raise ValueError("Average fitness is not calculated or zero.")
        
        return int((species.average_fitness / self.average_fitness) * config.population_size)

    def produce_interspecies_offspring(self):
        print("Producing interspecies offspring...")
        offspring = {}
        for _ in range(config.interspecies_mating_count):
            species_1 = self.random_species()
            species_2 = self.random_species()
            if species_1 != species_2:
                parent_1 = species_1.random_genome()
                parent_2 = species_2.random_genome()
                child = parent_1.crossover(parent_2)
                offspring[child.id] = child
        return offspring

    def random_species(self):
        print("Getting random species...")
        if not self.species:
            raise ValueError("No species available to choose from.")
        species_list = list(self.species.values())  # Convert the values of the dictionary to a list
        return random.choice(species_list)  # Randomly choose from the list


    def save_genomes_to_file(self, file_path):
        print(f"Saving genomes to file: {file_path}")
        with open(file_path, "wb") as file:
            pickle.dump(self.genomes, file)

    def load_genomes_from_file(self, file_path):
        print(f"Loading genomes from file: {file_path}")
        with open(file_path, "rb") as file:
            self.genomes = pickle.load(file)

    def evolve(self):
        self.speciate()
        self.evaluate()
        self.assess()
        self.prune_species()
        self.survive_and_reproduce()

class Species:
    def __init__(self):
        self.id = IdManager.get_new_id()
        self.genomes = {}
        self.elites = {}
        self.representative = None
        self.max_shared_fitness = None
        self.average_fitness = None
        self.age = 0
        self.generations_without_improvement = 0

    def is_same_species(self, genome):
        #print(f"Checking if genome {genome.id} is in species {self.id}...")
        distance = genome.calculate_genetic_distance(self.representative)
        #print(f"Genome {genome.id} distance from species {self.id}: {distance}")
        return distance < config.distance_threshold

    def add_genome(self, genome):
        #print(f"Adding genome {genome.id} to species {self.id}...")
        self.genomes[genome.id] = genome
        if not self.representative:
            self.representative = genome

    def cull(self, keep_best_percentage):
        if not 0 < keep_best_percentage <= 1:
            raise ValueError("keep_best_percentage must be between 0 and 1.")

        # Sort genomes by fitness and update genomes dictionary
        sorted_genomes = sorted(self.genomes.values(), key=lambda genome: genome.fitness, reverse=True)
        
        # Calculate cutoff, ensuring at least one genome is kept
        cutoff = max(1, int(len(sorted_genomes) * keep_best_percentage))
        
        self.genomes = {genome.id: genome for genome in sorted_genomes[:cutoff]}
        print(f"Culled a total of {len(sorted_genomes) - len(self.genomes)} genomes from species {self.id}")


    def produce_offspring(self, offspring_count=1):
        print(f"Producing {offspring_count} offspring(s) for species {self.id}...")
        offspring = {}
        for _ in range(offspring_count):
            if len(self.genomes) > 1:
                # If there are at least two members, randomly select two different parents
                parent1, parent2 = random.sample(list(self.genomes.values()), 2)
                print(f"Species {self.id} has {len(self.genomes)} members, crossing over genomes {parent1.id} and {parent2.id}...")
                new_genome = parent1.crossover(parent2)
            elif self.genomes:
                print(f"Species {self.id} has only one member, copying the genome...")
                # If there is only one member, use it as both parents
                parent = next(iter(self.genomes.values()))
                new_genome = parent.copy()
            else:
                # If there are no members in the species, skip this iteration
                print(f"No members in species {self.id} to produce offspring")
                continue

            new_genome.mutate()
            offspring[new_genome.id] = new_genome

        return offspring

    def random_genome(self):
        print(f"Getting random genome from species {self.id}...")
        if not self.genomes:
            raise ValueError("No genomes in the species to copy from.")

        random_genome = random.choice(list(self.genomes.values()))
        return random_genome.copy()  # This will now use the modified copy method

class Genome:
    def __init__(self):
        self.id = IdManager.get_new_id()
        self.neuron_genes = {}
        self.connection_genes = {}
        self.network = None
        self.network_needs_rebuild = True
        self.fitness = None
        self.shared_fitness = None

    def create(self):
        #print(f"Creating genome {self.id}...")
        self.add_neurons("input", config.input_neurons)
        self.add_neurons("output", config.output_neurons)
        self.add_neurons("hidden", config.hidden_neurons)
        max_possible_conn = config.hidden_neurons * (config.input_neurons + config.hidden_neurons + config.output_neurons)
        attempts = min(config.initial_conn_attempts, max_possible_conn * config.attempts_to_max_factor)
        self.attempt_connections(from_layer=None, to_layer=None, attempts=attempts)
        return self

    def copy(self):
        print(f"Copying genome {self.id}...")
        new_genome = Genome()  # Creates a new genome with a new ID

        # Copying all neuron genes
        for neuron_id, neuron_gene in self.neuron_genes.items():
            new_genome.neuron_genes[neuron_id] = neuron_gene.copy()

        # Copying all connection genes
        for conn_id, conn_gene in self.connection_genes.items():
            new_genome.connection_genes[conn_id] = conn_gene.copy()

        return new_genome

    def add_neurons(self, layer, count):
        #print(f"Adding {count} {layer} neurons to genome {self.id}...")
        for _ in range(count):
            # Create a new neuron and add it to the neuron_genes dictionary
            new_neuron = NeuronGene(layer)
            self.neuron_genes[new_neuron.id] = new_neuron

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
                elif attempting_from_layer == "hidden":
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

    def  crossover(self, other_genome):
        #print(f"Crossing over genome {self.id} with genome {other_genome.id}...")
        offspring = Genome()

        genes1 = {gene.innovation_number: gene for gene in self.connection_genes.values()}
        genes2 = {gene.innovation_number: gene for gene in other_genome.connection_genes.values()}

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

        # Inherit neuron genes
        neurons_to_inherit = set()
        for cg in offspring.connection_genes.values():
            neurons_to_inherit.add(cg.from_neuron)
            neurons_to_inherit.add(cg.to_neuron)

        for neuron_id in neurons_to_inherit:
            neuron1 = self.neuron_genes.get(neuron_id)
            neuron2 = other_genome.neuron_genes.get(neuron_id)

            if neuron1 or neuron2:
                neuron_to_add = neuron1.copy() if neuron1 else neuron2.copy()
                offspring.neuron_genes[neuron_to_add.id] = neuron_to_add

        return offspring

    def mutate(self):
        #print(f"Mutating genome {self.id}...")
        if random.random() < config.gene_add_chance:
            self.attempt_connections()

        if random.random() < config.neuron_add_chance:
            self.add_neurons("hidden", 1)

        if random.random() < config.weight_mutate_chance:
            self.mutate_weight()

        if random.random() < config.bias_mutate_chance:
            self.mutate_bias()

        if random.random() < config.activation_mutate_chance:
            self.mutate_activation_function()

        if random.random() < config.gene_toggle_chance:
            self.mutate_gene_toggle()

        if random.random() < config.neuron_toggle_chance:
            self.mutate_neuron_toggle()

        self.network_needs_rebuild = True

    def mutate_weight(self):
        print(f"Mutating weights for genome {self.id}...")
        for conn_gene in self.connection_genes.values():
            if random.random() < config.weight_mutate_factor:
                perturb = random.uniform(-1, 1) * config.weight_mutate_factor
                conn_gene.weight += perturb
            else:
                conn_gene.weight = random.uniform(*config.weight_init_range)

    def mutate_bias(self):
        print(f"Mutating biases for genome {self.id}...")
        for neuron_gene in self.neuron_genes.values():
            if random.random() < config.bias_mutate_chance:
                perturb = random.uniform(-1, 1) * config.bias_mutate_factor
                neuron_gene.bias += perturb
            else:
                neuron_gene.bias = random.uniform(*config.bias_init_range)

    def mutate_activation_function(self):
        print(f"Mutating activation functions for genome {self.id}...")
        available_functions = ActivationFunctions.get_activation_functions()

        for neuron_gene in self.neuron_genes.values():
            if random.random() < config.activation_mutate_chance:
                neuron_gene.activation = random.choice(available_functions)

    def mutate_gene_toggle(self):
        print(f"Mutating gene toggles for genome {self.id}...")
        for conn_gene in self.connection_genes.values():
            if random.random() < config.gene_toggle_chance:
                conn_gene.enabled = not conn_gene.enabled

    def mutate_neuron_toggle(self):
        print(f"Mutating neuron toggles for genome {self.id}...")
        for neuron_gene in self.neuron_genes.values():
            # Check if the neuron is a hidden neuron
            if neuron_gene.layer == "hidden":
                if random.random() < config.neuron_toggle_chance:
                    neuron_gene.enabled = not neuron_gene.enabled
                    print(f"Toggled neuron {neuron_gene.id}, new state: {'enabled' if neuron_gene.enabled else 'disabled'}")

    def build_network(self):
        print(f"Building network for genome {self.id}...")
        if self.network_needs_rebuild:
            self.network = NeuralNetwork(self)
            self.network_needs_rebuild = False
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
        distance = (config.disjoint_coefficient * disjoint_genes / N) + \
                   (config.excess_coefficient * excess_genes / N) + \
                   (config.weight_diff_coefficient * weight_diff) + \
                   (config.activation_diff_coefficient * activation_diff)

        print(f"Genome {self.id} vs {other_genome.id} - Distance: {distance}")

        return distance

class ConnectionGene:
    def __init__(self, from_neuron, to_neuron):
        self.id = IdManager.get_new_id()
        self.innovation_number = InnovationManager.get_new_innovation_number()
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = random.uniform(*config.weight_init_range)
        self.enabled = True

    def __repr__(self):
        return f"ConnectionGene(innovation_number={self.innovation_number}, from_neuron={self.from_neuron}, to_neuron={self.to_neuron}, weight={self.weight})"

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
        self.activation = config.default_hidden_activation if self.layer == "hidden" else (config.default_output_activation if self.layer == "output" else "identity")
        self.bias = random.uniform(*config.bias_init_range) if self.layer == "output" or self.layer == "hidden" else 0
        self.enabled = True

    def copy(self):
        new_gene = NeuronGene(self.layer)
        new_gene.activation = self.activation
        new_gene.bias = self.bias
        new_gene.enabled = self.enabled
        return new_gene

class NeuralNetwork(nn.Module):
    def __init__(self, genome):
        super(NeuralNetwork, self).__init__()
        self.genome = genome
        self.neuron_states = {gene.id: torch.zeros(1) for gene in genome.neuron_genes.values() if gene.enabled}
        self.weights = None
        self.biases = None
        self.input_neuron_mapping = None

        """# Check if the network needs rebuilding
        #if genome.network_needs_rebuild:
            # Create the network
            # Create a mapping for input neuron IDs
            #self.input_neuron_mapping = {neuron_id: idx for idx, neuron_id in enumerate(
                #sorted(neuron_id for neuron_id, neuron in genome.neuron_genes.items() if neuron.layer == "input" and neuron.enabled))}

            #self._create_network()
            #print(f"Neural network created for genome {genome.id}")
            # Store the newly created network in the genome
            #genome.network = self
            # workaround for the network not being reloadable
            #genome.network_needs_rebuild = True
        #else:
            # Use the existing network from the genome"""

        # Create the network
        # Create a mapping for input neuron IDs
        self.input_neuron_mapping = {
            neuron_id: idx 
            for idx, neuron_id in enumerate(
                sorted(
                    neuron_id 
                    for neuron_id, neuron in genome.neuron_genes.items() 
                    if neuron.layer == "input" and neuron.enabled
                )
            )
        }
        #print("Input Neuron Mapping:", self.input_neuron_mapping)
        input_neuron_ids = [neuron_id for neuron_id, neuron in self.genome.neuron_genes.items() if neuron.layer == "input" and neuron.enabled]
        #print("Input Neurons in Genome:", input_neuron_ids)

        self._create_network()
        #print(f"Neural network created for genome {genome.id}")
        #print("Neuron States:", self.neuron_states)
        #print("Weights:", self.weights)
        #print("Biases:", self.biases)
        #print("Neuron Genes:", self.genome.neuron_genes)
        #print("Connection Genes:", self.genome.connection_genes)
        #print("Input Neuron Mapping:", self.input_neuron_mapping)
        #print("Input Neurons in Genome:", input_neuron_ids)
        #print("Input Neurons in Network:", self.input_neuron_mapping.keys())
        #print("Output Neurons in Network:", [gene.id for gene in self.genome.neuron_genes.values() if gene.layer == "output" and gene.enabled])
        #print("Network:", self)
        #print("Network:", self.genome.network)

        # Store the newly created network in the genome
        genome.network = self
        #self.print_neuron_info()

    def _create_network(self):
        # Create weights for each connection in the genome
        self.weights = nn.ParameterDict({
            f"{gene.from_neuron}_{gene.to_neuron}": nn.Parameter(torch.tensor(gene.weight, dtype=torch.float32))
            for gene in self.genome.connection_genes.values() if gene.enabled
        })

        # Create biases for all neurons in the genome
        self.biases = nn.ParameterDict({
            f"bias_{gene.id}": nn.Parameter(torch.tensor(gene.bias, dtype=torch.float32))
            for gene in self.genome.neuron_genes.values() if gene.enabled
        })

    def print_neuron_info(self):
        print("Neuron Information Snapshot:")
        for neuron_id, neuron_gene in self.genome.neuron_genes.items():
            if neuron_gene.enabled:
                bias_key = f"bias_{neuron_id}"
                bias_value = self.biases[bias_key].item() if bias_key in self.biases else "No bias"
                print(f"Neuron ID: {neuron_id}, Layer: {neuron_gene.layer}, Activation: {neuron_gene.activation}, Bias: {bias_value}")

    def reset_neuron_states(self):
        # Initialize states for all neurons (input, hidden, output) that are part of the network
        self.neuron_states = {neuron_id: torch.zeros(1) for neuron_id, neuron in self.genome.neuron_genes.items() if neuron.enabled}


    def forward(self, input):
        if input.shape[0] != len(self.input_neuron_mapping):
            raise ValueError(f"Input size mismatch. Expected {len(self.input_neuron_mapping)}, got {input.shape[0]}")

        # Reset neuron states
        self.reset_neuron_states()

        # Update input neurons" states
        for neuron_id, idx in self.input_neuron_mapping.items():
            self.neuron_states[neuron_id] = input[idx]

        # Process connections and update neuron states
        for gene in self.genome.connection_genes.values():
            if gene.enabled:
                weight = self.weights[f"{gene.from_neuron}_{gene.to_neuron}"]
                from_neuron_id = gene.from_neuron
                to_neuron_id = gene.to_neuron

                # Update the state of the target neuron
                self.neuron_states[to_neuron_id] += weight * self.neuron_states[from_neuron_id]

        # Apply activation functions and biases
        for neuron_id, neuron_gene in self.genome.neuron_genes.items():
            if neuron_gene.enabled:
                activation_function = getattr(ActivationFunctions, neuron_gene.activation)
                bias_key = f"bias_{neuron_id}"
                if bias_key in self.biases:
                    self.neuron_states[neuron_id] = self.neuron_states[neuron_id] + self.biases[bias_key]
                self.neuron_states[neuron_id] = activation_function(self.neuron_states[neuron_id])

        # Extract output from output neurons
        output_neurons = [neuron_id for neuron_id in self.neuron_states if self.genome.neuron_genes[neuron_id].layer == "output"]
        output = torch.cat([self.neuron_states[neuron_id] for neuron_id in output_neurons])

        return output

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

def dumb_visualize_network(genome):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with different styles for different types of neurons
    for neuron_id, neuron_gene in genome.neuron_genes.items():
        if neuron_gene.enabled:
            if neuron_gene.layer == "input":
                G.add_node(neuron_id, color="skyblue", style="filled", shape="circle")
            elif neuron_gene.layer == "output":
                G.add_node(neuron_id, color="lightgreen", style="filled", shape="circle")
            else:  # hidden
                G.add_node(neuron_id, color="lightgrey", style="filled", shape="circle")

    # Add edges
    for _, conn_gene in genome.connection_genes.items():
        if conn_gene.enabled:
            G.add_edge(conn_gene.from_neuron, conn_gene.to_neuron, weight=conn_gene.weight)

    # Choose a layout for the network
    pos = nx.stochastic_graph(G, weight="weight", scale=1)

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color=[G.nodes[node]["color"] for node in G.nodes], edge_color="black", width=1, linewidths=1, node_size=500, alpha=0.9)
    
    # Show the plot
    plt.show()

def neat():
    population = Population(first=True)
    visualizer = Visualization()
    species_data = []
    fitness_data = []
    first_genome = next(iter(population.genomes.values()))
    dumb_visualize_network(first_genome)

    for generation in range(config.generations):
        print(f"Generation {generation}...")
    
        population.evolve()
    
        species_data.append(len(population.species))
        fitness_data.append(population.max_fitness)
        visualizer.plot_species_count(species_data)
        visualizer.plot_max_fitness(fitness_data)
    
        if generation % config.population_save_interval == 0:
            population.save_genomes_to_file(f"population_gen_{generation}.pkl")
    
    population.save_genomes_to_file("final_population.pkl")
    visualizer.visualize_network(population.best_genome)
    print("Objects created:", InnovationManager.get_new_innovation_number())

config = Config("config.ini", "DEFAULT")

if __name__ == "__main__":
    neat()