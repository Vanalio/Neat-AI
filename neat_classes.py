# FIXME: cull and stale species seem not to work properly
# FIXME: crossover count is not correct
# FIXME: implement parallel evaluation
# FIXME: parent selection must be based on chance proportional to rank of genome within the species, not chance proportional to its fitness
# FIXME: implement update of age and generations without improvement
# FIXME: implement interspecies mating

# ADD: randomize activation function at creation or mutate_add_neuron with random.choice(ActivationFunctions.get_activation_functions())

# CHECK: check if the network is built correctly and the computation is correct
# CHECK: disabled genes are correctly inherited 
# CHECK: removals should remove something more than just what they remove (dependencies?)
# CHECK: purge redundant methods

import random
import gymnasium as gym
import pickle

#from memory_profiler import profile

# custom imports
from torch_activation_functions import ActivationFunctions as activation_functions
from managers import IdManager
from genome import Genome
from config import Config

config = Config("config.ini", "DEFAULT")

class Population:
    def __init__(self, first=False):
        self.genomes = {}
        self.species = {}
        self.max_fitness = None
        self.best_genome = None
        self.environment = None
        if first:
            self._initialize_neuron_ids()
            self._first_population()
            self.print_neuron_ids()

    def _initialize_neuron_ids(self):
        self.input_ids = [IdManager.get_new_id() for _ in range(config.input_neurons)]
        self.output_ids = [IdManager.get_new_id() for _ in range(config.output_neurons)]
        self.hidden_ids = [IdManager.get_new_id() for _ in range(config.hidden_neurons)]

    def _first_population(self):
        print("Creating first population...")
        for _ in range(config.population_size):
            genome = Genome().create(self.input_ids, self.output_ids, self.hidden_ids)
            self.genomes[genome.id] = genome
        print(f"Genomes created for first population: {len(self.genomes)}")
        #for genome in self.genomes.values():
            #print(f"Genome ID: {genome.id}, Neurons: {len(genome.neuron_genes)}, Connections: {len(genome.connection_genes)}")

    def print_neuron_ids(self):
        print("Input Neuron IDs:", self.input_ids)
        print("Output Neuron IDs:", self.output_ids)
        print("Hidden Neuron IDs:", self.hidden_ids)

    def evolve(self):
        self.speciate()
        self.evaluate()
        self.relu_offset_fitness()
        self.stat_and_sort()
        self.prune()
        self.form_next_generation()

    def speciate(self):
        print("Speciating population...")
        self.species = {}
        for _, genome in self.genomes.items():
            placed_in_species = False
            for _, species_instance in self.species.items():
                if species_instance.is_same_species(genome):
                    species_instance.add_genome(genome)
                    placed_in_species = True
                    break
            if not placed_in_species:
                new_species = Species()
                new_species.add_genome(genome)
                self.species[new_species.id] = new_species
        species_ratio = len(self.species) / config.target_species
        if species_ratio < 1.0:
            config.distance_threshold *= (1.0 - config.distance_adj_factor)
        elif species_ratio > 1.0:
            config.distance_threshold *= (1.0 + config.distance_adj_factor)
        print(f"Species count: {len(self.species)}, Adjusted distance threshold: {config.distance_threshold}")

    def evaluate(self):
        # Initialize the environment
        self.environment = gym.make("BipedalWalker-v3", hardcore=True)
        self.environment = gym.wrappers.TimeLimit(self.environment, max_episode_steps=100)

        # Reset the environment and store the initial observation
        self.initial_observation = self.environment.reset()

        print("Evaluating population...")
        if config.parallelize:
            self.evaluate_parallel()
        else:
            self.evaluate_serial()
    
    def evaluate_serial(self):
        print("Serial evaluation")
        for genome in self.genomes.values():
            genome.fitness = self.evaluate_single_genome(genome)

    def evaluate_parallel(self):
        pass

    #def evaluate_single_genome(self, genome):
    #    # Reset environment to initial state
    #    observation = self.environment.reset()
    #    
    #    # Extract observation array from tuple if necessary
    #    if isinstance(observation, tuple):
    #        observation = observation[0]
#
    #    observation = torch.from_numpy(observation).float()
#
    #    neural_network = NeuralNetwork(genome)
    #    done = False
    #    total_reward = 0
#
    #    while not done:
    #        action = neural_network(observation)
    #        action = action.detach().cpu().numpy().ravel()
    #        observation, reward, terminated, truncated, _ = self.environment.step(action)
    #        total_reward += reward
#
    #        # Extract observation array from tuple if necessary
    #        if isinstance(observation, tuple):
    #            observation = observation[0]
#
    #        observation = torch.from_numpy(observation).float()
    #        done = terminated or truncated
#
    #    return total_reward

    def evaluate_single_genome(self, genome):
        return 1

    def relu_offset_fitness(self):

        for _, genome in self.genomes.items():
            genome.fitness = max(0, genome.fitness + config.fitness_offset)
            #print(f"Genome ID: {genome.id}, Fitness: {genome.fitness}")

    def stat_and_sort(self):
        print("\nStats and sorting...")
        # Calculate species fitness stats, find elites and sort genomes
        for _, species in self.species.items():
            sorted_genomes = sorted(species.genomes.values(), key=lambda genome: genome.fitness, reverse=True)
            species.genomes = {genome.id: genome for genome in sorted_genomes}
            species.elites = {genome.id: genome for genome in sorted_genomes[:config.elites_per_species]}
            species.total_fitness = sum(genome.fitness for genome in species.genomes.values())
            species.average_shared_fitness = species.total_fitness / (len(species.genomes) ** 2)

        self.max_fitness = None
        self.best_genome = None
        sorted_species = sorted(self.species.items(), key=lambda item: item[1].average_shared_fitness, reverse=True)
        self.species = {species_id: species for species_id, species in sorted_species}

        for _, species in self.species.items():
            print(f"Species ID: {species.id}, Average shared fitness: {species.average_shared_fitness}, Members: {len(species.genomes)}, Elites: {len(species.elites)}")

        # Calculate population fitness stats
        for _, genome in self.genomes.items():
            if self.max_fitness is None or genome.fitness > self.max_fitness:
                self.max_fitness = genome.fitness
                self.best_genome = genome
        print(f"Max fitness: {self.max_fitness}", "Best genome:", self.best_genome.id if self.best_genome else None)

    def prune(self):
        self.prune_genomes()
        self.stat_and_sort()
        self.prune_stale_species()
        self.stat_and_sort()
        self.prune_weak_species()
        self.stat_and_sort()

    def prune_genomes(self):
        print("\nPruning least fit genomes...")
        # Prune genomes from species
        for species_instance in self.species.values():
            #print(f"Discarding the least fit genomes from species {species_instance.id}...")
            species_instance.cull(config.keep_best_genomes_in_species)  # Keep only a portion of the genomes


    def prune_stale_species(self):
        # Prune stale species
        print("\nMass extinction of stale species...")
        max_removal_count = len(self.species) - config.min_species
        removal_count = 0
        for species_instance in self.species.values():
            if species_instance.generations_without_improvement >= config.max_stagnation:
                if removal_count < max_removal_count:
                # Remove stale species
                    del self.species[species_instance.id] # FIXME: remove also dependant objects
                    print(f"Removed stale species ID: {species_instance.id} with generations without improvement: {species_instance.generations_without_improvement}")
                    removal_count += 1
                else:
                    print(f"Cannot remove stale species ID: {species_instance.id}, max removal count reached")
                    break
        print(f"Removed {removal_count} stale species, {len(self.species)} species surviving")

    def prune_weak_species(self):
        print("\nMass extinction of weak species...")
        max_removal_count = len(self.species) - config.min_species
        print(f"Max removal count: {max_removal_count}")
        removal_count = 0
        if removal_count < max_removal_count:
            # Remove weak species
            if not 0 < config.keep_best_species <= 1:
                raise ValueError("config.keep_best_species must be between 0 and 1.")
            original_count = len(self.species)
            print(f"Original species count: {original_count}")
            cutoff = max(1, int(original_count * config.keep_best_species))
            print(f"Cutoff: {cutoff}")

            # Gather keys of species to be removed
            keys_to_remove = list(self.species.keys())[cutoff:]
            
            # Delete each species
            for key in keys_to_remove:
                del self.species[key]

            print(f"Removed weak species, new species count: {len(self.species)}")

    def form_next_generation(self):   
        print("\nForming next generation...")
        next_gen_genomes = {}

        # Carry over elites and reproduce
        next_gen_genomes = self.carry_over_elites(next_gen_genomes)
        next_gen_genomes = self.reproduce(next_gen_genomes)
        
        # Initialize a new population instance with the next generation genomes
        self.__init__()
        self.genomes = next_gen_genomes

        for genome in self.genomes.values():
            print(f"Genome ID: {genome.id}, Neurons: {len(genome.neuron_genes)}, Connections: {len(genome.connection_genes)}, Fitness: {genome.fitness}, Disabled connections: {len([gene for gene in genome.connection_genes.values() if not gene.enabled])}, Disabled neurons: {len([gene for gene in genome.neuron_genes.values() if not gene.enabled])}")
    
    def carry_over_elites(self, next_gen_genomes):
        # Carry over the elites
        for species_instance in self.species.values():
            print(f"Taking species {species_instance.id} elites to the next generation...")
            next_gen_genomes.update(species_instance.elites)
        return next_gen_genomes
    
    def reproduce(self, next_gen_genomes):
        # Calculate total offspring needed
        needed_offspring = config.population_size - len(next_gen_genomes)
        print(f"Total offspring needed: {needed_offspring} - next_gen_genomes: {len(next_gen_genomes)}")
        for species_instance in self.species.values():
            offspring_count = self.get_offspring_count(species_instance, needed_offspring, next_gen_genomes)
            offspring = species_instance.produce_offspring(offspring_count)
            next_gen_genomes.update(offspring)

        # Ensure population size is maintained adding random species offspring
        while len(next_gen_genomes) < config.population_size:
            #print(f"Adding random species offspring to maintain population size...")
            print(f"next_gen_genomes: {len(next_gen_genomes)}")
            next_gen_genomes.update(self.random_species().produce_offspring(1))
            #print(f"Taken random species offspring from species {species_instance.id} to the next generation...")
        return next_gen_genomes
        
    def get_offspring_count(self, species_instance, needed_offspring, next_gen_genomes):
        # Calculate the rank of the given species
        rank = list(self.species.keys()).index(species_instance.id) + 1
        total_rank_sum = sum(1 / i for i in range(1, len(self.species) + 1))
        # Calculate the offspring count for the given species
        offspring_count = int((needed_offspring * (1 / rank)) / total_rank_sum)

        return offspring_count

    def random_species(self):
        #print("Getting random species...")
        if not self.species:
            raise ValueError("No species available to choose from.")
        return random.choice(list(self.species.values()))

    def remove_species():
        pass

    def save_genomes_to_file(self, file_path):
        print(f"Saving genomes to file: {file_path}")
        with open(file_path, "wb") as file:
            pickle.dump(self.genomes, file)

    def load_genomes_from_file(self, file_path):
        print(f"Loading genomes from file: {file_path}")
        with open(file_path, "rb") as file:
            self.genomes = pickle.load(file)

class Species:
    def __init__(self):
        self.id = IdManager.get_new_id()
        self.genomes = {}
        self.elites = {}
        self.representative = None
        self.average_shared_fitness = None
        self.age = 0
        self.generations_without_improvement = 0

    def cull(self, keep_best_genomes_in_species):
        if not 0 < keep_best_genomes_in_species <= 1:
            raise ValueError("keep_best_genomes_in_species must be between 0 and 1.")
        original_count = len(self.genomes)
        cutoff = max(1, int(original_count * keep_best_genomes_in_species))
        self.genomes = dict(list(self.genomes.items())[:cutoff])
        print(f"Culled a total of {original_count - cutoff} genomes from species {self.id}, {len(self.genomes)} genomes remaining")
        # FIXME: order genomes, remove also dependant objects

    def produce_offspring(self, offspring_count=1):
        #print(f"\nProducing {offspring_count} offspring(s) for species {self.id}...")
        offspring = {}
        for _ in range(offspring_count):
            if len(self.genomes) > 1:
                # If there are at least two members, randomly select two different parents
                parent1, parent2 = random.sample(list(self.genomes.values()), 2)
                #print(f"Species {self.id} has {len(self.genomes)} members, crossing over genomes {parent1.id} and {parent2.id}...")
                new_genome = parent1.crossover(parent2)
            elif self.genomes:
                #print(f"Species {self.id} has only one member, copying the genome...")
                # If there is only one member, use it as both parents
                parent = next(iter(self.genomes.values()))
                new_genome = parent.copy()
            else:
                # If there are no members in the species, skip this iteration
                print(f"No members in species {self.id} to produce offspring")
                continue

            new_genome.mutate()
            offspring[new_genome.id] = new_genome
        #print(f"Produced {len(offspring)} offspring(s) for species {self.id}")
        return offspring

    def random_genome(self):
        print(f"Getting random genome from species {self.id}...")
        if not self.genomes:
            raise ValueError("No genomes in the species to copy from.")
        random_genome = random.choice(list(self.genomes.values()))
        return random_genome.copy()

    def is_same_species(self, genome):
        distance = genome.calculate_genetic_distance(self.representative)
        return distance < config.distance_threshold

    def add_genome(self, genome):
        self.genomes[genome.id] = genome
        if not self.representative:
            self.representative = genome