import random
import pickle
import re
import gymnasium as gym
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed

from managers import IdManager
from genome import Genome
from species import Species
from neural_network import NeuralNetwork

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
            self._initial_speciation()

    def _initialize_neuron_ids(self):
        self.input_ids = [IdManager.get_new_id() for _ in range(config.input_neurons)]
        self.output_ids = [IdManager.get_new_id() for _ in range(config.output_neurons)]
        self._print_neuron_ids()
        
    def _first_population(self):
        for _ in range(config.population_size):
            genome = Genome().create(self.input_ids, self.output_ids, hidden_ids=None)
            self.genomes[genome.id] = genome

    def _print_neuron_ids(self):
        print("Input Neuron IDs:", self.input_ids)
        print("Output Neuron IDs:", self.output_ids)

    def _initial_speciation(self):
        # execute speciate until species count stabilizes

        previous_species_count = None
        stabilized = False
        stabilizing = 0
        tries = 0

        
        while not stabilized and len(self.species) != config.target_species and tries < config.init_species_max_tries:
            self.speciate()
            self.remove_empty_species()
            tries += 1

            if len(self.species) == previous_species_count:
                stabilizing += 1
                stabilized = stabilizing >= config.init_species_stabilization
            else:
                stabilizing = 0
                stabilized = False

            previous_species_count = len(self.species)

    def evolve(self):
        print("Evaluation...")
        self.evaluate()
        print("Speciating...")
        self.speciate()
        print("Sort & Stats...")
        self.sort_and_stats()
        self.print_population_info()
        print("Pruning...")
        self.prune()
        print("Forming next generation...")
        self.form_next_generation()

    def speciate(self):
        def find_species_for_genome(genome):
            for species_instance in self.species.values():
                if species_instance.is_same_species(genome, config.distance_threshold):
                    return species_instance
            return None
        
        # Clear all genomes including elites from all species,
        # but keep representatives
        for species_instance in self.species.values():
            species_instance.genomes = {}
            species_instance.elites = {}

        for genome in self.genomes.values():
            species_instance = find_species_for_genome(genome)
            if species_instance:
                species_instance.add_genome(genome)
            else:
                new_species = Species()
                new_species.add_genome(genome)
                self.species[new_species.id] = new_species

        non_empty_species = len([s for s in self.species.values() if s.genomes or s.elites])
        species_ratio = non_empty_species / config.target_species

        adjustment_factor = (
            1.0 - config.distance_adj_factor
            if species_ratio < 1.0
            else 1.0 + config.distance_adj_factor
        )

        config.distance_threshold *= adjustment_factor

        print(f"Number of not empty species: {non_empty_species}, distance threshold: {config.distance_threshold}")

    def remove_empty_species(self):
        species_to_remove = []
        for species_instance in self.species.values():
            if not species_instance.genomes:
                # append species id to list of species to remove
                species_to_remove.append(species_instance.id)
        # remove species from population
        for species_id in species_to_remove:
            del self.species[species_id] 

    def evaluate(self):

        #self.environment = gym.make("BipedalWalker-v3", hardcore=True)
        #self.environment = gym.wrappers.TimeLimit(
            #self.environment, max_episode_steps=config.environment_steps
        #)

        self.environment = gym.make("LunarLander-v2", render=True)
        self.environment = gym.wrappers.TimeLimit(
            self.environment, max_episode_steps=config.environment_steps
        )

        self.initial_observation = self.environment.reset()
        print("Run mode:", config.mode)
        if config.mode == "parallel":
            self.evaluate_parallel()
        elif config.mode == "serial":
            self.evaluate_serial()
        elif config.mode == "dumb":
            self.evaluate_dumb()
        else:
            raise ValueError("No valid evaluation method specified.")

        self.relu_offset_fitness()

    def evaluate_serial(self):

        for genome in self.genomes.values():
            genome.fitness = self.evaluate_single_genome(genome)

    def evaluate_dumb(self):
            
        for genome in self.genomes.values():
            genome.fitness = 1

    def evaluate_parallel(self):
        with ThreadPoolExecutor(max_workers=config.parallelization) as executor:
            futures = {executor.submit(self.evaluate_single_genome, genome): genome for genome in self.genomes.values()}

            for future in as_completed(futures):
                genome = futures[future]
                try:
                    fitness = future.result()
                    genome.fitness = fitness
                except Exception as e:
                    print(f"An error occurred during genome evaluation: {e}")

    def evaluate_single_genome(self, genome):
        observation = self.environment.reset()

        if isinstance(observation, tuple):
            observation = observation[0]

        neural_network = NeuralNetwork(genome)
        neural_network.reset_states()

        done = False
        total_reward = 0

        #while not done:
            # BipedaWalker-v3
            #action = neural_network.forward(observation)
            #observation, reward, terminated, truncated, _ = self.environment.step(action)

            #total_reward += reward

            #if isinstance(observation, tuple):
                #observation = observation[0]

            #done = terminated or truncated

        while not done:
            # Get output logits from the network
            output_logits = neural_network.forward(observation)

            # Apply softmax to convert logits to probabilities (still on GPU)
            action_probabilities = F.softmax(output_logits, dim=0)

            # Choose the action with the highest probability and transfer to CPU
            action = torch.argmax(action_probabilities).cpu().item()

            # Step in the environment using the chosen action
            observation, reward, done, _ = self.environment.step(action)

            total_reward += reward

            if isinstance(observation, tuple):
                observation = observation[0]

        return total_reward

    def relu_offset_fitness(self):

        for _, genome in self.genomes.items():
            genome.fitness = max(0, genome.fitness + config.fitness_offset)

    def sort_and_stats(self):
        for _, species in self.species.items():
            # if species has at least one genome
            if species.genomes:
                sorted_genomes = sorted(
                    species.genomes.values(),
                    key=lambda genome: genome.fitness,
                    reverse=True,
                )
                species.genomes = {genome.id: genome for genome in sorted_genomes}
                species.elites = {
                    genome.id: genome
                    for genome in sorted_genomes[: config.elites_per_species]
                }
                species.total_fitness = sum(
                    genome.fitness for genome in species.genomes.values()
                )
                #print (f"{len(species.genomes)} genomes in species {species.id}") 
                species.average_shared_fitness = species.total_fitness / (
                    len(species.genomes) ** 2
                )
            else:
                species.average_shared_fitness = 0
                species.total_fitness = 0

        sorted_species = sorted(
            self.species.items(),
            key=lambda item: item[1].average_shared_fitness,
            reverse=True,
        )
        self.species = {species_id: species for species_id, species in sorted_species}

        self.max_fitness = 0
        self.best_genome = None

        for _, genome in self.genomes.items():
            if self.max_fitness is None or genome.fitness > self.max_fitness:
                self.max_fitness = genome.fitness
                self.best_genome = genome

        return self.best_genome

    def print_population_info(self):
        print("Population info:")
        for _, species in self.species.items():
            if species.genomes or species.elites:
                print(
                    f"Species ID: {species.id}, Average shared fitness: {species.average_shared_fitness}", \
                    f"Members: {len(species.genomes)}, Elites: {len(species.elites)}, Age: {species.age}"
                )
                non_empty_species = len([s for s in self.species.values() if s.genomes or s.elites])
        # count current number of species
        print(f"\nNumber of species: {len([s for s in self.species.values() if s.genomes or s.elites])},", \
              f"distance threshold: {config.distance_threshold}")
        print(f"BEST GENOME: {self.best_genome.id}, Fitness: {self.max_fitness}, connections: {len(self.best_genome.connection_genes)}, hidden neurons: {len(self.best_genome.neuron_genes) - config.input_neurons - config.output_neurons}")
        print(f"disabled connections: {len([c for c in self.best_genome.connection_genes.values() if not c.enabled])}, disabled neurons: {len([n for n in self.best_genome.neuron_genes.values() if not n.enabled])}\n")

    def prune(self):
        self.prune_genomes_from_species()
        self.sort_and_stats()
        self.prune_stale_species()
        self.sort_and_stats()
        self.prune_weak_species()
        self.sort_and_stats()

    def prune_genomes_from_species(self):

        for species_instance in self.species.values():

            species_instance.cull(config.keep_best_genomes_in_species)

    def prune_stale_species(self):

        max_removal_count = len(self.species) - config.min_species
        removal_count = 0
        for species_instance in self.species.values():
            if (
                species_instance.generations_without_improvement
                >= config.max_stagnation
            ):
                if removal_count < max_removal_count:

                    del self.species[species_instance.id]
                    removal_count += 1
                else:
                    break

    def prune_weak_species(self):
        max_removal_count = len(self.species) - config.min_species
        removal_count = 0
        if removal_count < max_removal_count:

            if not 0 < config.keep_best_species <= 1:
                raise ValueError("config.keep_best_species must be between 0 and 1.")
            original_count = len(self.species)
            cutoff = max(1, int(original_count * config.keep_best_species))
            self.species = dict(list(self.species.items())[:cutoff])

    def form_next_generation(self):
        next_gen = {}
        next_gen_elites = {}
        next_gen_crossovers = {}

        next_gen_elites = self.carry_over_elites(next_gen_elites)
        next_gen_crossovers = self.reproduce(next_gen_elites, next_gen_crossovers)

        # print the number of elites and crossovers
        next_gen.update(next_gen_elites)
        next_gen.update(next_gen_crossovers)

        self.genomes = next_gen

        for species_instance in self.species.values():
            species_instance.age += 1
        
        self.purge_species()
    
    def purge_species(self):
        # removes from all species all genomes that are not representative nor in population
        for species_instance in self.species.values():
            for genome_id in list(species_instance.genomes.keys()):
                if genome_id not in self.genomes and genome_id != species_instance.representative.id:
                    del species_instance.genomes[genome_id]
        
    def carry_over_elites(self, next_gen_elites):
        for species_instance in self.species.values():

            next_gen_elites.update(species_instance.elites)

        return next_gen_elites

    def reproduce(self, next_gen_elites, next_gen_crossovers):

        needed_offspring = config.population_size - len(next_gen_elites)
        for species_instance in self.species.values():
            # ignore if species has no genomes
            if not species_instance.genomes:
                continue
            offspring_count = self.get_offspring_count(
                species_instance, needed_offspring
            )
            offspring = species_instance.produce_offspring(offspring_count)
            next_gen_crossovers.update(offspring)

        while len(next_gen_crossovers) + len(next_gen_elites) < config.population_size:
            next_gen_crossovers.update(self.random_species().produce_offspring(1))

        return next_gen_crossovers

    def get_offspring_count(self, species_instance, needed_offspring):

        rank = list(self.species.keys()).index(species_instance.id) + 1
        total_rank_sum = sum(1 / i for i in range(1, len(self.species) + 1))

        offspring_count = int((needed_offspring * (1 / rank)) / total_rank_sum)

        return offspring_count

    def random_species(self):
        species_with_genomes = [s for s in self.species.values() if s.genomes]
        if not species_with_genomes:
            raise ValueError("No species available to choose from or no species with genomes.")
        return random.choice(species_with_genomes)

    def save_genomes_to_file(self, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(self.genomes, file)

    def load_genomes_from_file(self, file_path):
        with open(file_path, "rb") as file:
            self.genomes = pickle.load(file)
