import random
import pickle
import gymnasium as gym

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
            self.print_neuron_ids()
            self.speciate()

    def _initialize_neuron_ids(self):
        self.input_ids = [IdManager.get_new_id() for _ in range(config.input_neurons)]
        self.output_ids = [IdManager.get_new_id() for _ in range(config.output_neurons)]
        self.hidden_ids = [IdManager.get_new_id() for _ in range(config.hidden_neurons)]

    def _first_population(self):
        for _ in range(config.population_size):
            genome = Genome().create(self.input_ids, self.output_ids, self.hidden_ids)
            self.genomes[genome.id] = genome

    def print_neuron_ids(self):
        print("Input Neuron IDs:", self.input_ids)
        print("Hidden Neuron IDs:", self.hidden_ids)
        print("Output Neuron IDs:", self.output_ids)

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
                if species_instance.is_same_species(genome):
                    return species_instance
            return None

        # Print the genomes count of current population

        for genome in self.genomes.values():
            species_instance = find_species_for_genome(genome)
            if species_instance:
                species_instance.add_genome(genome)
            else:
                new_species = Species()
                new_species.add_genome(genome)
                self.species[new_species.id] = new_species

        species_ratio = len(self.species) / config.target_species
        adjustment_factor = (
            1.0 - config.distance_adj_factor
            if species_ratio < 1.0
            else 1.0 + config.distance_adj_factor
        )
        config.distance_threshold *= adjustment_factor

    def evaluate(self):

        self.environment = gym.make("BipedalWalker-v3", hardcore=True)
        self.environment = gym.wrappers.TimeLimit(
            self.environment, max_episode_steps=100
        )

        self.initial_observation = self.environment.reset()

        if config.parallelize:
            self.evaluate_parallel()
        else:
            self.evaluate_serial()

        self.relu_offset_fitness()

    def evaluate_serial(self):

        for genome in self.genomes.values():
            genome.fitness = self.evaluate_single_genome(genome)

    def evaluate_parallel(self):
        pass

    def evaluate_single_genome(self, genome):

        observation = self.environment.reset()

        if isinstance(observation, tuple):
            observation = observation[0]

        neural_network = NeuralNetwork(genome, self.input_ids, self.output_ids)
        neural_network.reset_hidden_states()

        done = False
        total_reward = 0

        while not done:
            action = neural_network.propagate(observation)

            observation, reward, terminated, truncated, _ = self.environment.step(
                action
            )
            total_reward += reward

            if isinstance(observation, tuple):
                observation = observation[0]

            done = terminated or truncated

        return total_reward

    def relu_offset_fitness(self):

        for _, genome in self.genomes.items():
            genome.fitness = max(0, genome.fitness + config.fitness_offset)

    def sort_and_stats(self):
        for _, species in self.species.items():
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
            species.average_shared_fitness = species.total_fitness / (
                len(species.genomes) ** 2
            )

        sorted_species = sorted(
            self.species.items(),
            key=lambda item: item[1].average_shared_fitness,
            reverse=True,
        )
        self.species = {species_id: species for species_id, species in sorted_species}

        self.max_fitness = 0
        self.best_genome = None

        for genome_id, genome in self.genomes.items():
            if self.max_fitness is None or genome.fitness > self.max_fitness:
                self.max_fitness = genome.fitness
                self.best_genome = genome

        return self.best_genome

    def print_population_info(self):
        print("Population info:")
        for _, species in self.species.items():
            print(
                f"Species ID: {species.id}, Average shared fitness: {species.average_shared_fitness}", \
                f"Members: {len(species.genomes)}, Elites: {len(species.elites)}, Age: {species.age}"
            )
        print(f"BEST GENOME: {self.best_genome.id}, Fitness: {self.max_fitness}")

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
        
        # removes all genomes that are not in self.genomes from all species
        self.remove_genomes_from_species()
    
    def remove_genomes_from_species(self):
        for species_instance in self.species.values():
            species_instance.genomes = {
                genome_id: genome
                for genome_id, genome in species_instance.genomes.items()
                if genome_id in self.genomes
            }

    def carry_over_elites(self, next_gen_elites):
        for species_instance in self.species.values():

            next_gen_elites.update(species_instance.elites)

        return next_gen_elites

    def reproduce(self, next_gen_elites, next_gen_crossovers):

        needed_offspring = config.population_size - len(next_gen_elites)
        for species_instance in self.species.values():
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

        if not self.species:
            raise ValueError("No species available to choose from.")
        return random.choice(list(self.species.values()))

    def save_genomes_to_file(self, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(self.genomes, file)

    def load_genomes_from_file(self, file_path):
        with open(file_path, "rb") as file:
            self.genomes = pickle.load(file)
