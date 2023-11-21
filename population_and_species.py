import random
import pickle
import gymnasium as gym

from managers import IdManager
from genome import Genome
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
        print("Creating first population...")
        for _ in range(config.population_size):
            genome = Genome().create(self.input_ids, self.output_ids, self.hidden_ids)
            self.genomes[genome.id] = genome
        print(f"Genomes created for first population: {len(self.genomes)}")

    def print_neuron_ids(self):
        print("Input Neuron IDs:", self.input_ids)
        print("Output Neuron IDs:", self.output_ids)
        print("Hidden Neuron IDs:", self.hidden_ids)

    def evolve(self):
        self.evaluate()
        self.stat_and_sort()
        self.form_next_generation()

    def speciate(self):
        print("Speciating population...")

        def find_species_for_genome(genome):
            for species_instance in self.species.values():
                if species_instance.is_same_species(genome):
                    return species_instance
            return None

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

        print(
            f"Species count: {len(self.species)}, Adjusted distance threshold: {config.distance_threshold}"
        )

    def evaluate(self):

        self.environment = gym.make("BipedalWalker-v3", hardcore=True)
        self.environment = gym.wrappers.TimeLimit(
            self.environment, max_episode_steps=100
        )

        self.initial_observation = self.environment.reset()

        print("Evaluating population...")
        if config.parallelize:
            self.evaluate_parallel()
        else:
            self.evaluate_serial()

        self.relu_offset_fitness()

    def evaluate_serial(self):
        print("Serial evaluation")
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

    def stat_and_sort(self):
        print("\nStats and sorting...")

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

        self.max_fitness = None
        self.best_genome = None
        sorted_species = sorted(
            self.species.items(),
            key=lambda item: item[1].average_shared_fitness,
            reverse=True,
        )
        self.species = {species_id: species for species_id, species in sorted_species}

        for _, species in self.species.items():
            print(
                f"Species ID: {species.id}, Average shared fitness: {species.average_shared_fitness}, Members: {len(species.genomes)}, Elites: {len(species.elites)}, Age: {species.age}"
            )

        for _, genome in self.genomes.items():
            if self.max_fitness is None or genome.fitness > self.max_fitness:
                self.max_fitness = genome.fitness
                self.best_genome = genome
        print(
            f"Max fitness: {self.max_fitness}",
            "Best genome:",
            self.best_genome.id if self.best_genome else None,
        )

    def prune(self):
        self.prune_genomes()
        self.stat_and_sort()
        self.prune_stale_species()
        self.stat_and_sort()
        self.prune_weak_species()
        self.stat_and_sort()

    def prune_genomes(self):
        print("\nPruning least fit genomes...")

        for species_instance in self.species.values():

            species_instance.cull(config.keep_best_genomes_in_species)

    def prune_stale_species(self):

        print("\nMass extinction of stale species...")
        max_removal_count = len(self.species) - config.min_species
        removal_count = 0
        for species_instance in self.species.values():
            if (
                species_instance.generations_without_improvement
                >= config.max_stagnation
            ):
                if removal_count < max_removal_count:

                    del self.species[species_instance.id]
                    print(
                        f"Removed stale species ID: {species_instance.id} with generations without improvement: {species_instance.generations_without_improvement}"
                    )
                    removal_count += 1
                else:
                    print(
                        f"Cannot remove stale species ID: {species_instance.id}, max removal count reached"
                    )
                    break
        print(
            f"Removed {removal_count} stale species, {len(self.species)} species surviving"
        )

    def prune_weak_species(self):
        print("\nMass extinction of weak species...")
        max_removal_count = len(self.species) - config.min_species
        print(f"Max removal count: {max_removal_count}")
        removal_count = 0
        if removal_count < max_removal_count:

            if not 0 < config.keep_best_species <= 1:
                raise ValueError("config.keep_best_species must be between 0 and 1.")
            original_count = len(self.species)
            print(f"Original species count: {original_count}")
            cutoff = max(1, int(original_count * config.keep_best_species))
            print(f"Cutoff: {cutoff}")

            keys_to_remove = list(self.species.keys())[cutoff:]

            for key in keys_to_remove:
                del self.species[key]

            print(f"Removed weak species, new species count: {len(self.species)}")

    def form_next_generation(self):
        next_gen = {}
        next_gen_elites = {}
        next_gen_crossovers = {}

        next_gen_elites = self.carry_over_elites(next_gen_elites)

        self.prune()

        next_gen_crossovers = self.reproduce(next_gen_elites, next_gen_crossovers)

        self.genomes = {}
        next_gen.update(next_gen_elites)
        for species_instance in self.species.values():
            species_instance.age += 1
        self.speciate()

        print(
            f"Best genome ID: {self.best_genome.id}, Fitness: {self.best_genome.fitness}, Neurons: {len(self.best_genome.neuron_genes)}, Connections: {len(self.best_genome.connection_genes)}, Disabled connections: {len([gene for gene in self.best_genome.connection_genes.values() if not gene.enabled])}, Disabled neurons: {len([gene for gene in self.best_genome.neuron_genes.values() if not gene.enabled])}"
        )

    def carry_over_elites(self, next_gen_elites):
        for species_instance in self.species.values():

            print(f"members in next generation: {len(next_gen_elites)}")
            print(
                f"Taking species {species_instance.id} elites to the next generation..."
            )
            print(f"Elites: {len(species_instance.elites)}")
            next_gen_elites.update(species_instance.elites)

            print(f"members in next generation after elite add: {len(next_gen_elites)}")

        return next_gen_elites

    def reproduce(self, next_gen_elites, next_gen_crossovers):

        needed_offspring = config.population_size - len(next_gen_elites)
        print(
            f"Total offspring needed: {needed_offspring} - next_gen_elites: {len(next_gen_elites)}"
        )
        for species_instance in self.species.values():
            offspring_count = self.get_offspring_count(
                species_instance, needed_offspring
            )
            print(f"Offspring count: {offspring_count}")
            offspring = species_instance.produce_offspring(offspring_count)
            print(f"Offspring produced: {len(offspring)}")
            next_gen_crossovers.update(offspring)

        while len(next_gen_crossovers) + len(next_gen_elites) < config.population_size:
            print(f"Adding random species offspring to maintain population size...")
            next_gen_crossovers.update(self.random_species().produce_offspring(1))
            print(
                f"Taken random species offspring from species {species_instance.id} to the next generation..."
            )

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
        print(
            f"Culled a total of {original_count - cutoff} genomes from species {self.id}, {len(self.genomes)} genomes remaining"
        )

    def produce_offspring(self, offspring_count=1):

        offspring = {}
        for _ in range(offspring_count):
            if len(self.genomes) > 1:

                parent1, parent2 = random.sample(list(self.genomes.values()), 2)
                new_genome = parent1.crossover(parent2)
            elif self.genomes:

                parent = next(iter(self.genomes.values()))
                new_genome = parent.copy()
            else:

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
        return random_genome.copy()

    def is_same_species(self, genome):
        distance = genome.calculate_genetic_distance(self.representative)

        return distance < config.distance_threshold

    def add_genome(self, genome):
        self.genomes[genome.id] = genome
        genome.species_id = self.id
        if not self.representative:
            self.representative = genome
