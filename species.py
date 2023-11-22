import random

from managers import IdManager
from genome import Genome

from config import Config

config = Config("config.ini", "DEFAULT")

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
                continue

            new_genome.mutate()
            offspring[new_genome.id] = new_genome

        return offspring

    def random_genome(self):
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
