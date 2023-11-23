import random

from managers import IdManager

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
        if not self.genomes:
            raise ValueError("No genomes in the species to produce offspring from.")
        
        else:
            genome_list = list(self.genomes.values())  # Convert dictionary values to a list
            total_genomes = len(genome_list)
            weights = [total_genomes - rank for rank in range(total_genomes)]

            offspring = {}

            for _ in range(offspring_count):
                # Select two parent genomes based on rank from the list
                parent1, parent2 = random.choices(genome_list, weights=weights, k=2)
                new_genome = parent1.crossover(parent2)
                new_genome.mutate()
                offspring[new_genome.id] = new_genome

        return offspring

    def random_genome(self):
        if not self.genomes:
            raise ValueError("No genomes in the species to copy from.")
        random_genome = random.choice(list(self.genomes.values()))
        return random_genome.copy()

    def is_same_species(self, genome, distance_threshold):
        # check if representative is set, raise error if not
        if not self.representative:
            raise ValueError("Species has no representative, this should not happen.")

        distance = genome.calculate_genetic_distance(self.representative)
        
        return distance <= distance_threshold

    def add_genome(self, genome):
        self.genomes[genome.id] = genome
        genome.species_id = self.id
        if not self.representative:
            self.representative = genome
