import random
import numpy as np

class Data:
    EVOLUTION_INPUT = ((1.0, 2.0, 3.0, 4.0), (-5.2, -4.2, -3.2, -2.2), (50000.0, 50001.0, 50002, 50003.0))
    EVOLUTION_OUTPUT = ((5.0, 6.0), (-1.2, -0.2), (50004.0, 50005.0))
    TEST_INPUT = ((-1724.0, -1723.0, -1722.0, -1721.0), (101.0, 102.0, 103.0, 104.0))
    TEST_OUTPUT = ((-1720.0, -1719.0), (105.0, 106.0))
class Evolution:
    POPULATION_SIZE = 5000
    GENERATIONS = 60
    EVOLUTION_BATCH_SIZE = 3
    ELITISM = 2
    MAX_STAGNATION = 20
    TARGET_SPECIES = 50
    INITIAL_DISTANCE_DELTA = 1
    DISTANCE_ADJ_FACTOR = 0.25
    DISJOINT_COEFFICIENT = 1
    EXCESS_COEFFICIENT = 1
    ACTIVATION_COEFFICIENT = 0.5
class Neural:    
    INPUT_NEURONS = 4
    OUTPUT_NEURONS = 2
    INITIAL_GENES_QUOTA = 0.3
    NEURON_ADD_CHANCE = 0.1
    NEURON_TOGGLE_CHANCE = 0.001
    BIAS_INIT_RANGE = (-2, 2)
    BIAS_MUTATE_CHANCE = 0.1
    BIAS_MUTATE_FACTOR = 0.5
    ACTIVATION_MUTATE_CHANCE = 0.1
    GENE_ADD_CHANCE = 0.2
    GENE_TOGGLE_CHANCE = 0.002
    WEIGHT_INIT_RANGE = (-2, 2)
    WEIGHT_MUTATE_CHANCE = 0.1
    WEIGHT_MUTATE_FACTOR = 0.5
class Layers:
    INPUT = "input_layer"
    HIDDEN = "hidden_layer"
    OUTPUT = "output_layer"
class ActivationFunctions:
    @staticmethod
    def linear(x):
        return x
    @staticmethod
    def relu(x):
        return max(0, x)
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    functions = {
        'linear': linear,
        'relu': relu,
        'tanh': tanh
    }
class Neuron:
    def __init__(self, layer_type):
        self.id = id_manager.get_id()
        self.layer = layer_type
        self.is_enabled = True
        self.value = 0
        self.bias = None
        self.always_enabled = False
        self.activation_function = None
        if layer_type == Layers.INPUT:
            self.always_enabled = True
        elif layer_type in [Layers.HIDDEN, Layers.OUTPUT]:
            self.activation_function = random.choice(list(ActivationFunctions.functions.keys()))
            self.bias = random.uniform(*Neural.BIAS_INIT_RANGE)
            if layer_type == Layers.OUTPUT:
                self.always_enabled = True
        else:
            raise ValueError("Invalid layer type")
    def mutate(self):
        if random.random() < Neural.BIAS_MUTATE_CHANCE:
            if random.random() < Neural.BIAS_MUTATE_FACTOR:
                self.bias += random.uniform(-0.5, 0.5)
            else:
                self.bias = random.uniform(*Neural.BIAS_INIT_RANGE)
        if random.random() < Neural.ACTIVATION_MUTATE_CHANCE:
            self.activation_function = random.choice(list(ActivationFunctions.functions.keys()))
class Gene:
    def __init__(self, genome, from_neuron, to_neuron):
        self.id = id_manager.get_id()
        self.innovation_number = id_manager.get_innovation_number()
        self.genome = genome
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = random.uniform(*Neural.WEIGHT_INIT_RANGE)
        self.is_enabled = True
    def mutate_weight(self):
        if random.random() < Neural.WEIGHT_MUTATE_FACTOR:
            self.weight += random.uniform(-0.5, 0.5)
        else:
            self.weight = random.uniform(*Neural.WEIGHT_INIT_RANGE)
class Genome:
    def __init__(self):
        self.id = id_manager.get_id()
        self.neurons = []
        self.genes = []
        self.result = None
        self.loss = None
    @classmethod
    def create_genome(cls):
        genome = cls()
        for _ in range(Neural.INPUT_NEURONS):
            neuron = Neuron(layer=Layers.INPUT)
            genome.neurons.append(neuron)
        for _ in range(Neural.OUTPUT_NEURONS):
            neuron = Neuron(layer=Layers.OUTPUT)
            genome.neurons.append(neuron)
        for i in range(Neural.INPUT_NEURONS):
            for j in range(Neural.INPUT_NEURONS, Neural.INPUT_NEURONS + Neural.OUTPUT_NEURONS):
                gene = Gene(genome=genome, from_neuron=genome.neurons[i], to_neuron=genome.neurons[j])
                genome.genes.append(gene)
        return genome
    def clone_genome(self):
        pass
    def compute(self, input, evolution_output=None, loss_function=None):
        pass
    def feedforward(self, inputs):
        if len(inputs) != Neural.INPUT_NEURONS:
            raise ValueError("Number of inputs provided doesn't match the number of input neurons.")
        for neuron in self.neurons:
            neuron.value = 0
        for i, input_value in enumerate(inputs):
            self.neurons[i].value = input_value
        for _ in range(len(self.neurons)):
            for gene in self.genes:
                if gene.is_enabled:
                    from_neuron = gene.from_neuron
                    to_neuron = gene.to_neuron
                    if from_neuron.layer != Layers.INPUT and to_neuron.layer != Layers.INPUT and from_neuron.id != to_neuron.id:
                        to_neuron.value += from_neuron.value * gene.weight
        for neuron in self.neurons:
            if neuron.layer != Layers.INPUT:
                func = ActivationFunctions.functions[neuron.activation_function]
                neuron.value = func(neuron.value + neuron.bias)
        outputs = [neuron.value for neuron in self.neurons if neuron.layer == Layers.OUTPUT]
        return outputs
    def mutate(self):
        self.change_topology()
        self.adjust_weight()
        self.set_weight()
        self.adjust_bias()
        self.set_bias()
        self.set_activation_function()
        return self
    def change_topology(self):
        self.add_neuron()
        self.toggle_neuron()
        self.add_gene()
        self.toggle_gene()
        self.check_recurrency()
    def add_neuron(self):
        if not self.genes:
            return
        gene_to_split = random.choice(self.genes)
        if not gene_to_split.is_enabled:
            return
        gene_to_split.is_enabled = False
        new_neuron = Neuron(Layers.HIDDEN)
        self.neurons.append(new_neuron)
        new_gene1 = Gene(genome=self, from_neuron=gene_to_split.from_neuron, to_neuron=new_neuron)
        new_gene1.weight = 1.0
        self.genes.append(new_gene1)
        new_gene2 = Gene(genome=self, from_neuron=new_neuron, to_neuron=gene_to_split.to_neuron)
        new_gene2.weight = gene_to_split.weight
        self.genes.append(new_gene2)
    def add_gene(self):
        potential_connections = [(from_neuron, to_neuron) for from_neuron in self.neurons for to_neuron in self.neurons if from_neuron != to_neuron]
        existing_connections = {(gene.from_neuron, gene.to_neuron) for gene in self.genes}
        potential_connections = [conn for conn in potential_connections if conn not in existing_connections]
        if not potential_connections:
            return
        from_neuron, to_neuron = random.choice(potential_connections)
        new_gene = Gene(self, from_neuron, to_neuron)
        self.genes.append(new_gene)
        gene_to_toggle = random.choice(self.genes)
        gene_to_toggle.toggle()
    def check_recurrency(self):
        pass
class Species:
    def __init__(self):
        self.id = id_manager.get_id()
        self.representative = None
        self.representative_raw_fitness = 0
        self.genomes = []
        self.adjusted_fitness = 0
        self.current_distance_delta = Evolution.INITIAL_DISTANCE_DELTA
    def add_genome(self, genome):
        pass
    def remove_genome(self, genome):
        pass
    def update_fitness(self):
        total_fitness = 0
        self.adjusted_fitness = 0
        for genome in self.genomes:
            raw_fitness = genome.raw_fitness
            total_fitness += raw_fitness
            if self.representative_raw_fitness < raw_fitness:
                self.representative = genome
                self.representative_raw_fitness = raw_fitness
        self.adjusted_fitness = total_fitness / len(self.genomes)
    def calculate_distance(self, from_genome, to_genome):
        matching_genes = 0
        activation_diff = 0
        excess_genes = 0
        disjoint_genes = 0
        from_innovation_numbers = {gene.innovation_number: gene for gene in from_genome.genes}
        to_innovation_numbers = {gene.innovation_number: gene for gene in to_genome.genes}
        max_innovation_from = max(from_innovation_numbers.keys()) if from_innovation_numbers else 0
        max_innovation_to = max(to_innovation_numbers.keys()) if to_innovation_numbers else 0
        all_innovations = set(from_innovation_numbers.keys()).union(to_innovation_numbers.keys())
        for innovation in all_innovations:
            from_gene = from_innovation_numbers.get(innovation)
            to_gene = to_innovation_numbers.get(innovation)
            if from_gene and to_gene:
                matching_genes += 1
                from_neuron_activation = from_genome.neurons[from_gene.to_neuron.id].activation_function
                to_neuron_activation = to_genome.neurons[to_gene.to_neuron.id].activation_function
                if from_neuron_activation != to_neuron_activation:
                    activation_diff += 1
            elif from_gene and innovation > max_innovation_to:
                excess_genes += 1
            elif to_gene and innovation > max_innovation_from:
                excess_genes += 1
            else: 
                disjoint_genes += 1
        N = max(len(from_genome.genes), len(to_genome.genes))
        N = max(N, 1)
        distance = (Evolution.EXCESS_COEFFICIENT * excess_genes + Evolution.DISJOINT_COEFFICIENT * disjoint_genes) / N
        distance += Evolution.ACTIVATION_COEFFICIENT * activation_diff / N
        return distance
    def adjust_distance_delta(self, number_of_species):
        if number_of_species < Evolution.TARGET_SPECIES:
            self.current_distance_delta -= Evolution.DISTANCE_ADJ_FACTOR
        elif number_of_species > Evolution.TARGET_SPECIES:
            self.current_distance_delta += Evolution.DISTANCE_ADJ_FACTOR
        self.current_distance_delta = max(self.current_distance_delta, Evolution.INITIAL_DISTANCE_DELTA)
class Population:
    def __init__(self):
        self.current_innovation_id = 0
        self.current_generation = 0
        self.best_fitness = None
        self.best_genome = None
        self.genomes = []
        self.species = []
    @classmethod
    def first_gen(cls):
        if not (0 <= Neural.INITIAL_GENES_QUOTA <= 1):
            raise ValueError(f"{Neural.INITIAL_GENES_QUOTA} should be between 0 and 1.")
        inital_max_genes=Neural.INPUT_NEURONS * Neural.OUTPUT_NEURONS
        for _ in range(Evolution.POPULATION_SIZE):
            upper_bound = round(inital_max_genes * Neural.INITIAL_GENES_QUOTA)
            genes_number = random.randint(0, upper_bound)            
            genome = Genome.create_genome(genes_number=genes_number)
            self.genomes.append(genome)
        return cls()
    def get_innovation_number(self):
        self.innovation_id += 1
        return self.innovation_id
    def speciate(self):
        pass
    def next_gen(self):
        new_genomes = []
        elite_count = int(Evolution.ELITISM * Evolution.POPULATION_SIZE)
        sorted_genomes = sorted(self.genomes, key=lambda genome: genome.result, reverse=True)
        new_genomes.extend(sorted_genomes[:elite_count])
        while len(new_genomes) < Evolution.POPULATION_SIZE:
            parent1 = random.choice(self.genomes)
            parent2 = random.choice(self.genomes)
            if parent1 == parent2:
                child = parent1.clone_genome()
            else:
                child = EvolutionManager().crossover(parent1, parent2)
            child.mutate()
            new_genomes.append(child)
        self.genomes = new_genomes
        self.current_generation += 1
#def save_progress():
#def load_progress():
#population = Population.first_gen(Evolution.POPULATION_SIZE)

"""
Optimizing the `speciate` method for performance could involve several strategies, especially considering that this method is likely to be called many times during the evolution process. Here are some suggestions:

1. **Efficient Data Structures**: Instead of using a list to store species, a more efficient data structure like a dictionary or a set could be used. This would allow for faster lookups when trying to find if a genome belongs to an existing species.

2. **Hashing for Fast Comparison**: Implement a hashing mechanism for genomes that allows you to quickly determine if two genomes are in the same species without having to compute the genetic distance every time.

3. **Caching**: Cache the results of genetic distance calculations since many comparisons are redundant across generations. This would prevent recalculating the distance between the same pairs of genomes.

4. **Bulk Operations**: When adding genomes to species, instead of adding them one by one, group genomes by their species and add them in bulk. This can reduce the overhead of multiple method calls.

5. **Parallel Processing**: If the genetic distance calculations are independent of each other, you could use parallel processing to compute distances in parallel. This could be done using Python's `multiprocessing` module or third-party libraries like `joblib`.

6. **Optimized Distance Calculation**: Review the `calculate_genetic_distance` function to ensure it is as efficient as possible. For example, you could avoid dividing by `N` if `N` equals 1 or use more efficient looping constructs.

7. **Early Stopping**: In the `is_same_species` method, if you determine early that the distance is greater than the compatibility threshold, you can break out of the loop early and avoid unnecessary calculations.

8. **Species Representative Sampling**: Instead of comparing a genome to all members of a species, you could compare it only to a representative sample. This could reduce the number of distance calculations significantly.

9. **Incremental Speciation**: Instead of re-speciating all genomes every generation, only re-speciate genomes that have mutated significantly since the last generation.

10. **Vectorization**: If you're using NumPy for distance calculations, ensure that you're taking advantage of vectorized operations instead of Python loops where possible.

11. **Profiling**: Use profiling tools like `cProfile` to identify the bottlenecks in the `speciate` method and focus optimization efforts there.

12. **Algorithmic Changes**: Consider if there are any algorithmic changes that could reduce the complexity of speciation. For example, you could implement a more sophisticated clustering algorithm that reduces the number of distance calculations needed.

By implementing these strategies, you should be able to significantly improve the performance of the `speciate` method and the NEAT algorithm as a whole.
"""