import multiprocessing
import random
import pickle
import gymnasium as gym
from gymnasium.wrappers import TransformReward
import numpy as np
import torch

from managers import IdManager
from genome import Genome
from species import Species
from neural_network import NeuralNetwork

from config import Config

config = Config("config.ini", "DEFAULT")

class Population:
    def __init__(self, initial=False):
        self.genomes = {}
        self.species = {}
        self.generation = 0
        self.max_fitness = None
        self.best_genome = None
        self.environment = None
        if initial:
            self._initialize_neurons()
            self._initial_population()
            self._initial_speciation()

    def _initialize_neurons(self):
        self.input_ids = [IdManager.get_new_id() for _ in range(config.input_neurons)]
        self.output_ids = [IdManager.get_new_id() for _ in range(config.output_neurons)]
        print(" # Initial population neurons:")
        print(" # INPUT:", len(self.input_ids), self.input_ids)
        print(" # OUTPUT:", len(self.output_ids), self.output_ids)
        print(" # HIDDEN:", config.initial_hidden_neurons, "\n #############################################\n")

    def _initial_population(self):
        for _ in range(config.population_size):
            genome = Genome().create(self.input_ids, self.output_ids)
            self.genomes[genome.id] = genome
            self.generation = 0

    def _initial_speciation(self):
        print("Initial speciation...")
        previous_species_count = None
        stabilized = False
        stabilizing = 0
        tries = 0

        while not stabilized \
              and len([s for s in self.species.values() if s.genomes]) != config.target_species \
              and tries < config.speciation_max_tries:
            
            self.remove_empty_species() # WHY?
            self.speciate()

            tries += 1
            print(f"Not empty species: {len([s for s in self.species.values() if s.genomes])}, distance set to: {config.distance_threshold}")

            if len(self.species) == previous_species_count:
                stabilizing += 1
                stabilized = stabilizing >= config.speciation_stabilization
            else:
                stabilizing = 0
                stabilized = False

            previous_species_count = len(self.species)

    def new_speciation(self):
        tries = 0

        while not (
                  config.target_species / config.target_species_tolerance < 
                  len([s for s in self.species.values() if s.genomes]) < 
                  config.target_species * config.target_species_tolerance
                  ) \
              and tries < config.speciation_max_tries:

            self.speciate()
            tries += 1
            print(f"Not empty species: {len([s for s in self.species.values() if s.genomes])}, distance set to: {config.distance_threshold}")

    def speciate(self):
        def find_species_for_genome(genome):
            for species_instance in self.species.values():
                is_same, matching_connections = species_instance.is_same_species(genome, config.distance_threshold)
                if is_same:
                    return species_instance, matching_connections
            return None, None

        # WHY?
        for species_instance in self.species.values():
            species_instance.genomes = {}
            species_instance.elites = {}

        for genome in self.genomes.values():
            species_instance, matching_connections = find_species_for_genome(genome)
            if species_instance:
                species_instance.add_genome(genome)
                genome.matching_connections = matching_connections
            else:
                new_species = Species()
                self.species[new_species.id] = new_species
                new_species.add_genome(genome)
                genome.matching_connections = len(genome.connection_genes)

        non_empty_species = len([s for s in self.species.values() if s.genomes])
        species_ratio = non_empty_species / config.target_species

        adjustment_factor = (
            1.0 - config.distance_adj_factor
            if species_ratio < 1.0
            else 1.0 + config.distance_adj_factor
        )

        config.distance_threshold *= adjustment_factor

    def remove_empty_species(self):
        species_to_remove = []
        for species_instance in self.species.values():
            if not species_instance.genomes:
                species_to_remove.append(species_instance.id)
        for species_id in species_to_remove:
            del self.species[species_id] 

    def evolve(self):
        print("Evaluation...")
        self.evaluate()
        print("Speciating...")
        self.new_speciation()
        print("Population assessment...")
        self.assess()
        print("Pruning...")
        self.prune()
        self.species_fitness()
        self.sort_species()
        self.print_population_info()
        print("Forming next generation...")
        self.form_next_generation()
 
    def evaluate(self):
        if config.run_mode == "parallel":
            self.evaluate_parallel()
        elif config.run_mode == "serial":
            self.evaluate_serial()
        elif config.run_mode == "dumb":
            self.evaluate_dumb()
        else:
            raise ValueError("No valid evaluation method specified.")
        
        # calculate fitness for each genome
        for genome in self.genomes.values():
            genome.fitness = self.relu_offset(genome.total_reward)
            #print(f"Genome {genome.id}: total reward {genome.total_reward}, fitness {genome.fitness}")

    def evaluate_serial(self):
        for genome in self.genomes.values():
            _, total_reward = self.evaluate_genome(genome)
            genome.total_reward = total_reward

    def evaluate_parallel(self):
        with multiprocessing.Pool(config.procs) as pool:
            # Prepare arguments for each genome (each argument should be a tuple)
            args = [(genome,) for genome in self.genomes.values()]

            # Map the function across the genomes
            results = pool.starmap(self.evaluate_genome, args)

            # Process the results to update fitness values
            for genome_id, total_reward in results:
                self.genomes[genome_id].total_reward = total_reward

    def evaluate_dumb(self):
        for genome in self.genomes.values():
            genome.total_reward = 1

    def evaluate_genome(self, genome, batch_size=config.batch_size, render_mode=None, max_episode_steps=config.max_episode_steps):
        environments = [TransformReward(gym.make("BipedalWalker-v3", hardcore=True, render_mode=render_mode, max_episode_steps=max_episode_steps), lambda r: r if r > 0 else 0) for _ in range(batch_size)]
        observations = [environment.reset(seed=self.generation * config.env_seed + i) for i, environment in enumerate(environments)]

        if isinstance(observations[0], tuple):
            observations = [observation[0] for observation in observations]

        observations_array = np.array(observations)
        observations_tensor = torch.tensor(observations_array, dtype=torch.float32)

        try:
            neural_network = NeuralNetwork(genome)
        except KeyError as e:
            print(f"Error processing genome. KeyError: {e.args[0]}")
            print(f"Genome details: {genome}")

        neural_network.reset_states()

        done = [False] * batch_size
        total_rewards = [0] * batch_size

        while not all(done):
            #output_logits = neural_network.forward_batch(observations_tensor)
            #action_probabilities = F.softmax(output_logits, dim=1)
            #actions = torch.argmax(action_probabilities, dim=1).cpu().numpy()
 
            actions = neural_network.forward_batch(observations_tensor).cpu().numpy()

            new_observations, new_rewards, new_done = [], [], []
            for i, (environment, action) in enumerate(zip(environments, actions)):
                if not done[i]:
                    observation, reward, terminated, truncated, _ = environment.step(action)
                    new_observations.append(observation)
                    total_rewards[i] += reward
                    new_done.append(terminated or truncated)
                else:
                    new_observations.append(observations[i])
                    new_rewards.append(0)
                    new_done.append(True)

            observations = new_observations
            done = new_done
            observations_tensor = torch.tensor(np.array(new_observations), dtype=torch.float32)
        
        neural_network.reset_states()
        
        for environment in environments:
            environment.close()

        avg_reward = sum(total_rewards) / batch_size

        return genome.id, avg_reward

    def relu_offset(self, reward):
        return max(0, reward + config.fitness_offset)

    def assess(self):
        self.sort_genomes_in_species()
        self.find_elites()
        self.update_stagnation()
        self.species_fitness()
        self.sort_species()
        self.find_best_genome()

    def sort_genomes_in_species(self):
        for _, species in self.species.items():
            if species.genomes:
                sorted_genomes = sorted(
                    species.genomes.values(),
                    key=lambda genome: genome.fitness,
                    reverse=True,
                )
                species.genomes = {genome.id: genome for genome in sorted_genomes}

    def find_elites(self):
        for _, species in self.species.items():
            if species.genomes:
                # Convert dictionary values to a list and take the first 'config.elites_per_species' elements
                top_elites = list(species.genomes.values())[:config.elites_per_species]
                species.elites = {genome.id: genome for genome in top_elites}

    def update_stagnation(self):
        for _, species in self.species.items():
            if species.genomes:
                first_elite_genome = next(iter(species.elites.values()))
                if species.all_time_best_fitness is None or first_elite_genome.fitness > species.all_time_best_fitness:
                    species.all_time_best_fitness = first_elite_genome.fitness
                    species.generations_without_improvement = 0
                else:
                    species.generations_without_improvement += 1

    def species_fitness(self):
        for _, species in self.species.items():
            if species.genomes:
                species.total_fitness = sum(
                    genome.fitness for genome in species.genomes.values()
                )
                species.average_shared_fitness = species.total_fitness / (
                    len(species.genomes) ** 2
                )
            else:
                species.average_shared_fitness = 0
                species.total_fitness = 0

    def sort_species(self):
        sorted_species = sorted(
        self.species.items(),
        key=lambda item: item[1].average_shared_fitness,
        reverse=True,
        )
        self.species = {species_id: species for species_id, species in sorted_species}

    def find_best_genome(self):
        self.max_fitness = 0
        self.best_genome = None
        for _, genome in self.genomes.items():
            if self.max_fitness is None or genome.fitness > self.max_fitness:
                self.max_fitness = genome.fitness
                self.best_genome = genome
        
        if self.best_genome:
            self.best_genome.save_to_file(f"saves/best_genome_{self.generation}.pkl")

        return self.best_genome

    def prune(self):
        self.prune_weak_species()
        self.prune_stale_species()
        self.prune_genomes_from_species()

    def prune_weak_species(self):
        max_removal_count = len(self.species) - config.min_species
        removal_count = 0
        if removal_count < max_removal_count:

            if not 0 < config.keep_best_species <= 1:
                raise ValueError("config.keep_best_species must be between 0 and 1.")
            original_count = len(self.species)
            cutoff = max(1, int(original_count * config.keep_best_species))
            self.species = dict(list(self.species.items())[:cutoff])

    def prune_stale_species(self):
        removal_count = 0
        species_to_remove = []

        for species_id, species_instance in self.species.items():
            if species_instance.generations_without_improvement >= config.max_stagnation:
                # Check if removing this species would still keep the total above or equal to min_species
                if len(self.species) - removal_count > config.min_species:
                    species_to_remove.append(species_id)
                    removal_count += 1
                    print(f"Species {species_instance.id} removed due to stagnation.")
                else:
                    break  # Stop if further removal would reduce total count below min_species

        # Remove the identified species outside the loop
        for species_id in species_to_remove:
            del self.species[species_id]

    def prune_genomes_from_species(self):
        for species_instance in self.species.values():

            #print(f"Species {species_instance.id} has {len(species_instance.genomes)} genomes.")
            species_instance.cull(config.keep_best_genomes_in_species)
            #print(f"Species {species_instance.id} has {len(species_instance.genomes)} genomes after culling.")

    def print_population_info(self):
        for _, species in self.species.items():
            if species.genomes:
                
                species_genomes = len(species.genomes)
                avg_connections = int(sum([len(genome.connection_genes) for genome in species.genomes.values()]) / species_genomes)
                avg_neurons = int(sum([len(genome.neuron_genes) for genome in species.genomes.values()]) / species_genomes) - config.input_neurons - config.output_neurons
                avg_disabled_neurons = int(sum([len([n for n in genome.neuron_genes.values() if not n.enabled]) for genome in species.genomes.values()]) / species_genomes)
                avg_disabled_connections = int(sum([len([c for c in genome.connection_genes.values() if not c.enabled]) for genome in species.genomes.values()]) / species_genomes)
                #avg_matching_connections = int(sum([genome.matching_connections for genome in species.genomes.values()]) / species_genomes)
                
                print(
                    f"Species: {species.id}, Age/Stagn: {species.age}/{species.generations_without_improvement}", \
                    f"Size: {len(species.genomes)}, Elites: {[e.id for e in species.elites.values()]}", \
                    f"shared/alltime: {int(species.average_shared_fitness)}/{int(species.all_time_best_fitness)}, ", \
                    f"conn/dis: {avg_connections}/{avg_disabled_connections}, hid/dis: {avg_neurons}/{avg_disabled_neurons}"
                )

        print(f"Not empty species: {len([s for s in self.species.values() if s.genomes])}, distance set to: {config.distance_threshold}")

        if self.best_genome:
            print(
                  f"BEST GENOME: {self.best_genome.id}, Fit: {self.max_fitness}, "
                  f"conn/dis: {len(self.best_genome.connection_genes)}/{len([c for c in self.best_genome.connection_genes.values() if not c.enabled])}, "
                  f"hid/dis: {len(self.best_genome.neuron_genes) - config.input_neurons - config.output_neurons}/{len([n for n in self.best_genome.neuron_genes.values() if not n.enabled])}"
                 )

            for n in self.best_genome.neuron_genes.values():
                if n.layer != "input":
                    # print id activation and bias of enabled neurons not in input layer
                    if n.enabled and n.layer != "input":
                        print(f"Neuron {n.id}: {n.activation}, {n.bias}")

    def form_next_generation(self):
        next_gen = {}

        next_gen_elites = self.carry_over_elites()

        next_gen_crossovers = self.reproduce(next_gen_elites)
        self.mutate_genomes(next_gen_crossovers)

        next_gen.update(next_gen_elites)
        next_gen.update(next_gen_crossovers)

        self.genomes = next_gen
        self.generation += 1

        for species_instance in self.species.values():
            species_instance.age += 1
        
        self.purge_species()
    
    def carry_over_elites(self):
        next_gen_elites = {}
        for species_instance in self.species.values():
            next_gen_elites.update(species_instance.elites)
        return next_gen_elites

    def reproduce(self, next_gen_elites):
        next_gen_crossovers = {}
        needed_offspring = config.population_size - len(next_gen_elites)
        for species_instance in self.species.values():
            if not species_instance.genomes:
                continue
            offspring_count = self.get_offspring_count(species_instance, needed_offspring)
            offspring = species_instance.produce_offspring(offspring_count)
            next_gen_crossovers.update(offspring)

        while len(next_gen_crossovers) + len(next_gen_elites) < config.population_size:
            next_gen_crossovers.update(self.random_species().produce_offspring(1))

        return next_gen_crossovers

    def mutate_genomes(self, genomes):
        for _, genome in genomes.items():
            genome.mutate()

    def get_offspring_count(self, species_instance, needed_offspring):

        rank = list(self.species.keys()).index(species_instance.id) + 1
        total_rank_sum = sum(1 / (i ** 0.5) for i in range(1, len(self.species) + 1))

        offspring_count = int((needed_offspring * (1 / rank)) / total_rank_sum)

        return offspring_count

    def random_species(self):
        species_with_genomes = [s for s in self.species.values() if s.genomes]
        if not species_with_genomes:
            raise ValueError("No species available to choose from or no species with genomes.")
        return random.choice(species_with_genomes)

    def purge_species(self):
        # removes from all species all genomes that are not representative nor in population
        for species_instance in self.species.values():
            for genome_id in list(species_instance.genomes.keys()):
                if genome_id not in self.genomes and genome_id != species_instance.representative.id:
                    del species_instance.genomes[genome_id]

    def save_genomes_to_file(self, file_path):
        with open(file_path, "wb") as file:
            pickle.dump(self.genomes, file)

    def load_genomes_from_file(self, file_path):
        with open(file_path, "rb") as file:
            self.genomes = pickle.load(file)
