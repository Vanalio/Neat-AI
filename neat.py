import argparse
from cgi import test
import multiprocessing  
import matplotlib.pyplot as plt

from population import Population
from genome import Genome
from visualization import NeatVisualizer

from config import Config

config = Config("config.ini", "DEFAULT")


def neat():
    
    print("#############################################")
    print(f"# NEAT ALGORITHM\n# Population size: {config.population_size}\n# Generations: {config.generations}\n# Run mode: {config.run_mode}\n#")

    population = Population(initial=True)

    for generation in range(config.generations):
        print("#############################################")
        print(f"# GENERATION: {population.generation}")

        population.evolve()

        test_genome = population.best_genome.copy(keep_id=False, keep_innovation=False)

        test_genome_reward = population.render_genome(test_genome)
        visualizer.visualize_genome(test_genome)
        visualizer.plot_rewards(generation, test_genome_reward)

        if generation % config.population_save_interval == 0:
            population.save_genomes_to_file(f"saves/population_gen_{generation}.pkl")

    population.save_genomes_to_file("saves/final_population.pkl")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    visualizer = NeatVisualizer()    

    parser = argparse.ArgumentParser(description="Run NEAT algorithm")
    parser.add_argument("--visualize", help="Visualize a saved genome file", type=str)
    parser.add_argument("--test", help="Test and render a specific genome file", type=str)
    args = parser.parse_args()

    if args.visualize:
        # Load and visualize the genome
        genome = Genome.load_from_file(args.visualize)
        visualizer.visualize_genome(genome)

    elif args.test:
        # Load and test/render the specific genome
        genome = Genome.load_from_file(args.test)
        # add genome to a new population
        population = Population(initial=False)
        population.genomes.append(genome)
        population.render_genome(genome)

    else:
        # Run the NEAT algorithm normally
        neat()
