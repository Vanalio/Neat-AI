import argparse
import multiprocessing  
import matplotlib.pyplot as plt

from population import Population
from genome import Genome
from visualization import NeatVisualizer

from config import Config

config = Config("config.ini", "DEFAULT")


def neat():
    visualizer = NeatVisualizer()
    
    print("\n#############################################")
    print(f"# GENERATION: 0")
    population = Population(first=True)

    for generation in range(config.generations):
        print("\n#############################################")
        print(f"# GENERATION: {population.generation}")

        best_genome = population.evolve()

        total_reward = population.test_genome(best_genome)
        visualizer.visualize_genome(best_genome)
        visualizer.plot_rewards(generation, total_reward)

        if generation % config.population_save_interval == 0:
            population.save_genomes_to_file(f"saves/population_gen_{generation}.pkl")

    population.save_genomes_to_file("saves/final_population.pkl")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    
    parser = argparse.ArgumentParser(description="Run NEAT algorithm")
    
    parser.add_argument("--visualize", help="Visualize a saved genome file", type=str)
    parser.add_argument("--test", help="Test and render a specific genome file", type=str)
    
    args = parser.parse_args()

    if args.visualize:
        # Load and visualize the genome
        genome = Genome.load_from_file(args.visualize)
        fig, ax = plt.subplots()
        visualize_genome(genome, ax)
        plt.show()
    elif args.test:
        # Load and test/render the specific genome
        genome = Genome.load_from_file(args.test)
        population = Population(first=False)
        population.test_genome(genome)
    else:
        # Run the NEAT algorithm normally
        neat()
