import argparse
import multiprocessing
import random

from population import Population
from genome import Genome
from visualization import NeatVisualizer

from config import Config

config = Config("config.ini", "DEFAULT")


def neat():
    
    print(" #############################################")
    print(
          " # NEAT ALGORITHM\n",
          f"# Population size: {config.population_size}\n",
          f"# Generations: {config.generations}\n",
          f"# Batch size: {config.batch_size}\n",
          f"# Run mode: {config.run_mode}\n",
          f"#"
         )

    population = Population(initial=True)

    for generation in range(config.generations):
        print("\n#############################################")
        print(f"# GENERATION: {population.generation}")

        population.evolve()

        # best genome exists, test genome is best genome, otherwise test genome is taken randomly from population
        if population.best_genome is not None:
            test_genome = population.best_genome
        else:
            print("******** No best genome found, testing random genome ********")
            # Select a random key from the dictionary
            random_key = random.choice(list(population.genomes.keys()))
            test_genome = population.genomes[random_key]
        
        population.evaluate_genome(test_genome, batch_size=1, render_mode="human")
        
        visualizer.visualize_genome(test_genome)
        visualizer.plot_fitness(generation, test_genome.fitness)

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
        ...

    else:
        # Run the NEAT algorithm normally
        neat()
