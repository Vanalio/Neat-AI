from neat_classes import Population
from config import Config

config = Config("config.ini", "DEFAULT")

def neat():
    
    print(f"\nGeneration: INITIAL")

    population = Population(first=True)

    max_fitness = []

    for generation in range(config.generations):
        print(f"\nGeneration: {generation + 1}")

        population.evolve()

        max_fitness.append(population.max_fitness)

        if generation % config.population_save_interval == 0:
            population.save_genomes_to_file(f"population_gen_{generation}.pkl")

    population.save_genomes_to_file("final_population.pkl")

if __name__ == "__main__":
    neat()