from population_and_species import Population
from config import Config

config = Config("config.ini", "DEFAULT")

def neat():
    
    # Create initial population
    print(f"\nGENERATION: zero")
    population = Population(first=True)

    max_fitness = []

    # Evolve for n generations
    for generation in range(config.generations):
        print(f"\nGENERATION: {generation + 1}")

        population.evolve()

        max_fitness.append(population.max_fitness)

        # Save population every n generations
        if generation % config.population_save_interval == 0:
            population.save_genomes_to_file(f"saves/population_gen_{generation}.pkl")

    # Save final population
    population.save_genomes_to_file("saves/final_population.pkl")

if __name__ == "__main__":
    neat()
