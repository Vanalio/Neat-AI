from population import Population
from config import Config

config = Config("config.ini", "DEFAULT")


def neat():
    print("\n#############################################")
    print(f"# GENERATION: 0")
    population = Population(first=True)

    for generation in range(config.generations):
        print("\n#############################################")
        print(f"# GENERATION: {population.generation}")

        population.evolve()

        if generation % config.population_save_interval == 0:
            population.save_genomes_to_file(f"saves/population_gen_{generation}.pkl")

    population.save_genomes_to_file("saves/final_population.pkl")


if __name__ == "__main__":
    neat()
