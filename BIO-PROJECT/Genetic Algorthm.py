import random
import numpy as np
import matplotlib.pyplot as plt

# Define a City with x, y coordinates
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        return np.sqrt((self.x - city.x) ** 2 + (self.y - city.y) ** 2)

    def __repr__(self):
        return f"({self.x}, {self.y})"


# Fitness function to calculate total distance of the route
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def route_distance(self):
        if self.distance == 0:
            path_distance = 0
            for i in range(len(self.route)):
                from_city = self.route[i]
                to_city = self.route[(i + 1) % len(self.route)]
                path_distance += from_city.distance(to_city)
            self.distance = path_distance
        return self.distance

    def route_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.route_distance())
        return self.fitness


# Create a random route from cities
def create_route(city_list):
    route = random.sample(city_list, len(city_list))
    return route


# Create the initial population of routes
def initial_population(pop_size, city_list):
    population = []
    for _ in range(pop_size):
        population.append(create_route(city_list))
    return population


# Rank routes based on their fitness
def rank_routes(population):
    fitness_results = {}
    for i in range(len(population)):
        fitness_results[i] = Fitness(population[i]).route_fitness()
    return sorted(fitness_results.items(), key=lambda x: x[1], reverse=True)


# Selection using Roulette Wheel method
def selection(pop_ranked, elite_size):
    selection_results = []
    df = sum([pop_ranked[i][1] for i in range(len(pop_ranked))])
    selection_probabilities = [pop_ranked[i][1] / df for i in range(len(pop_ranked))]

    for _ in range(elite_size):
        selection_results.append(pop_ranked[_][0])
    for _ in range(len(pop_ranked) - elite_size):
        pick = random.choices(pop_ranked, weights=selection_probabilities, k=1)[0][0]
        selection_results.append(pick)
    return selection_results


# Create mating pool
def mating_pool(population, selection_results):
    matingpool = []
    for i in range(len(selection_results)):
        matingpool.append(population[selection_results[i]])
    return matingpool


# Crossover: Ordered crossover method
def crossover(parent1, parent2):
    child = []
    child_p1 = []
    child_p2 = []

    gene_a = int(random.random() * len(parent1))
    gene_b = int(random.random() * len(parent1))

    start_gene = min(gene_a, gene_b)
    end_gene = max(gene_a, gene_b)

    for i in range(start_gene, end_gene):
        child_p1.append(parent1[i])

    child_p2 = [item for item in parent2 if item not in child_p1]

    child = child_p1 + child_p2
    return child


# Perform crossover on the population
def crossover_population(mating_pool, elite_size):
    children = []
    pool = random.sample(mating_pool, len(mating_pool))

    for i in range(elite_size):
        children.append(mating_pool[i])

    for i in range(len(mating_pool) - elite_size):
        child = crossover(pool[i], pool[len(mating_pool) - i - 1])
        children.append(child)
    return children


# Mutate by swapping two cities
def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swap_with]

            individual[swapped] = city2
            individual[swap_with] = city1
    return individual


# Perform mutation on the population
def mutate_population(population, mutation_rate):
    mutated_pop = []
    for ind in range(0, len(population)):
        mutated_ind = mutate(population[ind], mutation_rate)
        mutated_pop.append(mutated_ind)
    return mutated_pop


# Create the next generation
def next_generation(current_gen, elite_size, mutation_rate):
    pop_ranked = rank_routes(current_gen)
    selection_results = selection(pop_ranked, elite_size)
    matingpool = mating_pool(current_gen, selection_results)
    children = crossover_population(matingpool, elite_size)
    next_gen = mutate_population(children, mutation_rate)
    return next_gen


# Run the genetic algorithm
def genetic_algorithm(city_list, pop_size, elite_size, mutation_rate, generations):
    pop = initial_population(pop_size, city_list)
    print(f"Initial distance: {1 / rank_routes(pop)[0][1]}")

    for i in range(0, generations):
        pop = next_generation(pop, elite_size, mutation_rate)

    print(f"Final distance: {1 / rank_routes(pop)[0][1]}")
    best_route_index = rank_routes(pop)[0][0]
    best_route = pop[best_route_index]
    return best_route


# Visualize the route
def plot_route(route):
    x = [city.x for city in route]
    y = [city.y for city in route]
    plt.plot(x, y, 'o-', label="Route")
    plt.title("Best TSP Route Found")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


# Driver code
if __name__ == "__main__":
    # Create list of cities
    city_list = []
    for _ in range(25):
        city_list.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

    # Run Genetic Algorithm
    best_route = genetic_algorithm(city_list, pop_size=100, elite_size=20, mutation_rate=0.01, generations=500)

    # Plot the best route
    plot_route(best_route)
