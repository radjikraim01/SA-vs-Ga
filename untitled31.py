import numpy as np
import copy
import matplotlib.pyplot as plt

# Function to generate random cities with coordinates
def generate_random_cities(num_cities):
    cities = {}
    for i in range(1, num_cities + 1):
        cities[f'p{i}'] = (np.random.rand(), np.random.rand())
    return cities

# Parameters
num_cities = 10  # Set the number of cities
population_size = 50
mutation_probability = 0.8  # Set the probability of mutation (e.g., 80%)
generations = 1000
num_runs = 100  # Number of times to run the algorithms

# Generate random cities
cities = generate_random_cities(num_cities)

# Print the cities used in the kernel
print("Cities:")
for city, coordinates in cities.items():
    print(f"{city}: {coordinates}")

# Function to calculate the total distance of a tour
def calculate_distance(tour):
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += np.linalg.norm(np.array(cities[tour[i]]) - np.array(cities[tour[i + 1]]))
    total_distance += np.linalg.norm(np.array(cities[tour[-1]]) - np.array(cities[tour[0]]))
    return total_distance

# Function to initialize a population of tours
def initialize_population(size):
    return [list(np.random.permutation(list(cities.keys()))) for _ in range(size)]

# Function to perform crossover (order crossover)
def crossover(parent1, parent2):
    start = np.random.randint(0, len(parent1))
    end = np.random.randint(start + 1, len(parent1) + 1)
    child = [-1] * len(parent1)
    child[start:end] = parent1[start:end]
    remaining = [city for city in parent2 if city not in child]
    child[:start] = remaining[:start]
    child[end:] = remaining[start:]

    # Ensure the initial city is in the child tour
    if child[0] == -1:
        initial_city = parent1[0]
        child[0] = initial_city

    # Check if the child has better fitness than both parents
    if 1 / calculate_distance(child) > min(1 / calculate_distance(parent1), 1 / calculate_distance(parent2)):
        return child
    else:
        # If not, return the better of the two parents
        return parent1 if 1 / calculate_distance(parent1) < 1 / calculate_distance(parent2) else parent2

# Function to perform mutation (swap mutation)
def mutate(tour):
    idx1, idx2 = np.random.choice(len(tour), 2, replace=False)
    tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
    return tour

# Simulated Annealing Algorithm
def simulated_annealing(initial_tour, initial_temperature, cooling_rate, num_iterations, run_number):
    current_tour = initial_tour
    current_distance = calculate_distance(current_tour)
    best_tour = copy.deepcopy(current_tour)
    best_distance = current_distance
    temperature = initial_temperature

    for _ in range(num_iterations):
        neighbor_tour = mutate(current_tour)
        neighbor_distance = calculate_distance(neighbor_tour)

        # Accept the neighbor if it has a better solution
        if neighbor_distance < current_distance or np.random.rand() < np.exp((current_distance - neighbor_distance) / temperature):
            current_tour = neighbor_tour
            current_distance = neighbor_distance

        # Update the best solution if needed
        if current_distance < best_distance:
            best_tour = copy.deepcopy(current_tour)
            best_distance = current_distance

        # Cool the temperature
        temperature *= cooling_rate

    # Print the final best distance and run number
    print(f"Simulated Annealing - Run {run_number}: Best Distance - {best_distance}")

    return best_tour, best_distance

# Genetic Algorithm
def genetic_algorithm(pop_size, mutation_prob, num_generations, run_number):
    population = initialize_population(pop_size)

    for generation in range(num_generations):
        # Evaluate fitness of each individual in the population
        fitness = [1 / calculate_distance(ind) for ind in population]

        # Select parents based on fitness
        selected_indices = np.random.choice(range(pop_size), 2 * (pop_size // 2), p=fitness / np.sum(fitness))
        parents = [population[i] for i in selected_indices]

        # Create offspring through crossover
        offspring = []
        for i in range(0, len(parents), 2):
            child1 = crossover(parents[i], parents[i + 1])
            child2 = crossover(parents[i + 1], parents[i])
            offspring.extend([child1, child2])

        # Apply mutation to offspring
        for i in range(len(offspring)):
            if np.random.rand() < mutation_prob:
                offspring[i] = mutate(offspring[i])

        # Replace the old population with the new population
        population = offspring

    # Find the best individual in the final population
    best_tour = min(population, key=lambda x: calculate_distance(x))
    best_distance = calculate_distance(best_tour)

    # Print the final best distance and run number
    print(f"Genetic Algorithm - Run {run_number}: Best Distance - {best_distance}")

    return best_distance

# Run simulated annealing first
best_distances_sa = []
for run_number in range(1, num_runs + 1):
    _, best_distance_sa = simulated_annealing(list(cities.keys()), 100.0, 0.001, 1000, run_number)
    best_distances_sa.append(best_distance_sa)

# Run genetic algorithm after simulated annealing
best_distances_ga = []
for run_number in range(1, num_runs + 1):
    best_distance_ga = genetic_algorithm(population_size, mutation_probability, generations, run_number)
    best_distances_ga.append(best_distance_ga)

# Plot separate histograms for both algorithms
fig, axs = plt.subplots(2, 1, figsize=(8, 10))

# Simulated Annealing Histogram
axs[0].hist(best_distances_sa, bins=20, color='green', alpha=0.5, edgecolor='black')
axs[0].set_title('Simulated Annealing')
axs[0].set_xlabel('Distance')
axs[0].set_ylabel('Frequency')

# Genetic Algorithm Histogram
axs[1].hist(best_distances_ga, bins=20, color='blue', alpha=0.5, edgecolor='black')
axs[1].set_title('Genetic Algorithm')
axs[1].set_xlabel('Distance')
axs[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
