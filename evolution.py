from genome import Genome
from cppn import CPPN
import play_rate_music
import config
import melody_generation
import numpy as np
import random

def run_evolution():
    """Manages the creation, evaluation, and evolution of the CPPN population."""
    population = [Genome() for _ in range(config.POPULATION_SIZE)]
    print(f"Starting CPPN Evolution with POPULATION_SIZE={config.POPULATION_SIZE} for {config.NUM_GENERATIONS} generations.")

    for generation in range(config.NUM_GENERATIONS):
        print(f"\n--- GENERATION {generation + 1}/{config.NUM_GENERATIONS} ---")
        
        # 1. Evaluation
        for i, genome in enumerate(population):
            print(f"Evaluating Genome {i+1}/{config.POPULATION_SIZE}...")
            cppn = CPPN(genome)
            melody = play_rate_music.generate_melody(cppn)
            genome.fitness = play_rate_music.get_human_score(melody)
        
        # 2. Sort by Fitness (Descending)
        population.sort(key=lambda g: g.fitness, reverse=True)
        
        best_fitness = population[0].fitness
        avg_fitness = np.mean([g.fitness for g in population])
        
        print(f"\nGeneration {generation + 1} Summary:")
        print(f"  Best Fitness: {best_fitness:.2f}")
        print(f"  Average Fitness: {avg_fitness:.2f}")
        print(f"  Top 5 Genomes' Scores: {[f'{g.fitness:.2f}' for g in population[:5]]}")

        if generation == config.NUM_GENERATIONS - 1:
            break # Stop after the last evaluation

        # 3. Reproduction (Creating the next generation)
        next_population = []
        
        # Elitism: Keep the best parents directly (they are already sorted)
        for i in range(config.NUM_PARENTS_TO_KEEP):
            next_population.append(population[i].clone())

        # Select parents for breeding (e.g., top 50% as the breeding pool)
        breeding_pool = population[:config.POPULATION_SIZE // 2]
        
        # Fill the rest of the population
        while len(next_population) < config.POPULATION_SIZE:
            # Select two random parents from the pool (can be the same parent, but crossover will still apply)
            parent1 = random.choice(breeding_pool)
            parent2 = random.choice(breeding_pool)

            # Crossover to create a child
            child = genome.crossover_genomes(parent1, parent2)
            
            # Mutation
            child = genome.mutate_genome(child)
            
            next_population.append(child)

        population = next_population

    print("\n--- EVOLUTION COMPLETE ---")
    print("The best genome is the first one in the final sorted list.")
    
    final_best_cppn = CPPN(population[0])
    final_melody = melody_generation.generate_melody(final_best_cppn)
    print(f"\nFinal Best Fitness: {population[0].fitness:.2f}")
    print(f"Final Best Melody (MIDI Notes): {final_melody}")