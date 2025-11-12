import numpy as np
import config
import random

# --- Genome Class ---
class Genome:
    """
    Contains the evolvable parameters of a single CPPN.
    """
    def __init__(self, is_random=True):
        self.fitness = 0.0

        # Genotype components (weights, biases, and activation functions)
        if is_random:
            self.randomize()
        else:
            # Placeholder for child genome initialization
            self.weights_in_to_hidden = None
            self.weights_hidden_to_out = None
            self.biases_hidden = None
            self.biases_output = None
            self.hidden_activations = None
            self.output_activations = None

    def randomize(self):
        """Initializes all parameters randomly."""
        # Weights and biases are initialized with small random numbers
        scale = 0.5
        self.weights_in_to_hidden = np.random.randn(1, config.NUM_HIDDEN_NODES) * scale
        self.biases_hidden = np.random.randn(1, config.NUM_HIDDEN_NODES) * scale
        self.weights_hidden_to_out = np.random.randn(config.NUM_HIDDEN_NODES, config.TOTAL_OUTPUTS) * scale
        self.biases_output = np.random.randn(1, config.TOTAL_OUTPUTS) * scale

        # Activation functions are chosen randomly
        self.hidden_activations = np.random.choice(config.ACTIVATION_CHOICES, config.NUM_HIDDEN_NODES, replace=True)
        self.output_activations = np.random.choice(config.ACTIVATION_CHOICES, config.TOTAL_OUTPUTS, replace=True)

    def clone(self):
        """Creates an identical copy of the genome."""
        new_genome = Genome(is_random=False)
        new_genome.weights_in_to_hidden = self.weights_in_to_hidden.copy()
        new_genome.weights_hidden_to_out = self.weights_hidden_to_out.copy()
        new_genome.biases_hidden = self.biases_hidden.copy()
        new_genome.biases_output = self.biases_output.copy()
        new_genome.hidden_activations = self.hidden_activations.copy()
        new_genome.output_activations = self.output_activations.copy()
        new_genome.fitness = self.fitness
        return new_genome
    
# --- Genome Operators ---

# Crossover
def crossover_genomes(parent1: Genome, parent2: Genome):
    """
    Performs Uniform Crossover on two parent genomes.
    For each parameter, randomly selects the value from Parent1 or Parent2.
    """
    # Ensure parent1 is the fitter parent (Elitism in crossover)
    if parent2.fitness > parent1.fitness:
        parent1, parent2 = parent2, parent1

    child = Genome(is_random=False)
    
    # Helper to perform uniform crossover on NumPy arrays
    def uniform_crossover_array(arr1, arr2):
        mask = np.random.rand(*arr1.shape) < 0.5
        child_arr = np.where(mask, arr1, arr2)
        # For simplicity, we choose randomly, rather than always from the fitter parent
        return child_arr

    # Helper to perform uniform crossover on Activation arrays (function objects)
    def uniform_crossover_activations(act1, act2):
        child_act = []
        for a1, a2 in zip(act1, act2):
            child_act.append(a1 if random.random() < 0.5 else a2)
        return np.array(child_act)

    # Crossover all weight/bias arrays
    child.weights_in_to_hidden = uniform_crossover_array(parent1.weights_in_to_hidden, parent2.weights_in_to_hidden)
    child.weights_hidden_to_out = uniform_crossover_array(parent1.weights_hidden_to_out, parent2.weights_hidden_to_out)
    child.biases_hidden = uniform_crossover_array(parent1.biases_hidden, parent2.biases_hidden)
    child.biases_output = uniform_crossover_array(parent1.biases_output, parent2.biases_output)

    # Crossover activation arrays
    child.hidden_activations = uniform_crossover_activations(parent1.hidden_activations, parent2.hidden_activations)
    child.output_activations = uniform_crossover_activations(parent1.output_activations, parent2.output_activations)

    return child

# Mutation
def mutate_genome(genome: Genome):
    """
    Applies two types of mutation: weight adjustment and activation swap.
    """
    # 1. Weight Mutation
    def mutate_array(arr):
        mutation_mask = np.random.rand(*arr.shape) < config.P_MUTATE_WEIGHT
        
        # Perturbation (Small adjustment for most mutations)
        perturbation = np.random.normal(0, config.WEIGHT_PERTURB_STRENGTH, arr.shape)
        arr[mutation_mask] += perturbation[mutation_mask]
        
        # Total reset (Randomly reassign a few weights completely)
        reset_mask = np.random.rand(*arr.shape) < 0.05 # 5% chance of reset if selected for mutation
        arr[reset_mask] = np.random.randn(np.sum(reset_mask)) * 0.5
        return arr

    genome.weights_in_to_hidden = mutate_array(genome.weights_in_to_hidden)
    genome.weights_hidden_to_out = mutate_array(genome.weights_hidden_to_out)
    genome.biases_hidden = mutate_array(genome.biases_hidden)
    genome.biases_output = mutate_array(genome.biases_output)

    # 2. Activation Mutation (Function Swap)
    def mutate_activations(act_array):
        for i in range(len(act_array)):
            if random.random() < config.P_MUTATE_ACTIVATION:
                # Select a new function randomly from the choices
                new_func = random.choice(config.ACTIVATION_CHOICES)
                act_array[i] = new_func
        return act_array

    genome.hidden_activations = mutate_activations(genome.hidden_activations)
    genome.output_activations = mutate_activations(genome.output_activations)
    
    return genome