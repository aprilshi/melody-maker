import numpy as np
import config
import random
import copy

class ConnectionGene:
    def __init__(self, in_node, out_node, weight, innovation):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = True
        self.innovation = innovation

class NodeGene:
    def __init__(self, id, node_type, activation_func):
        self.id = id
        self.type = node_type # 'input', 'output', or 'hidden'
        self.activation = activation_func

class Genome:
    def __init__(self, is_random=True):
        self.fitness = 0.0
        self.nodes = {} 
        self.connections = {} 
        self.next_node_id = 0
        self.innovation_counter = 0

        if is_random:
            self.initialize_base_structure()

    def initialize_base_structure(self):
        # Input 0: Time
        self.add_node('input', id=0)
        # Input 1: Bias (Crucial for preventing flatlines)
        self.add_node('input', id=1)
        
        # Outputs: 1 per pitch in PITCH_MAP
        # Start IDs from 2 to avoid overlap with inputs 0 and 1
        for i in range(len(config.PITCH_MAP)):
            out_id = i + 2
            self.add_node('output', id=out_id)
            # Connect both Time and Bias to the outputs
            self.add_connection(0, out_id)
            self.add_connection(1, out_id)

    def add_node(self, node_type, id=None):
        if id is None: id = self.next_node_id
        activation = random.choice(config.ACTIVATION_CHOICES)
        self.nodes[id] = NodeGene(id, node_type, activation)
        self.next_node_id = max(self.next_node_id, id + 1)
        return id

    def add_connection(self, in_id, out_id):
        weight = np.random.randn() * config.WEIGHT_SCALE
        innov = self.innovation_counter
        self.connections[(in_id, out_id)] = ConnectionGene(in_id, out_id, weight, innov)
        self.innovation_counter += 1

    def clone(self):
        return copy.deepcopy(self)

def mutate_genome(genome: Genome):
    # 1. Weight Mutation
    for conn in genome.connections.values():
        if random.random() < config.P_MUTATE_WEIGHT:
            conn.weight += np.random.normal(0, config.WEIGHT_PERTURB_STRENGTH)

    # 2. Add Node (Structural)
    if random.random() < config.P_ADD_NODE and genome.connections:
        conn = random.choice(list(genome.connections.values()))
        conn.enabled = False
        new_id = genome.add_node('hidden')
        genome.add_connection(conn.in_node, new_id)
        genome.add_connection(new_id, conn.out_node)

    # 3. Add Connection (Structural)
    if random.random() < config.P_ADD_CONN:
        nodes = list(genome.nodes.keys())
        in_id, out_id = random.sample(nodes, 2)
        if genome.nodes[out_id].type != 'input' and (in_id, out_id) not in genome.connections:
            genome.add_connection(in_id, out_id)
    return genome

def crossover_genomes(p1: Genome, p2: Genome):
    if p2.fitness > p1.fitness: p1, p2 = p2, p1
    child = p1.clone()
    for key, conn2 in p2.connections.items():
        if key in child.connections and random.random() < 0.5:
            child.connections[key].weight = conn2.weight
    return child