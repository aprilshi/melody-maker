import numpy as np
import config

class CPPN:
    def __init__(self, genome):
        self.genome = genome
        self.cache = {}

    def forward_pass(self, normalized_t):
        # Map node 0 to Time and node 1 to a constant Bias of 1.0
        self.cache = {0: normalized_t, 1: 1.0} 
        outputs = []
        
        # Only grab nodes marked as 'output'
        output_ids = [n.id for n in self.genome.nodes.values() if n.type == 'output']
        
        for out_id in sorted(output_ids):
            outputs.append(self.compute_node(out_id))
        return np.array(outputs)

    def compute_node(self, node_id):
        if node_id in self.cache: return self.cache[node_id]
        
        node = self.genome.nodes[node_id]
        incoming_sum = 0
        for (in_id, out_id), conn in self.genome.connections.items():
            if out_id == node_id and conn.enabled:
                incoming_sum += self.compute_node(in_id) * conn.weight
        
        result = node.activation(incoming_sum)
        self.cache[node_id] = result
        return result