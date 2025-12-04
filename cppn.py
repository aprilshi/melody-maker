import numpy as np
import config

# --- CPPN Class (template) ---
class CPPN:
    def __init__(self, genome):
        self.num_inputs = 1
        self.num_hidden = config.NUM_HIDDEN_NODES
        self.num_outputs = len(config.PITCH_MAP)

        # load genetic material from Genome
        self.weights_in_to_hidden = genome.weights_in_to_hidden
        self.weights_hidden_to_out = genome.weights_hidden_to_out
        self.biases_hidden = genome.biases_hidden
        self.biases_output = genome.biases_output
        self.hidden_activations = genome.hidden_activations
        self.output_activations = genome.output_activations

    def forward_pass(self, normalized_timestep):
        """Performs a forward pass to get pitch/rest preferences."""
        input_layer = np.array([[normalized_timestep]])

        # Hidden Layer
        hidden_input = np.dot(input_layer, self.weights_in_to_hidden) + self.biases_hidden
        hidden_output = np.zeros_like(hidden_input)
        for i in range(self.num_hidden):
            hidden_output[0, i] = self.hidden_activations[i](hidden_input[0, i])

        # Output Layer
        output_input = np.dot(hidden_output, self.weights_hidden_to_out) + self.biases_output
        raw_output = np.zeros_like(output_input)
        for i in range(self.num_outputs):
            # apply the chosen activation function
            raw_output[0, i] = self.output_activations[i](output_input[0, i])

        return raw_output.flatten()