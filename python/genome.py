import random
import copy


class Gen:
    """A single gene representing one neural network connection.

    Each gene encodes:
      - Which layer the connection targets (hidden or output)
      - The source neuron index (from the input layer)
      - The target neuron index (in the destination layer)
      - The connection weight (-1 to 1)
    """

    def __init__(self):
        # Does this connection go to the hidden layer (True) or output layer (False)?
        self.source_hidden_layer = random.random() < 0.5

        # Source: always one of the 7 input neurons
        self.id_source_neuron = random.randint(0, 6)

        # Target depends on the destination layer
        if self.source_hidden_layer:
            self.id_target_neuron = random.randint(0, 6)  # 7 hidden neurons
        else:
            self.id_target_neuron = random.randint(0, 1)  # 2 output neurons

        # Connection strength, randomly initialized between -1 and 1
        self.weight = random.uniform(-1, 1)


class Genome:
    """The complete genetic code of a dinosaur.

    A genome holds 16 genes (neural network connections) and bias
    values for the hidden and output layers.

    The genetic algorithm uses three operations on genomes:
      - copy()      → clone without changes (elitism)
      - mutate()    → clone with a few random gene replacements
      - crossover() → combine genes from two parents
    """

    def __init__(self):
        self.length = 16
        self.genes = [Gen() for _ in range(self.length)]

        # Bias values shift the neuron's activation threshold
        self.hidden_layer_bias = [random.uniform(-1, 1) for _ in range(7)]
        self.output_layer_bias = [random.uniform(-1, 1) for _ in range(2)]

    def copy(self):
        """Create an exact deep copy of this genome."""
        new_genome = Genome()
        new_genome.genes = copy.deepcopy(self.genes)
        new_genome.hidden_layer_bias = self.hidden_layer_bias[:]
        new_genome.output_layer_bias = self.output_layer_bias[:]
        return new_genome

    def mutate(self):
        """Return a mutated copy: replace 1–4 random genes with new ones."""
        mutated = self.copy()
        num_mutations = random.randint(1, 4)
        for _ in range(num_mutations):
            index = random.randint(0, self.length - 1)
            mutated.genes[index] = Gen()
        return mutated

    def crossover(self, other):
        """Return a new genome combining genes from self and other."""
        child = self.copy()
        num_crossovers = random.randint(1, 4)
        for _ in range(num_crossovers):
            index = random.randint(0, self.length - 1)
            child.genes[index] = other.genes[index]
        return child
