import random
import copy
import json
import os


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
        """Return a mutated copy: replace 1–4 random genes and occasionally a bias."""
        mutated = self.copy()
        num_mutations = random.randint(1, 4)
        for _ in range(num_mutations):
            index = random.randint(0, self.length - 1)
            mutated.genes[index] = Gen()
        # 30% chance to also mutate one bias value
        if random.random() < 0.3:
            if random.random() < 0.5:
                i = random.randint(0, 6)
                mutated.hidden_layer_bias[i] = random.uniform(-1, 1)
            else:
                i = random.randint(0, 1)
                mutated.output_layer_bias[i] = random.uniform(-1, 1)
        return mutated

    def save(self, filepath, score=0, generation=0):
        """Serialize this genome to a JSON file."""
        data = {
            "score":              score,
            "generation":         generation,
            "hidden_layer_bias":  self.hidden_layer_bias,
            "output_layer_bias":  self.output_layer_bias,
            "genes": [
                {
                    "source_hidden_layer": g.source_hidden_layer,
                    "id_source_neuron":    g.id_source_neuron,
                    "id_target_neuron":    g.id_target_neuron,
                    "weight":              g.weight,
                }
                for g in self.genes
            ],
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath):
        """Deserialize a genome from a JSON file.

        Returns (genome, score, generation) or raises FileNotFoundError.
        """
        with open(filepath) as f:
            data = json.load(f)

        genome = cls.__new__(cls)
        genome.length = len(data["genes"])
        genome.hidden_layer_bias = data["hidden_layer_bias"]
        genome.output_layer_bias = data["output_layer_bias"]

        genome.genes = []
        for gd in data["genes"]:
            g = Gen.__new__(Gen)
            g.source_hidden_layer = gd["source_hidden_layer"]
            g.id_source_neuron    = gd["id_source_neuron"]
            g.id_target_neuron    = gd["id_target_neuron"]
            g.weight              = gd["weight"]
            genome.genes.append(g)

        return genome, data.get("score", 0), data.get("generation", 0)

    def crossover(self, other):
        """Return a new genome combining genes and biases from self and other."""
        child = self.copy()
        num_crossovers = random.randint(1, 4)
        for _ in range(num_crossovers):
            index = random.randint(0, self.length - 1)
            child.genes[index] = other.genes[index]
        # Cross over each bias independently with 50% probability
        child.hidden_layer_bias = [
            other.hidden_layer_bias[i] if random.random() < 0.5 else child.hidden_layer_bias[i]
            for i in range(len(child.hidden_layer_bias))
        ]
        child.output_layer_bias = [
            other.output_layer_bias[i] if random.random() < 0.5 else child.output_layer_bias[i]
            for i in range(len(child.output_layer_bias))
        ]
        return child
