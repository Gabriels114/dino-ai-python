import numpy as np
import pygame


class Brain:
    """A 2-layer neural network that decides what the dinosaur does.

    Architecture:
      Input layer  →  7 neurons  (normalized game state)
      Hidden layer →  7 neurons  (ReLU activation)
      Output layer →  2 neurons  (ReLU activation)
                       output[0] = jump signal
                       output[1] = crouch signal

    Weights are NOT learned via backpropagation.
    Instead, they evolve across generations through the genetic algorithm.
    """

    def __init__(self, genome):
        # Build weight matrices from the genome's genes
        # hidden_layer_weights[target][source] = weight
        self.hidden_layer_weights = np.zeros((7, 7))
        self.output_layer_weights = np.zeros((2, 7))

        for gen in genome.genes:
            if gen.source_hidden_layer:
                self.hidden_layer_weights[gen.id_target_neuron][gen.id_source_neuron] = gen.weight
            else:
                self.output_layer_weights[gen.id_target_neuron][gen.id_source_neuron] = gen.weight

        self.hidden_layer_bias = np.array(genome.hidden_layer_bias)
        self.output_layer_bias = np.array(genome.output_layer_bias)

        # Current activation values — used both for decisions and visualization
        self.inputs = np.zeros(7)
        self.hidden_outputs = np.zeros(7)
        self.outputs = np.array([1.0, 0.0])  # default: jump

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def feed_forward(self, input_values):
        """Run the network: input → hidden (ReLU) → output (ReLU).

        Steps:
          1. Multiply inputs by hidden-layer weight matrix
          2. Add hidden-layer bias
          3. Apply ReLU (set negatives to 0)
          4. Repeat for the output layer
        """
        self.inputs = input_values

        # Hidden layer
        self.hidden_outputs = np.dot(self.hidden_layer_weights, input_values)
        self.hidden_outputs += self.hidden_layer_bias
        self.hidden_outputs = np.maximum(0, self.hidden_outputs)  # ReLU

        # Output layer
        self.outputs = np.dot(self.output_layer_weights, self.hidden_outputs)
        self.outputs += self.output_layer_bias
        self.outputs = np.maximum(0, self.outputs)  # ReLU

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def _connection_style(self, weight):
        """Return (color, line_width) for drawing a connection.

        Green  = positive weight  (excitatory connection)
        Red    = negative weight  (inhibitory connection)
        Gray   = zero weight      (inactive connection)
        Width  = magnitude of the weight
        """
        if weight > 0:
            color = (0, 200, 0)
        elif weight < 0:
            color = (200, 0, 0)
        else:
            color = (200, 200, 200)

        # Map abs(weight) from [0, 1] → line width [1, 5]
        width = max(1, int(0.5 + abs(weight) * 4.5))
        return color, width

    def draw(self, screen, font_small, font_medium):
        """Draw the neural network diagram on screen.

        Layout (x positions):
          700 = input layer circles
          800 = hidden layer circles
          900 = output layer circles
          y starts at 64 with 40px spacing between neurons
        """
        # ---- Input labels ----
        input_labels = [
            ("(obstacle) distance", 550),
            ("(obstacle) x",        598),
            ("(obstacle) y",        598),
            ("(obstacle) width",    568),
            ("(obstacle) height",   563),
            ("(dino) y",            625),
            ("(game) speed",        586),
        ]
        for i, (label, x) in enumerate(input_labels):
            surf = font_small.render(label, True, (0, 0, 0))
            screen.blit(surf, (x, 59 + i * 40))

        # Output labels
        screen.blit(font_small.render("jump",   True, (0, 0, 0)), (927, 160))
        screen.blit(font_small.render("crouch", True, (0, 0, 0)), (925, 200))

        # ---- Connection lines (drawn behind circles) ----
        for i in range(7):
            # Input → Hidden
            for j in range(7):
                weight = self.hidden_layer_weights[i][j]
                color, width = self._connection_style(weight)
                pygame.draw.line(screen, color,
                                 (716, 64 + i * 40),
                                 (784, 64 + j * 40), width)
            # Hidden → Output
            for j in range(2):
                weight = self.output_layer_weights[j][i]
                color, width = self._connection_style(weight)
                pygame.draw.line(screen, color,
                                 (816, 64 + i * 40),
                                 (884, 165 + j * 40), width)

        # ---- Neuron circles (drawn on top of lines) ----
        for i in range(7):
            # Input neuron: always white
            pygame.draw.circle(screen, (255, 255, 255), (700, 64 + i * 40), 16)
            pygame.draw.circle(screen, (83, 83, 83),    (700, 64 + i * 40), 16, 1)

            # Input value inside the circle
            val = font_small.render(f"{self.inputs[i]:.2f}", True, (0, 0, 0))
            screen.blit(val, (684, 59 + i * 40))

            # Hidden neuron: gray if active (output > 0), white if inactive
            fill = (170, 170, 170) if self.hidden_outputs[i] != 0 else (255, 255, 255)
            pygame.draw.circle(screen, fill,       (800, 64 + i * 40), 16)
            pygame.draw.circle(screen, (0, 0, 0),  (800, 64 + i * 40), 16, 1)

        # Output neurons
        for j, y in enumerate([165, 205]):
            fill = (170, 170, 170) if self.outputs[j] != 0 else (255, 255, 255)
            pygame.draw.circle(screen, fill,      (900, y), 16)
            pygame.draw.circle(screen, (0, 0, 0), (900, y), 16, 1)
