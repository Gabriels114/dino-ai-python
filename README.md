# Dino AI — Genetic Algorithm + Neural Network

A simulation where 1000 dinosaurs learn to play the Chrome Dino game through a genetic algorithm and a hand-coded neural network.

Watch the original explanation video (Spanish, 1M+ views): [YouTube](https://youtu.be/gC85en0Vmh4)

## How it works

Each dinosaur is controlled by a small neural network:

```
Inputs (7)       Hidden (7, ReLU)    Output (2, ReLU)
─────────────    ────────────────    ────────────────
distance         neuron 0            jump
obstacle x       neuron 1            crouch
obstacle y       neuron 2
obstacle width   neuron 3
obstacle height  neuron 4
dino y           neuron 5
game speed       neuron 6
```

Weights are not trained via backpropagation — they evolve across generations through a genetic algorithm.

Each generation (1000 dinos):

| Strategy        | % of population | Description                          |
|-----------------|-----------------|--------------------------------------|
| Elitism         | 5%              | Best dinos kept unchanged            |
| Random          | 5%              | Fresh random genomes (diversity)     |
| Mutation (best) | 30%             | Best dino mutated                    |
| Mutation (pool) | 40%             | Random top-5% dino mutated           |
| Crossover       | 20%             | Two elite parents produce a child    |

## Setup

```bash
pip install -r requirements.txt
python main.py
```

### GPU acceleration (NVIDIA)

The forward pass for all 1000 dinos is batched into a single GPU operation using CuPy. Install the version matching your CUDA:

```bash
# Check your CUDA version first
nvidia-smi

# CUDA 12.x
pip install cupy-cuda12x

# CUDA 11.x
pip install cupy-cuda11x
```

If CuPy is not installed the simulation runs automatically on CPU with NumPy.

## Project structure

```
├── main.py          # Entry point, game loop
├── simulation.py    # Genetic algorithm + game orchestration
├── brain.py         # Neural network (forward pass + visualization)
├── genome.py        # Genome encoding (genes, mutation, crossover)
├── dino.py          # Dinosaur agent (physics + input normalization)
├── enemy.py         # Cactus and Bird obstacles
├── game_object.py   # Base class + Ground
├── assets/
│   └── sprites.png  # Sprite sheet
└── requirements.txt
```
