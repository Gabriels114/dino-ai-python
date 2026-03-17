import random
import numpy as np
import pygame

try:
    import cupy as cp
    _to_numpy = cp.asnumpy
    _GPU = True
except ImportError:
    import numpy as cp   # transparent CPU fallback — same API, no GPU
    _to_numpy = lambda x: x
    _GPU = False

from dino import Dino
from enemy import Cactus, Bird
from game_object import Ground

# ── Population settings ────────────────────────────────────────────────────────
DINOS_PER_GENERATION = 1000

# ── Enemy spawn timing (milliseconds) ─────────────────────────────────────────
MIN_SPAWN_MS = 500
MAX_SPAWN_MS = 1500


class Simulation:
    """Orchestrates the game loop and the genetic algorithm.

    Each generation:
      1. 1 000 dinosaurs run the game simultaneously.
      2. When all are dead, their scores are used to rank them.
      3. The next generation is bred from the survivors.

    Breeding breakdown (1 000 total):
      •  5%  elite       – best dinosaurs kept unchanged
      •  5%  new random  – brand-new genomes (diversity)
      • 30%  mutation    – copies of the best, with random gene changes
      • 40%  mutation    – copies of random top-5%, with random gene changes
      • 20%  crossover   – two parents from top-5% produce a child
    """

    def __init__(self, game_sprites):
        self.game_sprites = game_sprites

        self.dinos   = [Dino() for _ in range(DINOS_PER_GENERATION)]
        self.enemies = []
        self.ground  = Ground()

        self.speed        = 15.0
        self.score        = 0
        self.generation   = 1
        self.dinos_alive  = DINOS_PER_GENERATION

        self.last_gen_avg_score = 0
        self.last_gen_max_score = 0

        self.last_spawn_time = pygame.time.get_ticks()
        self.time_to_spawn   = random.uniform(MIN_SPAWN_MS, MAX_SPAWN_MS)

        # Upload initial weights to GPU (once per generation)
        self._build_gpu_weights()

        # Fonts (initialized here so Simulation owns them)
        self.font_large  = pygame.font.SysFont(None, 36)
        self.font_medium = pygame.font.SysFont(None, 24)
        self.font_small  = pygame.font.SysFont(None, 18)

    # ── Per-frame update ───────────────────────────────────────────────────────

    def update(self):
        """Advance the simulation by one frame."""
        speed = int(self.speed)

        # Phase 1: gather inputs for all living dinosaurs
        alive_dinos   = []
        alive_indices = []
        for i, dino in enumerate(self.dinos):
            if dino.alive:
                dino.prepare_inputs(self._next_obstacle_info(dino), speed)
                alive_dinos.append(dino)
                alive_indices.append(i)

        # Phase 2: single batched forward pass on GPU
        self._gpu_batch_forward(alive_dinos, alive_indices)

        # Phase 3: act on outputs and run physics
        for dino in alive_dinos:
            dino.apply_brain_and_physics()

        # Move enemies and discard those that have scrolled off screen
        for enemy in self.enemies:
            enemy.update(speed)
        self.enemies = [e for e in self.enemies if not e.is_offscreen()]

        # Spawn a new enemy if enough time has passed
        now = pygame.time.get_ticks()
        if now - self.last_spawn_time > self.time_to_spawn:
            self._spawn_enemy()
            self.last_spawn_time = now
            self.time_to_spawn   = random.uniform(MIN_SPAWN_MS, MAX_SPAWN_MS)

        # Phase 4: collision detection — reuses alive_dinos, no extra scan needed
        self.dinos_alive = 0
        for dino in alive_dinos:
            for enemy in self.enemies:
                if dino.alive and dino.is_colliding_with(enemy):
                    dino.die(self.score)
            if dino.alive:
                self.dinos_alive += 1

        if self.dinos_alive == 0:
            self._next_generation()

        self.ground.update(speed)
        self.speed += 0.001   # gradually increase difficulty

    # ── Drawing ────────────────────────────────────────────────────────────────

    def draw(self, screen):
        """Draw all game elements and the HUD."""
        self.ground.draw(screen, self.game_sprites)

        for enemy in self.enemies:
            enemy.draw(screen, self.game_sprites)

        for dino in self.dinos:
            if dino.alive:
                dino.draw(screen, self.game_sprites)

        self._draw_hud(screen)

    # ── Animation ticks (called by main loop on fixed intervals) ──────────────

    def tenth_of_second(self):
        """Advance dino walking animation and increment score (every 0.1 s)."""
        for dino in self.dinos:
            if dino.alive:
                dino.toggle_sprite()
        self.score += 1

    def quarter_of_second(self):
        """Advance enemy animation (every 0.25 s)."""
        for enemy in self.enemies:
            enemy.toggle_sprite()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _next_generation(self):
        """Apply the genetic algorithm to produce a new population."""
        self.score  = 0
        self.speed  = 15.0
        self.generation += 1
        self.enemies.clear()

        # ── Statistics ──────────────────────────────────────────────
        total_score = sum(d.score for d in self.dinos)
        self.last_gen_avg_score = total_score // DINOS_PER_GENERATION

        self.dinos.sort(key=lambda d: d.score, reverse=True)
        self.last_gen_max_score = self.dinos[0].score

        elite_size = int(DINOS_PER_GENERATION * 0.05)   # top 5%
        new_dinos  = []

        # 5% — Elitism: keep the best dinosaurs exactly as they are
        for i in range(elite_size):
            self.dinos[i].reset()
            new_dinos.append(self.dinos[i])

        # 5% — New random dinosaurs (inject genetic diversity)
        for _ in range(elite_size):
            new_dinos.append(Dino())

        # 30% — Mutate the single best dinosaur
        for _ in range(int(DINOS_PER_GENERATION * 0.30)):
            child = Dino()
            child.genome = self.dinos[0].genome.mutate()
            child.init_brain()
            new_dinos.append(child)

        # 40% — Mutate a random parent chosen from the elite pool
        for _ in range(int(DINOS_PER_GENERATION * 0.40)):
            father = self.dinos[random.randint(0, elite_size - 1)]
            child  = Dino()
            child.genome = father.genome.mutate()
            child.init_brain()
            new_dinos.append(child)

        # 20% — Sexual reproduction: crossover between two elite parents
        for _ in range(int(DINOS_PER_GENERATION * 0.20)):
            father = self.dinos[random.randint(0, elite_size - 1)]
            mother = self.dinos[random.randint(0, elite_size - 1)]
            child  = Dino()
            child.genome = father.genome.crossover(mother.genome)
            child.init_brain()
            new_dinos.append(child)

        self.dinos = new_dinos
        self._build_gpu_weights()   # weights changed — refresh GPU arrays

    def _build_gpu_weights(self):
        """Upload all brain weight matrices to GPU (called once per generation)."""
        n  = len(self.dinos)
        hw = np.zeros((n, 7, 7))
        ow = np.zeros((n, 2, 7))
        hb = np.zeros((n, 7))
        ob = np.zeros((n, 2))
        for i, dino in enumerate(self.dinos):
            hw[i] = dino.brain.hidden_layer_weights
            ow[i] = dino.brain.output_layer_weights
            hb[i] = dino.brain.hidden_layer_bias
            ob[i] = dino.brain.output_layer_bias
        self._gpu_hw = cp.asarray(hw)
        self._gpu_ow = cp.asarray(ow)
        self._gpu_hb = cp.asarray(hb)
        self._gpu_ob = cp.asarray(ob)

    def _gpu_batch_forward(self, alive_dinos, alive_indices):
        """Single batched forward pass for all living dinosaurs.

        Runs entirely on GPU (or falls back to batched NumPy when CuPy is
        unavailable). Weights are already resident on the GPU from
        _build_gpu_weights(); only the tiny input vectors are transferred
        each frame.
        """
        if not alive_dinos:
            return

        # Stack inputs: (N, 7)  — tiny transfer to GPU each frame
        inputs_np = np.array([d.brain_inputs for d in alive_dinos])
        gpu_in    = cp.asarray(inputs_np)
        idx       = cp.asarray(alive_indices)

        # Gather weight slices for alive dinos (already on GPU)
        hw = self._gpu_hw[idx]   # (N, 7, 7)
        ow = self._gpu_ow[idx]   # (N, 2, 7)
        hb = self._gpu_hb[idx]   # (N, 7)
        ob = self._gpu_ob[idx]   # (N, 2)

        # Batched forward pass: one GPU kernel covers all dinos
        hidden  = cp.einsum('nij,nj->ni', hw, gpu_in) + hb
        hidden  = cp.maximum(0, hidden)                         # ReLU
        outputs = cp.einsum('nij,nj->ni', ow, hidden) + ob
        outputs = cp.maximum(0, outputs)                        # ReLU

        # Transfer results back to CPU and distribute to each brain
        hidden_np  = _to_numpy(hidden)
        outputs_np = _to_numpy(outputs)
        for i, dino in enumerate(alive_dinos):
            dino.brain.inputs         = inputs_np[i]
            dino.brain.hidden_outputs = hidden_np[i]
            dino.brain.outputs        = outputs_np[i]

    def _next_obstacle_info(self, dino):
        """Return [distance, x, y, width, height] of the nearest obstacle ahead."""
        result = [1280, 0, 0, 0, 0]   # defaults when no obstacle is visible
        for enemy in self.enemies:
            if enemy.x_pos > dino.x_pos:
                result[0] = enemy.x_pos - dino.x_pos
                result[1] = enemy.x_pos
                result[2] = enemy.y_pos
                result[3] = enemy.obj_width
                result[4] = enemy.obj_height
                break   # enemies list is ordered left-to-right; first match is closest
        return result

    def _spawn_enemy(self):
        if random.random() < 0.5:
            self.enemies.append(Cactus())
        else:
            self.enemies.append(Bird())

    def _draw_hud(self, screen):
        """Draw score, generation info, alive count, and neural network."""
        black = (0, 0, 0)
        screen.blit(self.font_large.render(str(self.score), True, black), (1200, 60))
        screen.blit(self.font_large.render(f"Generation: {self.generation}", True, black), (80, 60))
        screen.blit(self.font_large.render(f"Average Score (last gen): {self.last_gen_avg_score}", True, black), (80, 100))
        screen.blit(self.font_large.render(f"Max Score (last gen): {self.last_gen_max_score}", True, black), (80, 140))
        screen.blit(self.font_large.render(f"Alive: {self.dinos_alive}", True, black), (80, 180))

        # Draw the neural network of the first living dinosaur
        for dino in self.dinos:
            if dino.alive:
                dino.brain.draw(screen, self.font_small, self.font_medium)
                break
