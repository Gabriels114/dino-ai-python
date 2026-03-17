import random
import numpy as np

from game_object import GameObject
from genome import Genome
from brain import Brain


class Dino(GameObject):
    """A dinosaur agent controlled by its neural network (Brain).

    Each dinosaur has a unique Genome that encodes its Brain's weights.
    The dinosaur reads the game state, feeds it into the network, and
    translates the network's output into jump or crouch actions.

    After all dinosaurs die, the Simulation runs the genetic algorithm
    on their genomes to produce the next generation.
    """

    GROUND_Y    = 450   # y position when standing on the ground
    CROUCH_Y    = 484   # y position when crouching

    def __init__(self):
        super().__init__()
        self.x_pos      = random.randint(100, 299)
        self.y_pos      = self.GROUND_Y
        self.obj_width  = 80
        self.obj_height = 86

        self.genome = Genome()
        self.brain  = Brain(self.genome)

        self.jump_stage = 0.0   # 0 = grounded; 0 < stage ≤ 1 = mid-jump
        self.alive      = True
        self.score      = 0

        # 7 inputs fed into the neural network each frame
        self.brain_inputs = np.zeros(7)

        self.sprite        = "walking_dino_1"
        self.sprite_offset = [-4, -2]

    # ------------------------------------------------------------------
    # Public interface used by Simulation
    # ------------------------------------------------------------------

    def init_brain(self):
        """Rebuild the Brain from the current Genome (called after mutation)."""
        self.brain = Brain(self.genome)

    def update(self, next_obstacle_info, speed):
        """Run one frame: sense → think → act."""
        self._update_brain_inputs(next_obstacle_info, speed)
        self.brain.feed_forward(self.brain_inputs)
        self._process_brain_output()
        if self.is_jumping():
            self._update_jump()

    def prepare_inputs(self, next_obstacle_info, speed):
        """Phase 1 of GPU-batched update: compute and store network inputs."""
        self._update_brain_inputs(next_obstacle_info, speed)

    def apply_brain_and_physics(self):
        """Phase 2 of GPU-batched update: act on brain outputs and run physics."""
        self._process_brain_output()
        if self.is_jumping():
            self._update_jump()

    def die(self, sim_score):
        self.alive = False
        self.score = sim_score

    def reset(self):
        """Reuse an elite dinosaur in the next generation."""
        self.alive = True
        self.score = 0

    def toggle_sprite(self):
        """Alternate between walking/crouching animation frames."""
        if   self.sprite == "walking_dino_1":   self.sprite = "walking_dino_2"
        elif self.sprite == "walking_dino_2":   self.sprite = "walking_dino_1"
        elif self.sprite == "crouching_dino_1": self.sprite = "crouching_dino_2"
        elif self.sprite == "crouching_dino_2": self.sprite = "crouching_dino_1"

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def is_jumping(self):
        return self.jump_stage > 0

    def is_crouching(self):
        return self.obj_width == 110   # width changes when crouching

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------

    def _update_brain_inputs(self, info, speed):
        """Normalize all game values to [0, 1] before feeding the network.

        Raw pixel values and speeds are divided into meaningful ranges so
        the network always receives inputs on a consistent scale.
        """
        self.brain_inputs[0] = info[0] / 900                       # distance to obstacle
        self.brain_inputs[1] = (info[1] - 450)  / (1350 - 450)    # obstacle x position
        self.brain_inputs[2] = (info[2] - 370)  / (480  - 370)    # obstacle y position
        self.brain_inputs[3] = (info[3] - 30)   / (146  - 30)     # obstacle width
        self.brain_inputs[4] = (info[4] - 40)   / (96   - 40)     # obstacle height
        self.brain_inputs[5] = (self.y_pos - 278) / (484 - 278)   # dino y position
        self.brain_inputs[6] = (speed - 15) / (30 - 15)           # game speed

    def _update_jump(self):
        """Parabolic jump arc using a parametric equation.

        y(t) = ground - height * (-4t)(t - 1)   for t in (0, 1]
        The factor -4t(t-1) is a parabola that peaks at t=0.5.
        """
        self.y_pos       = self.GROUND_Y - ((-4 * self.jump_stage * (self.jump_stage - 1)) * 172)
        self.jump_stage += 0.03
        if self.jump_stage > 1:
            self._stop_jump()

    def _process_brain_output(self):
        """Map network outputs to game actions.

        outputs[0] != 0  →  jump (if not already crouching or jumping)
        outputs[1] != 0  →  crouch (cancels jump if mid-air)
        """
        if self.brain.outputs[0] != 0:
            if not self.is_crouching() and not self.is_jumping():
                self._jump()

        if self.brain.outputs[1] == 0:
            if self.is_crouching():
                self._stop_crouch()
        else:
            if self.is_jumping():
                self._stop_jump()
            self._crouch()

    def _jump(self):
        self.jump_stage = 0.0001
        self.sprite     = "standing_dino"

    def _stop_jump(self):
        self.jump_stage = 0
        self.y_pos      = self.GROUND_Y
        self.sprite     = "walking_dino_1"

    def _crouch(self):
        if not self.is_crouching():
            self.y_pos      = self.CROUCH_Y
            self.obj_width  = 110
            self.obj_height = 52
            self.sprite     = "crouching_dino_1"

    def _stop_crouch(self):
        self.y_pos      = self.GROUND_Y
        self.obj_width  = 80
        self.obj_height = 86
        self.sprite     = "walking_dino_1"
