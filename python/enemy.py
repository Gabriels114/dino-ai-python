import random

from game_object import GameObject


class Enemy(GameObject):
    """Base class for all obstacles.

    Enemies always spawn off-screen to the right (x=1350) and
    move left at the current game speed each frame.
    """

    def __init__(self):
        super().__init__()
        self.x_pos = 1350

    def update(self, speed):
        self.x_pos -= speed

    def is_offscreen(self):
        """True when the enemy has scrolled completely past the left edge."""
        return self.x_pos + self.obj_width < 0


class Cactus(Enemy):
    """A cactus obstacle. Six variations with different sizes and y positions."""

    # Each index corresponds to one cactus type (0–5)
    WIDTHS  = [30,  64,  98,  46,  96,  146]
    HEIGHTS = [66,  66,  66,  96,  96,   96]
    Y_POS   = [470, 470, 470, 444, 444,  444]

    def __init__(self):
        super().__init__()
        t = random.randint(0, 5)          # pick a random cactus type
        self.obj_width      = self.WIDTHS[t]
        self.obj_height     = self.HEIGHTS[t]
        self.y_pos          = self.Y_POS[t]
        self.sprite         = f"cactus_type_{t + 1}"
        self.sprite_offset  = [-2, -2]


class Bird(Enemy):
    """A flying bird obstacle. Three height variations force the dino to
    either jump over, duck under, or jump and duck."""

    Y_POSITIONS = [435, 480, 370]

    def __init__(self):
        super().__init__()
        self.obj_width     = 84
        self.obj_height    = 40
        t                  = random.randint(0, 2)
        self.y_pos         = self.Y_POSITIONS[t]
        self.sprite        = "bird_flying_1"
        self.sprite_offset = [-4, -16]

    def toggle_sprite(self):
        """Animate the bird's wing flap."""
        if self.sprite == "bird_flying_1":
            self.sprite = "bird_flying_2"
        else:
            self.sprite = "bird_flying_1"
