class GameObject:
    """Base class for all objects in the game (dinosaurs, enemies, ground).

    Every object has a position (x, y), a size (width, height),
    a sprite name, and an optional drawing offset.
    """

    def __init__(self):
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.obj_width = 0.0
        self.obj_height = 0.0
        self.sprite = None
        self.sprite_offset = [0, 0]

    def draw(self, screen, game_sprites):
        """Draw the object's sprite at its current position."""
        img = game_sprites[self.sprite]
        screen.blit(img, (self.x_pos + self.sprite_offset[0],
                          self.y_pos + self.sprite_offset[1]))

    def is_colliding_with(self, other):
        """AABB (axis-aligned bounding box) collision detection."""
        return (
            self.x_pos + self.obj_width  > other.x_pos and
            self.x_pos                   < other.x_pos + other.obj_width and
            self.y_pos + self.obj_height > other.y_pos and
            self.y_pos                   < other.y_pos + other.obj_height
        )

    def toggle_sprite(self):
        """Override in subclasses to animate the sprite."""
        pass


class Ground(GameObject):
    """The scrolling ground strip.

    Two copies of the ground image are drawn side by side so the
    strip appears infinite as it scrolls to the left.
    """

    def __init__(self):
        super().__init__()
        self.x_pos = 2400
        self.y_pos = 515
        self.sprite = "ground"

    def update(self, speed):
        self.x_pos -= speed
        if self.x_pos <= 0:
            self.x_pos = 2400  # loop back to the right

    def draw(self, screen, game_sprites):
        ground_img = game_sprites["ground"]
        screen.blit(ground_img, (self.x_pos, self.y_pos))
        screen.blit(ground_img, (self.x_pos - 2400, self.y_pos))  # second tile
