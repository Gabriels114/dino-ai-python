"""Dino AI — Genetic Algorithm + Neural Network
================================================
Entry point. Run with:

    python main.py

Requirements:
    pip install pygame numpy
"""

import os
import sys
import pygame

from simulation import Simulation

WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 720
FPS           = 60


def initialize_sprites(sprite_sheet):
    """Extract individual sprites from the sprite sheet PNG.

    The original sprite sheet contains all images packed into a single file.
    We crop each one by its pixel coordinates (x, y, width, height).
    """
    def crop(x, y, w, h):
        surface = pygame.Surface((w, h), pygame.SRCALPHA)
        surface.blit(sprite_sheet, (0, 0), (x, y, w, h))
        return surface

    return {
        # Dinosaur states
        "standing_dino":    crop(1338,  2,  88,  94),
        "walking_dino_1":   crop(1514,  2,  88,  94),
        "walking_dino_2":   crop(1602,  2,  88,  94),
        "dead_dino":        crop(1690,  2,  88,  94),
        "crouching_dino_1": crop(1866, 36, 118,  60),
        "crouching_dino_2": crop(1984, 36, 118,  60),

        # Cactus types (small → large)
        "cactus_type_1":    crop( 446,  2,  34,  70),
        "cactus_type_2":    crop( 480,  2,  68,  70),
        "cactus_type_3":    crop( 548,  2, 102,  70),
        "cactus_type_4":    crop( 652,  2,  50, 100),
        "cactus_type_5":    crop( 702,  2, 100, 100),
        "cactus_type_6":    crop( 802,  2, 150, 100),

        # Bird animation frames
        "bird_flying_1":    crop( 260,  2,  92,  80),
        "bird_flying_2":    crop( 352,  2,  92,  80),

        # Ground strip (2400 px wide, tiled for infinite scroll)
        "ground":           crop(   2, 104, 2400, 24),
    }


def main():
    pygame.init()

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Dino AI — Genetic Algorithm")

    clock = pygame.time.Clock()

    # Load sprite sheet from assets/
    assets_dir   = os.path.join(os.path.dirname(__file__), "assets")
    sprite_sheet = pygame.image.load(os.path.join(assets_dir, "sprites.png")).convert_alpha()
    game_sprites = initialize_sprites(sprite_sheet)

    simulation = Simulation(game_sprites)

    # Animation clock: incremented every 50 ms to drive sprite flipping
    animation_clock     = 0
    last_animation_tick = pygame.time.get_ticks()

    while True:
        # ── Events ──────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # ── Game logic ───────────────────────────────────────────────
        simulation.update()

        # ── Rendering ────────────────────────────────────────────────
        screen.fill((247, 247, 247))   # light-gray background
        simulation.draw(screen)

        # ── Animation timing ─────────────────────────────────────────
        now = pygame.time.get_ticks()
        if now - last_animation_tick > 50:      # every 50 ms
            last_animation_tick = now
            animation_clock += 1

            if animation_clock % 2 == 0:        # every 0.1 s
                simulation.tenth_of_second()

            if animation_clock % 5 == 0:        # every 0.25 s
                simulation.quarter_of_second()

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
