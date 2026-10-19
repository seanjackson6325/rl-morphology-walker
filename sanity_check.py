"""Visual sanity check for the creature rig: build one fixed-morphology biped,
drop it on the ground with all motors idle, and watch what happens.

No RL involved yet -- this is purely "did I wire the joints/motors/collision
filters correctly", before any learning enters the picture. A correct rig
should settle into a standing (or gently toppling) pose under gravity, not
explode, clip through the floor, or fly off to infinity.
"""

from __future__ import annotations

import sys

import pygame
import pymunk

from walker.creature import GROUND_COLLISION_TYPE, GROUND_FRICTION, build_creature
from walker.morphology import fixed_morphology

PPM = 300.0  # pixels per meter
SCREEN_W, SCREEN_H = 1200, 600
GROUND_SCREEN_Y = SCREEN_H - 80
SIM_SECONDS = 6.0
PHYSICS_DT = 1.0 / 240.0
SUBSTEPS_PER_FRAME = 4  # render at 60 Hz, physics solved at 240 Hz for stability


def to_screen(point: tuple[float, float], camera_x: float) -> tuple[int, int]:
    sx = SCREEN_W / 2 + (point[0] - camera_x) * PPM
    sy = GROUND_SCREEN_Y - point[1] * PPM
    return int(sx), int(sy)


def main() -> None:
    space = pymunk.Space()
    space.gravity = (0.0, -9.81)

    ground = pymunk.Segment(space.static_body, (-200, 0), (200, 0), 0.05)
    ground.friction = GROUND_FRICTION
    ground.collision_type = GROUND_COLLISION_TYPE
    space.add(ground)

    creature = build_creature(space, fixed_morphology(), origin_x=0.0, ground_y=0.0)
    print(f"standing_height={creature.standing_height:.3f} m")

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Creature rig sanity check (no RL yet)")
    clock = pygame.time.Clock()

    total_frames = int(SIM_SECONDS * 60)
    for frame in range(total_frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        for _ in range(SUBSTEPS_PER_FRAME):
            space.step(PHYSICS_DT)

        torso = creature.torso
        if frame % 30 == 0:
            print(
                f"t={frame / 60:.2f}s  torso_y={torso.position.y:.3f}  "
                f"torso_angle={torso.angle:+.3f}  vx={torso.velocity.x:+.3f}"
            )
            if not (torso.position.y == torso.position.y) or abs(torso.position.y) > 1000:
                print("ERROR: simulation exploded (NaN or huge position)")
                sys.exit(1)

        camera_x = torso.position.x
        screen.fill((30, 30, 40))
        pygame.draw.line(
            screen, (90, 70, 50), (0, GROUND_SCREEN_Y), (SCREEN_W, GROUND_SCREEN_Y), 4
        )

        for shape in creature.shapes:
            verts = [shape.body.local_to_world(v) for v in shape.get_vertices()]
            points = [to_screen((v.x, v.y), camera_x) for v in verts]
            pygame.draw.polygon(screen, (120, 180, 220), points)
            pygame.draw.polygon(screen, (20, 20, 20), points, 2)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("Sanity check finished without exploding.")


if __name__ == "__main__":
    main()
