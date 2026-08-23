import pygame

from maze import GOAL_POSITION, MAZE, TILE_SIZE, draw_maze
from player import Player

pygame.init()

WIDTH = len(MAZE[0]) * TILE_SIZE
MAZE_HEIGHT = len(MAZE) * TILE_SIZE
HUD_HEIGHT = 70
HEIGHT = MAZE_HEIGHT + HUD_HEIGHT

screen = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Q-Learning Maze")

clock = pygame.time.Clock()
font = pygame.font.Font(None, 30)
small_font = pygame.font.Font(None, 23)
player = Player()
won = False

KEY_MOVES = {
    pygame.K_UP: (-1, 0),
    pygame.K_DOWN: (1, 0),
    pygame.K_LEFT: (0, -1),
    pygame.K_RIGHT: (0, 1),
}

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                player.reset()
                won = False
            elif event.key in KEY_MOVES and not won:
                player.move(*KEY_MOVES[event.key])
                won = player.position == GOAL_POSITION

    screen.fill((24, 29, 38))

    draw_maze(screen)
    player.draw(screen)

    status = f"Moves: {player.moves}"
    if won:
        status += "   Goal reached!"
    status_surface = font.render(status, True, (245, 247, 250))
    help_surface = small_font.render(
        "Arrow keys: move     R: reset", True, (174, 183, 196)
    )
    screen.blit(status_surface, (16, MAZE_HEIGHT + 10))
    screen.blit(help_surface, (16, MAZE_HEIGHT + 40))

    pygame.display.flip()

    clock.tick(60)

pygame.quit()
