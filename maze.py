import pygame

TILE_SIZE = 50

WALL_COLOR = (44, 52, 64)
FLOOR_COLOR = (238, 241, 245)
GRID_COLOR = (202, 208, 218)
GOAL_COLOR = (46, 204, 113)

MAZE = [
    "##########",
    "#P       #",
    "# ### ## #",
    "#   #    #",
    "# # #### #",
    "#      #G#",
    "##########",
]


def find_cell(symbol):
    """Return the (row, column) containing a maze symbol."""
    for row_index, row in enumerate(MAZE):
        for col_index, cell in enumerate(row):
            if cell == symbol:
                return row_index, col_index
    raise ValueError(f"Maze does not contain {symbol!r}")


START_POSITION = find_cell("P")
GOAL_POSITION = find_cell("G")


def is_walkable(row, col):
    """Check whether a grid position is inside the maze and not a wall."""
    return (
        0 <= row < len(MAZE)
        and 0 <= col < len(MAZE[0])
        and MAZE[row][col] != "#"
    )


def draw_maze(screen):
    for row_index, row in enumerate(MAZE):
        for col_index, cell in enumerate(row):
            x = col_index * TILE_SIZE
            y = row_index * TILE_SIZE

            rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)

            color = WALL_COLOR if cell == "#" else FLOOR_COLOR
            pygame.draw.rect(screen, color, rect)

            if cell == "G":
                center = rect.center
                pygame.draw.circle(screen, GOAL_COLOR, center, TILE_SIZE // 3)
                pygame.draw.circle(screen, (30, 132, 73), center, TILE_SIZE // 3, 3)

            pygame.draw.rect(screen, GRID_COLOR, rect, 1)
