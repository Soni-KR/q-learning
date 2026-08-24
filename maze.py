from dataclasses import dataclass

import pygame

TILE_SIZE = 40

WALL_COLOR = (28, 57, 86)
WALL_ACCENT = (38, 82, 122)
FLOOR_COLOR = (229, 238, 242)
GRID_COLOR = (190, 205, 214)
GOAL_COLOR = (255, 202, 40)


@dataclass(frozen=True)
class MazeDefinition:
    key: str
    name: str
    grid: tuple[str, ...]
    training_episodes_per_move: int

    def __post_init__(self):
        widths = {len(row) for row in self.grid}
        if len(widths) != 1:
            raise ValueError(f"Maze {self.name!r} must be rectangular")
        if sum(row.count("P") for row in self.grid) != 1:
            raise ValueError(f"Maze {self.name!r} needs exactly one P")
        if sum(row.count("G") for row in self.grid) != 1:
            raise ValueError(f"Maze {self.name!r} needs exactly one G")

    @property
    def rows(self):
        return len(self.grid)

    @property
    def cols(self):
        return len(self.grid[0])


MAZES = (
    MazeDefinition(
        "green_hill", "GREEN HILL SPRINT",
        (
            "##########",
            "#P       #",
            "# ### ## #",
            "#   #    #",
            "# # #### #",
            "#      #G#",
            "##########",
        ),
        12,
    ),
    MazeDefinition(
        "chemical_dash", "CHEMICAL DASH",
        (
            "###############",
            "#P# #         #",
            "# # # # # #####",
            "# #   # # #   #",
            "# ##### ### # #",
            "# #   # #   # #",
            "# # # # # ### #",
            "#   # #     # #",
            "##### ####### #",
            "#            G#",
            "###############",
        ),
        20,
    ),
    MazeDefinition(
        "sky_chase", "SKY CHASE CIRCUIT",
        (
            "###################",
            "#P    #     #     #",
            "##### # ### # ### #",
            "# #   # #   # #   #",
            "# # ### # ### # ###",
            "# #   # # #   #   #",
            "# ### # # ####### #",
            "#   # # #       # #",
            "# # # # ####### # #",
            "# # # #   #   # # #",
            "# ### ### # # # # #",
            "#         # #    G#",
            "###################",
        ),
        30,
    ),
)

DEFAULT_MAZE = MAZES[0]
# Backward-compatible aliases used by the earlier learning phases.
MAZE = DEFAULT_MAZE.grid


def find_cell(symbol, maze=DEFAULT_MAZE):
    for row_index, row in enumerate(maze.grid):
        for col_index, cell in enumerate(row):
            if cell == symbol:
                return row_index, col_index
    raise ValueError(f"Maze does not contain {symbol!r}")


START_POSITION = find_cell("P")
GOAL_POSITION = find_cell("G")


def is_walkable(row, col, maze=DEFAULT_MAZE):
    return (
        0 <= row < maze.rows
        and 0 <= col < maze.cols
        and maze.grid[row][col] != "#"
    )


def draw_maze(screen, maze=DEFAULT_MAZE, offset=(0, 0)):
    offset_x, offset_y = offset
    for row_index, row in enumerate(maze.grid):
        for col_index, cell in enumerate(row):
            x = offset_x + col_index * TILE_SIZE
            y = offset_y + row_index * TILE_SIZE
            rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
            color = WALL_COLOR if cell == "#" else FLOOR_COLOR
            pygame.draw.rect(screen, color, rect)

            if cell == "#":
                pygame.draw.line(screen, WALL_ACCENT, rect.topleft, rect.bottomright, 2)
            elif (row_index + col_index) % 2 == 0:
                pygame.draw.rect(screen, (221, 232, 237), rect)

            if cell == "G":
                center = rect.center
                pygame.draw.circle(screen, GOAL_COLOR, center, TILE_SIZE // 3, 5)
                pygame.draw.circle(screen, (255, 237, 142), center, TILE_SIZE // 4, 2)

            pygame.draw.rect(screen, GRID_COLOR, rect, 1)
