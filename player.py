import pygame

from maze import DEFAULT_MAZE, TILE_SIZE, find_cell, is_walkable


class Player:
    """Grid-based player used by the manual version of the maze."""

    COLOR = (52, 152, 219)

    def __init__(self, maze=DEFAULT_MAZE):
        self.maze = maze
        self.start_position = find_cell("P", maze)
        self.reset()

    @property
    def position(self):
        return self.row, self.col

    def reset(self):
        self.row, self.col = self.start_position
        self.moves = 0

    def move(self, row_change, col_change):
        """Attempt a move and return True only when the player changed cell."""
        next_row = self.row + row_change
        next_col = self.col + col_change

        if not is_walkable(next_row, next_col, self.maze):
            return False

        self.row = next_row
        self.col = next_col
        self.moves += 1
        return True

    def draw(self, screen):
        center = (
            self.col * TILE_SIZE + TILE_SIZE // 2,
            self.row * TILE_SIZE + TILE_SIZE // 2,
        )
        pygame.draw.circle(screen, self.COLOR, center, TILE_SIZE // 3)
        pygame.draw.circle(screen, (33, 97, 140), center, TILE_SIZE // 3, 3)
