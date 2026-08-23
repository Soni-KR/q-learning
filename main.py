from pathlib import Path

import pygame

from agent import QLearningAgent
from environment import MazeEnvironment
from maze import GOAL_POSITION, MAZE, TILE_SIZE, draw_maze
from player import Player
from training import train_agent
from visualization import save_training_plot

pygame.init()

WIDTH = len(MAZE[0]) * TILE_SIZE
MAZE_HEIGHT = len(MAZE) * TILE_SIZE
HUD_HEIGHT = 100
HEIGHT = MAZE_HEIGHT + HUD_HEIGHT

BACKGROUND = (24, 29, 38)
TEXT = (245, 247, 250)
MUTED = (174, 183, 196)
BUTTON = (52, 73, 94)
BUTTON_HOVER = (65, 95, 122)
ACCENT = (52, 152, 219)
AGENT_COLOR = (231, 76, 60)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q-Learning Maze")
clock = pygame.time.Clock()
title_font = pygame.font.Font(None, 45)
font = pygame.font.Font(None, 28)
small_font = pygame.font.Font(None, 21)

player = Player()
environment = MazeEnvironment()
trained_agent = None
training_history = None
graph_surface = None
mode = "menu"
won = False
status_message = "Choose how you want to explore the maze."
agent_position = environment.state
agent_reward = 0
last_agent_move = 0
WATCH_DELAY_MS = 280
MIN_WATCH_DELAY = 60
MAX_WATCH_DELAY = 800
MODEL_PATH = Path(__file__).with_name("q_table.npz")
agent_path = [agent_position]

KEY_MOVES = {
    pygame.K_UP: (-1, 0),
    pygame.K_DOWN: (1, 0),
    pygame.K_LEFT: (0, -1),
    pygame.K_RIGHT: (0, 1),
}

BUTTONS = {
    "play": pygame.Rect(110, 150, 280, 52),
    "train": pygame.Rect(110, 217, 280, 52),
    "watch": pygame.Rect(110, 284, 280, 52),
    "graph": pygame.Rect(110, 351, 280, 52),
}


def draw_centered(text, used_font, color, center_y):
    surface = used_font.render(text, True, color)
    screen.blit(surface, surface.get_rect(center=(WIDTH // 2, center_y)))


def draw_button(rect, label, enabled=True):
    hovering = rect.collidepoint(pygame.mouse.get_pos()) and enabled
    color = BUTTON_HOVER if hovering else BUTTON
    if not enabled:
        color = (43, 49, 59)
    pygame.draw.rect(screen, color, rect, border_radius=8)
    pygame.draw.rect(screen, ACCENT if enabled else (75, 82, 94), rect, 2, 8)
    surface = font.render(label, True, TEXT if enabled else MUTED)
    screen.blit(surface, surface.get_rect(center=rect.center))


def draw_agent(position):
    row, col = position
    center = (col * TILE_SIZE + TILE_SIZE // 2, row * TILE_SIZE + TILE_SIZE // 2)
    pygame.draw.circle(screen, AGENT_COLOR, center, TILE_SIZE // 3)
    pygame.draw.circle(screen, (146, 43, 33), center, TILE_SIZE // 3, 3)


def draw_agent_path():
    if len(agent_path) < 2:
        return
    points = [
        (col * TILE_SIZE + TILE_SIZE // 2, row * TILE_SIZE + TILE_SIZE // 2)
        for row, col in agent_path
    ]
    pygame.draw.lines(screen, (241, 196, 15), False, points, 5)


def draw_menu():
    draw_centered("Q-Learning Maze", title_font, TEXT, 65)
    draw_centered(status_message, small_font, MUTED, 105)
    draw_button(BUTTONS["play"], "PLAY YOURSELF")
    draw_button(BUTTONS["train"], "TRAIN AGENT")
    draw_button(BUTTONS["watch"], "WATCH AGENT", trained_agent is not None)
    draw_button(BUTTONS["graph"], "LEARNING GRAPHS", training_history is not None)
    draw_centered("Esc returns to this menu", small_font, MUTED, HEIGHT - 20)


def draw_maze_mode():
    draw_maze(screen)
    if mode == "manual":
        player.draw(screen)
        message = f"Manual play   Moves: {player.moves}"
        if won:
            message += "   Goal reached!"
        help_text = "Arrow keys: move     R: reset     Esc: menu"
    else:
        draw_agent_path()
        draw_agent(agent_position)
        speed = 1000 / WATCH_DELAY_MS
        message = (
            f"Agent   Step: {environment.steps}   Reward: {agent_reward}   "
            f"Speed: {speed:.1f}/s"
        )
        if environment.done:
            message += "   Goal reached!"
        help_text = "R: replay     +/-: speed     Esc: menu"

    screen.blit(font.render(message, True, TEXT), (14, MAZE_HEIGHT + 14))
    screen.blit(small_font.render(help_text, True, MUTED), (14, MAZE_HEIGHT + 54))


def start_training():
    global trained_agent, training_history, graph_surface, status_message
    trained_agent, training_history = train_agent(report_every=0)
    trained_agent.save(MODEL_PATH)
    graph_path = Path(__file__).with_name("training_progress.png")
    save_training_plot(training_history, graph_path)
    loaded = pygame.image.load(str(graph_path)).convert()
    graph_surface = pygame.transform.smoothscale(loaded, (WIDTH, HEIGHT - 45))
    final_window = training_history[-100:]
    success_rate = sum(item.reached_goal for item in final_window) / len(final_window)
    status_message = (
        f"Training complete: {success_rate:.0%} recent success, "
        f"epsilon {trained_agent.epsilon:.2f}"
    )


def reset_watch():
    global agent_position, agent_reward, last_agent_move, agent_path
    agent_position = environment.reset()
    agent_reward = 0
    agent_path = [agent_position]
    last_agent_move = pygame.time.get_ticks()


def load_saved_agent():
    global trained_agent, status_message
    if not MODEL_PATH.exists():
        return
    try:
        candidate = QLearningAgent.load(MODEL_PATH, seed=42)
        expected_shape = (environment.rows, environment.cols, len(environment.ACTIONS))
        if candidate.q_table.shape != expected_shape:
            raise ValueError("saved model does not match this maze")
        trained_agent = candidate
        status_message = "Saved trained agent loaded. Train again or watch it escape."
    except (OSError, ValueError, KeyError):
        status_message = "Saved model could not be loaded; train a new agent."


load_saved_agent()


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                mode = "menu"
            elif mode == "manual":
                if event.key == pygame.K_r:
                    player.reset()
                    won = False
                elif event.key in KEY_MOVES and not won:
                    player.move(*KEY_MOVES[event.key])
                    won = player.position == GOAL_POSITION
            elif mode == "watch":
                if event.key == pygame.K_r:
                    reset_watch()
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    WATCH_DELAY_MS = max(MIN_WATCH_DELAY, WATCH_DELAY_MS - 40)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    WATCH_DELAY_MS = min(MAX_WATCH_DELAY, WATCH_DELAY_MS + 40)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and mode == "menu":
            if BUTTONS["play"].collidepoint(event.pos):
                player.reset()
                won = False
                mode = "manual"
            elif BUTTONS["train"].collidepoint(event.pos):
                start_training()
            elif BUTTONS["watch"].collidepoint(event.pos) and trained_agent is not None:
                reset_watch()
                mode = "watch"
            elif BUTTONS["graph"].collidepoint(event.pos) and graph_surface is not None:
                mode = "graph"

    if mode == "watch" and not environment.done:
        now = pygame.time.get_ticks()
        if now - last_agent_move >= WATCH_DELAY_MS:
            action = trained_agent.choose_action(agent_position, training=False)
            agent_position, reward, _ = environment.step(action)
            agent_path.append(agent_position)
            agent_reward += reward
            last_agent_move = now

    screen.fill(BACKGROUND)
    if mode == "menu":
        draw_menu()
    elif mode in ("manual", "watch"):
        draw_maze_mode()
    elif mode == "graph":
        screen.blit(graph_surface, (0, 0))
        draw_centered("Esc: menu", small_font, TEXT, HEIGHT - 20)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
