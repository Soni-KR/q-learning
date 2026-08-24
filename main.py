from pathlib import Path

import pygame

from agent import QLearningAgent
from environment import MazeEnvironment
from maze import MAZES, TILE_SIZE, draw_maze
from player import Player
from training import run_episode, train_agent
from visualization import save_training_plot

pygame.init()

WIDTH, HEIGHT = 800, 700
HUD_TOP = 555
BACKGROUND = (8, 25, 48)
PANEL = (14, 42, 73)
TEXT = (247, 250, 252)
MUTED = (159, 188, 209)
BLUE = (38, 139, 230)
ORANGE = (243, 116, 46)
GOLD = (255, 202, 40)
BUTTON = (24, 74, 113)
BUTTON_HOVER = (35, 105, 155)

APP_ICON_PATH = Path(__file__).with_name("assets") / "sonic-maze-icon.png"
app_icon = pygame.image.load(str(APP_ICON_PATH))
pygame.display.set_icon(app_icon)
menu_icon = pygame.transform.smoothscale(app_icon, (88, 88))

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Velocity Maze: Human vs Q-Learner")
clock = pygame.time.Clock()
title_font = pygame.font.Font(None, 52)
font = pygame.font.Font(None, 28)
small_font = pygame.font.Font(None, 21)

maze_index = 0
selected_maze = MAZES[maze_index]
player = Player(selected_maze)
environment = MazeEnvironment(selected_maze)
trained_agents = {}
training_history = None
graph_surface = None
mode = "menu"
status_message = "Pick a circuit, then race yourself or the learner."
won = False
agent_position = environment.state
agent_reward = 0
agent_path = [agent_position]
last_agent_move = 0
watch_delay_ms = 240

# Race state: the human moves first; every valid tile funds live AI training.
race_agent = None
race_environment = None
race_training_environment = None
race_agent_position = None
race_agent_path = []
race_episodes = 0
race_result = ""

KEY_MOVES = {
    pygame.K_UP: (-1, 0), pygame.K_DOWN: (1, 0),
    pygame.K_LEFT: (0, -1), pygame.K_RIGHT: (0, 1),
}
ACTION_FOR_KEY = {
    pygame.K_UP: MazeEnvironment.UP, pygame.K_DOWN: MazeEnvironment.DOWN,
    pygame.K_LEFT: MazeEnvironment.LEFT, pygame.K_RIGHT: MazeEnvironment.RIGHT,
}

BUTTONS = {
    "maze": pygame.Rect(215, 120, 370, 48),
    "play": pygame.Rect(215, 190, 370, 48),
    "race": pygame.Rect(215, 250, 370, 58),
    "train": pygame.Rect(215, 320, 370, 48),
    "watch": pygame.Rect(215, 380, 370, 48),
    "graph": pygame.Rect(215, 440, 370, 48),
}


def model_path(maze):
    return Path(__file__).with_name(f"q_table_{maze.key}.npz")


def maze_offset(maze):
    return ((WIDTH - maze.cols * TILE_SIZE) // 2, 20)


def cell_center(position, maze):
    row, col = position
    ox, oy = maze_offset(maze)
    return ox + col * TILE_SIZE + TILE_SIZE // 2, oy + row * TILE_SIZE + TILE_SIZE // 2


def draw_centered(text, used_font, color, y):
    surface = used_font.render(text, True, color)
    screen.blit(surface, surface.get_rect(center=(WIDTH // 2, y)))


def draw_button(rect, label, enabled=True, accent=BLUE):
    hovered = enabled and rect.collidepoint(pygame.mouse.get_pos())
    color = BUTTON_HOVER if hovered else BUTTON
    if not enabled:
        color = (27, 48, 65)
    pygame.draw.rect(screen, color, rect, border_radius=9)
    pygame.draw.rect(screen, accent if enabled else (70, 91, 105), rect, 2, 9)
    surface = font.render(label, True, TEXT if enabled else MUTED)
    screen.blit(surface, surface.get_rect(center=rect.center))


def draw_speed_background():
    for index in range(14):
        y = 20 + index * 48
        length = 80 + (index * 37) % 180
        x = (index * 91) % WIDTH
        pygame.draw.line(screen, (12, 53, 88), (x, y), (min(WIDTH, x + length), y), 3)
    for row in range(2):
        for col in range(20):
            color = (20, 62, 94) if (row + col) % 2 else (10, 36, 65)
            pygame.draw.rect(screen, color, (col * 40, HEIGHT - 80 + row * 40, 40, 40))


def draw_racer(position, color, outline, radius=13):
    center = cell_center(position, selected_maze)
    pygame.draw.circle(screen, color, center, radius)
    pygame.draw.circle(screen, outline, center, radius, 3)
    pygame.draw.polygon(screen, (245, 245, 245), [
        (center[0] - radius, center[1] + 5),
        (center[0] - radius - 10, center[1] + 10),
        (center[0] - radius + 1, center[1] + 11),
    ])


def draw_path(path, color):
    if len(path) > 1:
        pygame.draw.lines(screen, color, False,
                          [cell_center(pos, selected_maze) for pos in path], 4)


def draw_menu():
    draw_speed_background()
    screen.blit(menu_icon, (24, 18))
    # A few floating rings reinforce the classic speed-game reference.
    for x, y, radius in ((690, 44, 18), (742, 76, 12), (650, 92, 9)):
        pygame.draw.circle(screen, GOLD, (x, y), radius, 4)
        pygame.draw.circle(screen, (255, 238, 145), (x - 2, y - 2), radius - 4, 2)
    draw_centered("VELOCITY MAZE", title_font, TEXT, 48)
    draw_centered("HUMAN vs Q-LEARNER", font, GOLD, 84)
    draw_button(BUTTONS["maze"], f"CIRCUIT: {selected_maze.name}", accent=GOLD)
    draw_button(BUTTONS["play"], "TIME TRIAL")
    draw_button(BUTTONS["race"], "RACE THE LEARNER", accent=ORANGE)
    draw_button(BUTTONS["train"], "TRAIN THIS CIRCUIT")
    ready = selected_maze.key in trained_agents
    draw_button(BUTTONS["watch"], "WATCH BEST RUN", ready)
    draw_button(BUTTONS["graph"], "LEARNING TELEMETRY", training_history is not None)
    draw_centered(status_message, small_font, MUTED, 520)
    draw_centered("Click CIRCUIT to change maze  •  Esc returns to menu", small_font, TEXT, 670)


def draw_course():
    draw_maze(screen, selected_maze, maze_offset(selected_maze))
    pygame.draw.rect(screen, PANEL, (0, HUD_TOP, WIDTH, HEIGHT - HUD_TOP))


def draw_game_mode():
    draw_course()
    if mode == "manual":
        draw_racer(player.position, BLUE, (11, 76, 145), 14)
        headline = f"TIME TRIAL   MOVES {player.moves}"
        detail = "Arrow keys: sprint   R: restart   Esc: menu"
        if won:
            headline += "   •   RING REACHED!"
    elif mode == "watch":
        draw_path(agent_path, GOLD)
        draw_racer(agent_position, ORANGE, (150, 55, 20), 14)
        headline = f"Q-LEARNER   STEP {environment.steps}   REWARD {agent_reward}"
        detail = f"R: replay   +/-: speed ({1000 / watch_delay_ms:.1f}/s)   Esc: menu"
        if environment.done:
            headline += "   •   OPTIMAL RUN COMPLETE"
    else:
        draw_path(race_agent_path, (255, 178, 73))
        draw_racer(player.position, BLUE, (11, 76, 145), 13)
        draw_racer(race_agent_position, ORANGE, (150, 55, 20), 10)
        headline = f"YOU {player.moves} MOVES   vs   AI {race_environment.steps} MOVES"
        detail = (
            f"AI training: {race_episodes} episodes   •   "
            f"{selected_maze.training_episodes_per_move} episodes earned per human tile"
        )
        if race_result:
            headline = race_result
            detail = "R: rematch   Esc: menu"

    screen.blit(font.render(headline, True, TEXT), (22, HUD_TOP + 20))
    screen.blit(small_font.render(detail, True, MUTED), (22, HUD_TOP + 58))
    screen.blit(small_font.render(selected_maze.name, True, GOLD), (22, HUD_TOP + 95))


def select_next_maze():
    global maze_index, selected_maze, player, environment, training_history, graph_surface
    maze_index = (maze_index + 1) % len(MAZES)
    selected_maze = MAZES[maze_index]
    player = Player(selected_maze)
    environment = MazeEnvironment(selected_maze)
    training_history = None
    graph_surface = None
    load_saved_agent(selected_maze)


def load_saved_agent(maze):
    path = model_path(maze)
    if not path.exists():
        return False
    try:
        agent = QLearningAgent.load(path, seed=42)
        if agent.q_table.shape != (maze.rows, maze.cols, 4):
            return False
        trained_agents[maze.key] = agent
        return True
    except (OSError, ValueError, KeyError):
        return False


def train_selected_maze():
    global training_history, graph_surface, status_message
    episodes = 1000 + selected_maze.rows * selected_maze.cols * 3
    agent, training_history = train_agent(
        episodes=episodes, max_steps=selected_maze.rows * selected_maze.cols * 3,
        report_every=0, maze=selected_maze,
    )
    trained_agents[selected_maze.key] = agent
    agent.save(model_path(selected_maze))
    graph_path = Path(__file__).with_name(f"training_{selected_maze.key}.png")
    save_training_plot(training_history, graph_path)
    loaded = pygame.image.load(str(graph_path)).convert()
    graph_surface = pygame.transform.smoothscale(loaded, (WIDTH, HEIGHT - 42))
    recent = training_history[-100:]
    success = sum(item.reached_goal for item in recent) / len(recent)
    status_message = f"{selected_maze.name}: training complete, {success:.0%} recent success."


def reset_watch():
    global environment, agent_position, agent_reward, agent_path, last_agent_move
    environment = MazeEnvironment(selected_maze)
    agent_position = environment.reset()
    agent_reward = 0
    agent_path = [agent_position]
    last_agent_move = pygame.time.get_ticks()


def reset_race():
    global player, race_agent, race_environment, race_training_environment
    global race_agent_position, race_agent_path, race_episodes, race_result
    player = Player(selected_maze)
    race_agent = QLearningAgent(
        selected_maze.rows, selected_maze.cols, epsilon_decay=0.992, seed=7
    )
    race_environment = MazeEnvironment(selected_maze)
    race_training_environment = MazeEnvironment(selected_maze)
    race_agent_position = race_environment.reset()
    race_agent_path = [race_agent_position]
    race_episodes = 0
    race_result = ""


def take_race_turn(key):
    global race_agent_position, race_episodes, race_result
    dr, dc = KEY_MOVES[key]
    if not player.move(dr, dc):
        return

    budget = selected_maze.training_episodes_per_move
    max_steps = selected_maze.rows * selected_maze.cols * 3
    for _ in range(budget):
        run_episode(race_training_environment, race_agent, max_steps=max_steps, training=True)
        race_agent.decay_exploration()
    race_episodes += budget

    if not race_environment.done:
        action = race_agent.choose_action(race_agent_position, training=False)
        race_agent_position, _, _ = race_environment.step(action)
        race_agent_path.append(race_agent_position)

    human_done = player.position == race_environment.goal_position
    ai_done = race_environment.done
    if human_done and ai_done:
        race_result = "PHOTO FINISH — DRAW!"
    elif human_done:
        race_result = "YOU WIN — PURE SPEED!"
    elif ai_done:
        race_result = "Q-LEARNER WINS — ADAPT AND REMATCH!"


for maze in MAZES:
    load_saved_agent(maze)

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
                    player.reset(); won = False
                elif event.key in KEY_MOVES and not won:
                    player.move(*KEY_MOVES[event.key])
                    won = player.position == environment.goal_position
            elif mode == "race":
                if event.key == pygame.K_r:
                    reset_race()
                elif event.key in KEY_MOVES and not race_result:
                    take_race_turn(event.key)
            elif mode == "watch":
                if event.key == pygame.K_r:
                    reset_watch()
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    watch_delay_ms = max(50, watch_delay_ms - 40)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    watch_delay_ms = min(800, watch_delay_ms + 40)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and mode == "menu":
            if BUTTONS["maze"].collidepoint(event.pos):
                select_next_maze()
            elif BUTTONS["play"].collidepoint(event.pos):
                player = Player(selected_maze); environment = MazeEnvironment(selected_maze)
                won = False; mode = "manual"
            elif BUTTONS["race"].collidepoint(event.pos):
                reset_race(); mode = "race"
            elif BUTTONS["train"].collidepoint(event.pos):
                train_selected_maze()
            elif BUTTONS["watch"].collidepoint(event.pos) and selected_maze.key in trained_agents:
                reset_watch(); mode = "watch"
            elif BUTTONS["graph"].collidepoint(event.pos) and graph_surface is not None:
                mode = "graph"

    if mode == "watch" and not environment.done:
        now = pygame.time.get_ticks()
        if now - last_agent_move >= watch_delay_ms:
            action = trained_agents[selected_maze.key].choose_action(agent_position, training=False)
            agent_position, reward, _ = environment.step(action)
            agent_reward += reward
            agent_path.append(agent_position)
            last_agent_move = now

    screen.fill(BACKGROUND)
    if mode == "menu":
        draw_menu()
    elif mode in ("manual", "watch", "race"):
        draw_game_mode()
    elif mode == "graph":
        screen.blit(graph_surface, (0, 0))
        draw_centered("Esc: return to circuit menu", small_font, TEXT, HEIGHT - 18)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
