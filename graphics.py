import pygame
import numpy as np

COLORS = {
    0: (204, 192, 179),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}

FONT_COLOR = (119, 110, 101)
GRID_SIZE = 4
CELL_SIZE = 100
MARGIN = 10
SCREEN_SIZE = GRID_SIZE * CELL_SIZE + (GRID_SIZE + 1) * MARGIN
INFO_HEIGHT = 260 
SCREEN_HEIGHT = SCREEN_SIZE + INFO_HEIGHT

class Graphics2048:
    def __init__(self, game, agent):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_HEIGHT))
        pygame.display.set_caption("Learn 2048 | mxweir")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 48)
        self.info_font = pygame.font.Font(None, 36)
        self.game = game
        self.agent = agent
        self.highscore = 0
        self.cumulative_reward = 0
        self.moves_made = 0
        self.improvements = 0
        self.agent_running = False 

    def set_agent_status(self, status):
        self.agent_running = status

    def draw_board(self, board, tile_positions=None):
        self.screen.fill(COLORS[0])
        
        # Zeichne das Spielfeld
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                value = board[row][col]
                color = COLORS.get(value, COLORS[2048])
                
                # Berechnung der Position
                if tile_positions and (row, col) in tile_positions:
                    current_x, current_y = tile_positions[(row, col)]
                else:
                    current_x = MARGIN + col * (CELL_SIZE + MARGIN)
                    current_y = MARGIN + row * (CELL_SIZE + MARGIN)
                
                # Zeichne das Rechteck für die Tile
                pygame.draw.rect(self.screen, color, (current_x, current_y, CELL_SIZE, CELL_SIZE), border_radius=8)
                
                # Zeichne den Wert auf das Tile
                if value > 0:
                    text = self.font.render(str(value), True, FONT_COLOR)
                    text_rect = text.get_rect(center=(current_x + CELL_SIZE / 2, current_y + CELL_SIZE / 2))
                    self.screen.blit(text, text_rect)

        # Aktualisiere den Highscore, wenn aktueller Score größer ist
        if self.game.get_score() > self.highscore:
            self.highscore = self.game.get_score()
            self.improvements += 1

        # Zeichne den aktuellen Score und den Highscore nebeneinander oben
        score_text = self.info_font.render(f"Score: {self.game.get_score()}", True, FONT_COLOR)
        highscore_text = self.info_font.render(f"Highscore: {self.highscore}", True, FONT_COLOR)
        self.screen.blit(score_text, (MARGIN, SCREEN_SIZE + 10))
        self.screen.blit(highscore_text, (SCREEN_SIZE // 2, SCREEN_SIZE + 10))

        # Zeichne die Agent-Infos unterhalb des Scores
        agent_info_x = MARGIN
        agent_info_y_start = SCREEN_SIZE + 60
        spacing = 40

        

        # Zeichne die kumulierte Belohnung
        cumulative_reward_text = self.info_font.render(f"Cumulative Reward: {self.cumulative_reward}", True, FONT_COLOR)
        self.screen.blit(cumulative_reward_text, (agent_info_x, agent_info_y_start))

        # Zeichne die Anzahl der Züge
        moves_text = self.info_font.render(f"Moves Made: {self.moves_made}", True, FONT_COLOR)
        self.screen.blit(moves_text, (agent_info_x, agent_info_y_start + spacing))

        # Zeichne die Anzahl der Verbesserungen
        improvements_text = self.info_font.render(f"Improvements: {self.improvements}", True, FONT_COLOR)
        self.screen.blit(improvements_text, (agent_info_x, agent_info_y_start + 2 * spacing))

        # Zeichne den Agentenstatus
        agent_status_text = "Running" if self.agent_running else "Stopped"
        agent_status_display = self.info_font.render(f"Agent: {agent_status_text}", True, FONT_COLOR)
        self.screen.blit(agent_status_display, (agent_info_x, agent_info_y_start + 3.3 * spacing))

        # Zeichne die Anleitung zum Starten und Stoppen des Agenten
        instructions_text = self.info_font.render("Press 'A' to Start/Stop Agent", True, FONT_COLOR)
        self.screen.blit(instructions_text, (MARGIN, SCREEN_SIZE + 220))

    def update_agent_info(self, reward):
        self.cumulative_reward += reward
        self.moves_made += 1

    def animate_move(self, previous_board, direction):
        current_board = self.game.board.copy()
        movement_info = self.get_movement_info(previous_board, current_board)
        frames = 10  # Anzahl der Frames für die Animation
        tile_positions = {}

        for row, col in movement_info.keys():
            current_x = MARGIN + col * (CELL_SIZE + MARGIN)
            current_y = MARGIN + row * (CELL_SIZE + MARGIN)
            tile_positions[(row, col)] = (current_x, current_y)

        # Interpoliere über mehrere Frames nur für die bewegten Tiles
        for frame in range(1, frames + 1):
            interpolation = frame / frames
            new_positions = {}

            for (row, col), (target_row, target_col) in movement_info.items():
                start_x, start_y = tile_positions[(row, col)]
                target_x = MARGIN + target_col * (CELL_SIZE + MARGIN)
                target_y = MARGIN + target_row * (CELL_SIZE + MARGIN)
                current_x = start_x + interpolation * (target_x - start_x)
                current_y = start_y + interpolation * (target_y - start_y)
                new_positions[(row, col)] = (current_x, current_y)

            # Zeichne das Board mit interpolierten Positionen
            self.draw_board(previous_board, tile_positions=new_positions)
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS für eine flüssige Animation

    def update_display(self, previous_board=None, direction=None):
        if previous_board is not None and direction is not None:
            self.animate_move(previous_board, direction)
        self.draw_board(self.game.board)
        pygame.display.flip()

    def show_game_over(self):
        # Zeige den Game-Over-Bildschirm
        game_over_text = self.info_font.render("Game Over! Press R to Restart", True, FONT_COLOR)
        self.screen.blit(game_over_text, (SCREEN_SIZE // 4, SCREEN_SIZE // 2))
        pygame.display.flip()

    def handle_quit_event(self):
        pygame.quit()

    def tick(self, fps=30):
        self.clock.tick(fps)
