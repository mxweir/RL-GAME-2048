import pygame
import numpy as np
import ttkbootstrap as ttk
import tkinter.simpledialog as simpledialog
from ttkbootstrap import Style
import os
import pickle

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
INFO_HEIGHT = 300  
SCREEN_HEIGHT = SCREEN_SIZE + INFO_HEIGHT

class Graphics2048:
    def __init__(self, game, agent):
        # Display model selection GUI
        self.select_model_gui(agent)

        pygame.init()
        icon_path = os.path.join('utils', 'favicon.ico')
        if os.path.exists(icon_path):
            icon = pygame.image.load(icon_path)
            pygame.display.set_icon(icon)
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_HEIGHT))
        pygame.display.set_caption("Learn 2048 | by mxweir")
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

    def select_model_gui(self, agent):
        style = Style(theme='cosmo') 
        root = style.master
        # Set window icon for tkinter GUI
        icon_path = os.path.join('utils', 'favicon.ico')
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
        root.title("Learn 2048 | by mxweir")

        # Main frame for all GUI elements
        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill='both', expand=True)

        # Frame for dropdown menu and load button
        model_frame = ttk.Frame(main_frame)
        model_frame.pack(pady=10, fill='x')

        # Dropdown to select an existing model
        model_files = [f for f in os.listdir() if f.endswith('.pkl')]
        selected_model = ttk.StringVar(root)
        selected_model.set("Select a model")
        model_dropdown = ttk.Combobox(model_frame, textvariable=selected_model, values=model_files, bootstyle='primary')
        model_dropdown.pack(side='left', padx=10, fill='x', expand=True)

        def load_model():
            model_file = selected_model.get()
            if model_file and os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    agent.q_table = pickle.load(f)
                    agent.model_file = model_file
                    print(f"Modell '{model_file}' geladen.")
                root.destroy()
            else:
                print("Bitte ein gültiges Modell auswählen.")

        load_button = ttk.Button(model_frame, text="Load", command=load_model, bootstyle='success-outline')
        load_button.pack(side='left', padx=10)

        # Frame for creating a new model
        new_model_frame = ttk.Labelframe(main_frame, text="Create New Model", padding=10, bootstyle='info')
        new_model_frame.pack(pady=20, fill='x')

        # Sliders for adjusting Alpha, Gamma, Epsilon
        def create_new_model():
            model_name = simpledialog.askstring("New Model", "Enter model name:")
            if model_name:
                agent.model_file = f"{model_name}.pkl"
            if model_name:
                agent.alpha = alpha_slider.get()
                agent.gamma = gamma_slider.get()
                agent.epsilon = epsilon_slider.get()
                agent.q_table = {} 
                agent.set_params(alpha=agent.alpha, gamma=agent.gamma, epsilon=agent.epsilon)
                with open(agent.model_file, 'wb') as f:
                    pickle.dump(agent.q_table, f)
                print("Neues Modell '{}' erstellt mit alpha={}, gamma={}, epsilon={}".format(model_name, agent.alpha, agent.gamma, agent.epsilon))
                root.destroy()
            else:
                print("Modellerstellung abgebrochen.")

        # Alpha slider with value display
        ttk.Label(new_model_frame, text="Alpha (learning rate)").pack(anchor='w')
        alpha_slider = ttk.Scale(new_model_frame, from_=0.0, to=1.0, value=agent.alpha, bootstyle='info', length=200, orient='horizontal')
        alpha_slider.pack(fill='x', pady=5)
        alpha_value_label = ttk.Label(new_model_frame, text=f"{agent.alpha:.2f}", bootstyle='secondary')
        alpha_value_label.pack(anchor='e', padx=10)
        alpha_slider.bind("<Motion>", lambda event: alpha_value_label.config(text=f"{alpha_slider.get():.2f}"))

        # Gamma slider with value display
        ttk.Label(new_model_frame, text="Gamma (discount factor)").pack(anchor='w')
        gamma_slider = ttk.Scale(new_model_frame, from_=0.0, to=1.0, value=agent.gamma, bootstyle='info', length=200, orient='horizontal')
        gamma_slider.pack(fill='x', pady=5)
        gamma_value_label = ttk.Label(new_model_frame, text=f"{agent.gamma:.2f}", bootstyle='secondary')
        gamma_value_label.pack(anchor='e', padx=10)
        gamma_slider.bind("<Motion>", lambda event: gamma_value_label.config(text=f"{gamma_slider.get():.2f}"))

        # Epsilon slider with value display
        ttk.Label(new_model_frame, text="Epsilon (exploration rate)").pack(anchor='w')
        epsilon_slider = ttk.Scale(new_model_frame, from_=0.0, to=1.0, value=agent.epsilon, bootstyle='info', length=200, orient='horizontal')
        epsilon_slider.pack(fill='x', pady=5)
        epsilon_value_label = ttk.Label(new_model_frame, text=f"{agent.epsilon:.2f}", bootstyle='secondary')
        epsilon_value_label.pack(anchor='e', padx=10)
        epsilon_slider.bind("<Motion>", lambda event: epsilon_value_label.config(text=f"{epsilon_slider.get():.2f}"))

        # Button to create a new model
        create_button = ttk.Button(new_model_frame, text="Create New Model", command=create_new_model, bootstyle='primary')
        create_button.pack(pady=10)

        root.mainloop()

    def set_agent_status(self, status):
        self.agent_running = status

    def draw_board(self, board, tile_positions=None):
        self.screen.fill(COLORS[0])
        
        # Draw the game board
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                value = board[row][col]
                color = COLORS.get(value, COLORS[2048])
                
                # Calculate position
                if tile_positions and (row, col) in tile_positions:
                    current_x, current_y = tile_positions[(row, col)]
                else:
                    current_x = MARGIN + col * (CELL_SIZE + MARGIN)
                    current_y = MARGIN + row * (CELL_SIZE + MARGIN)
                
                # Draw the rectangle for the tile
                pygame.draw.rect(self.screen, color, (current_x, current_y, CELL_SIZE, CELL_SIZE), border_radius=8)
                
                # Draw the value on the tile
                if value > 0:
                    text = self.font.render(str(value), True, FONT_COLOR)
                    text_rect = text.get_rect(center=(current_x + CELL_SIZE / 2, current_y + CELL_SIZE / 2))
                    self.screen.blit(text, text_rect)

        # Update high score if the current score is greater
        if self.game.get_score() > self.highscore:
            self.highscore = self.game.get_score()
            self.improvements += 1

        # Draw current score and high score at the top
        score_text = self.info_font.render(f"Score: {self.game.get_score()}", True, FONT_COLOR)
        highscore_text = self.info_font.render(f"Highscore: {self.highscore}", True, FONT_COLOR)
        self.screen.blit(score_text, (MARGIN, SCREEN_SIZE + 10))
        self.screen.blit(highscore_text, (SCREEN_SIZE // 2, SCREEN_SIZE + 10))

        # Draw agent information below the scores
        agent_info_x = MARGIN
        agent_info_y_start = SCREEN_SIZE + 60
        spacing = 40

        # Draw cumulative reward
        cumulative_reward_text = self.info_font.render(f"Cumulative Reward: {self.cumulative_reward}", True, FONT_COLOR)
        self.screen.blit(cumulative_reward_text, (agent_info_x, agent_info_y_start))

        # Draw number of moves made
        moves_text = self.info_font.render(f"Moves Made: {self.moves_made}", True, FONT_COLOR)
        self.screen.blit(moves_text, (agent_info_x, agent_info_y_start + spacing))

        # Draw number of improvements
        improvements_text = self.info_font.render(f"Improvements: {self.improvements}", True, FONT_COLOR)
        self.screen.blit(improvements_text, (agent_info_x, agent_info_y_start + 2  * spacing))

        # Draw agent status
        agent_status_text = "Running" if self.agent_running else "Stopped"
        agent_status_display = self.info_font.render(f"Agent: {agent_status_text}", True, FONT_COLOR)
        self.screen.blit(agent_status_display, (agent_info_x, agent_info_y_start + 3.2 * spacing))

        # Zeichne den Agentenstatus
        model_name_text = f"Model: {self.agent.model_file}" if hasattr(self.agent, 'model_file') else "Model: None"
        model_name_display = self.info_font.render(model_name_text, True, FONT_COLOR)
        self.screen.blit(model_name_display, (agent_info_x, agent_info_y_start + 4 * spacing))

        # Draw instructions for starting and stopping the agent
        instructions_text = self.info_font.render("Press 'A' to Start/Stop Agent", True, FONT_COLOR)
        self.screen.blit(instructions_text, (MARGIN, SCREEN_SIZE + 250))

    def update_agent_info(self, reward):
        self.cumulative_reward += reward
        self.moves_made += 1

    def animate_move(self, previous_board, direction):
        current_board = self.game.board.copy()
        movement_info = self.get_movement_info(previous_board, current_board)
        frames = 10 
        tile_positions = {}

        for row, col in movement_info.keys():
            current_x = MARGIN + col * (CELL_SIZE + MARGIN)
            current_y = MARGIN + row * (CELL_SIZE + MARGIN)
            tile_positions[(row, col)] = (current_x, current_y)

        # Interpolate over multiple frames for moving tiles only
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

            # Draw the board with interpolated positions
            self.draw_board(previous_board, tile_positions=new_positions)
            pygame.display.flip()
            self.clock.tick(60) 

    def update_display(self, previous_board=None, direction=None):
        if previous_board is not None and direction is not None:
            self.animate_move(previous_board, direction)
        self.draw_board(self.game.board)
        pygame.display.flip()

    def show_game_over(self):
        # Display the game-over screen
        game_over_text = self.info_font.render("Game Over! Press R to Restart", True, FONT_COLOR)
        self.screen.blit(game_over_text, (SCREEN_SIZE // 4, SCREEN_SIZE // 2))
        pygame.display.flip()

    def handle_quit_event(self):
        pygame.quit()

    def tick(self, fps=30):
        self.clock.tick(fps)
