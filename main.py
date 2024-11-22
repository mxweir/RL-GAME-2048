import numpy as np
from game import Game2048
from graphics import Graphics2048
from agent import QLearningAgent
import pygame
import threading
import time

# Tracks whether the agent is currently running
agent_running = False

def agent_training_thread(game, agent, graphics):
    global agent_running
    while True:
        if agent_running and not game.is_game_over():
            previous_board = game.board.copy()  # Save previous state for animation
            state = game.board.copy()
            
            # Agent chooses an action
            direction = agent.choose_action(state)

            # Execute the move
            game.move(direction)

            # Calculate reward (e.g., score increase)
            next_state = game.board.copy()
            reward = game.get_score() - np.sum(previous_board)

            # Agent learns from the reward
            agent.learn(state, direction, reward, next_state)

            # Update agent information in the graphics
            graphics.update_agent_info(reward)

            # Delay between agent moves to reduce GUI load
            time.sleep(0.1)

        elif game.is_game_over():
            # Save the model after each episode
            agent.save_model()
            print("Model saved.")
            time.sleep(2)  # Pause to show game-over message
            game.__init__()  # Reset the game

if __name__ == "__main__":
    game = Game2048()
    agent = QLearningAgent(actions=['up', 'down', 'left', 'right'])
    graphics = Graphics2048(game, agent)

    # Start the agent in a separate thread
    training_thread = threading.Thread(target=agent_training_thread, args=(game, agent, graphics))
    training_thread.daemon = True  # Ensure the thread stops when the main program exits
    training_thread.start()

    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and not agent_running:
                    game.move('up')
                elif event.key == pygame.K_DOWN and not agent_running:
                    game.move('down')
                elif event.key == pygame.K_LEFT and not agent_running:
                    game.move('left')
                elif event.key == pygame.K_RIGHT and not agent_running:
                    game.move('right')
                elif event.key == pygame.K_a:
                    # Start or stop the agent with 'A' key
                    agent_running = not agent_running
                    graphics.set_agent_status(agent_running)
                elif event.key == pygame.K_r and game.is_game_over():
                    # Reset the game after game over
                    game = Game2048()
                    graphics = Graphics2048(game, agent)

        # Update the game board (without learning in the main thread)
        graphics.update_display()
        graphics.tick(30)

    graphics.handle_quit_event()
