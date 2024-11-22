import numpy as np
from game import Game2048
from graphics import Graphics2048
from agent import QLearningAgent
import pygame
import threading
import time

def agent_training_thread(game, agent, graphics):
    while True:
        if not game.is_game_over():
            previous_board = game.board.copy()  # Speichere den vorherigen Zustand für die Animation
            state = game.board.copy()
            
            # Agent wählt eine Aktion
            direction = agent.choose_action(state)

            # Bewegung ausführen
            game.move(direction)

            # Belohnung berechnen (z.B. Punktzuwachs)
            next_state = game.board.copy()
            reward = game.get_score() - np.sum(previous_board)

            # Agent lernt basierend auf der Belohnung
            agent.learn(state, direction, reward, next_state)

            # Aktualisiere die Agenteninformationen in der Grafik
            graphics.update_agent_info(reward)

            # Wartezeit zwischen den Zügen des Agenten, um die GUI zu entlasten
            time.sleep(0.1)
        else:
            # Falls das Spiel vorbei ist, neu starten
            time.sleep(2)  # Kleine Pause, um die Game-Over-Nachricht anzuzeigen
            game.__init__()  # Setzt das Spiel zurück

if __name__ == "__main__":
    game = Game2048()
    agent = QLearningAgent(actions=['up', 'down', 'left', 'right'])
    graphics = Graphics2048(game, agent)

    # Starte den Agenten in einem separaten Thread
    training_thread = threading.Thread(target=agent_training_thread, args=(game, agent, graphics))
    training_thread.daemon = True  # Beendet den Thread, wenn das Hauptprogramm beendet wird
    training_thread.start()

    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Update des Spielfelds (ohne Lernen im Hauptthread)
        graphics.update_display()
        graphics.tick(30)

    graphics.handle_quit_event()
