import numpy as np
import pickle
import os

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1, model_file='qtable.pkl'):
        self.actions = actions
        self.alpha = alpha  # Lernrate
        self.gamma = gamma  # Diskontierungsfaktor
        self.epsilon = epsilon  # Explorationsrate
        self.q_table = {}
        self.model_file = model_file
        self.load_model()  # Versuche, das Modell zu laden

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return self.get_best_action(state)

    def get_best_action(self, state):
        state_str = str(state)
        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(len(self.actions))
        return self.actions[np.argmax(self.q_table[state_str])]

    def learn(self, state, action, reward, next_state):
        state_str = str(state)
        next_state_str = str(next_state)

        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(len(self.actions))

        if next_state_str not in self.q_table:
            self.q_table[next_state_str] = np.zeros(len(self.actions))

        action_index = self.actions.index(action)
        predict = self.q_table[state_str][action_index]
        target = reward + self.gamma * np.max(self.q_table[next_state_str])
        self.q_table[state_str][action_index] += self.alpha * (target - predict)

    def save_model(self):
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_model(self):
        if os.path.exists(self.model_file):
            with open(self.model_file, 'rb') as f:
                self.q_table = pickle.load(f)
            print("Modell geladen.")
        else:
            print("Kein gespeichertes Modell gefunden, starte neues Training.")
