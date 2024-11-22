import numpy as np
import random

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        self.q_table = {}  # Q-Table wird dynamisch gefüllt
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def get_state_key(self, state):
        return tuple(state.flatten())  # Zustand als flacher Tupel zur Speicherung in der Q-Tabelle

    def choose_action(self, state):
        # Epsilon-Greedy-Strategie
        if np.random.rand() < self.exploration_rate:
            return random.choice(self.actions)
        else:
            state_key = self.get_state_key(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(len(self.actions))
            return self.actions[np.argmax(self.q_table[state_key])]

    def learn(self, current_state, action, reward, next_state):
        state_key = self.get_state_key(current_state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.actions))
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(len(self.actions))

        current_q = self.q_table[state_key][self.actions.index(action)]
        max_future_q = np.max(self.q_table[next_state_key])
        
        # Q-Learning-Formel
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[state_key][self.actions.index(action)] = new_q

        # Exploration reduzieren
        self.exploration_rate *= self.exploration_decay
