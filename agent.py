import numpy as np
import pickle
import os

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1, model_file=None, epsilon_decay_rate=0.995):
        self.actions = actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}
        self.model_file = model_file
        self.epsilon_decay_rate = epsilon_decay_rate
        # Load initial model if specified
        if self.model_file:
            self.load_model()

    def set_params(self, alpha=None, gamma=None, epsilon=None):
        """ Set agent parameters """
        if alpha is not None:
            self.alpha = alpha
        if gamma is not None:
            self.gamma = gamma
        if epsilon is not None:
            self.epsilon = epsilon
        print(f"Agent parameters set: alpha={self.alpha}, gamma={self.gamma}, epsilon={self.epsilon}")

    def choose_action(self, state):
        # Choose action based on epsilon-greedy policy
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
        # Decay epsilon to improve exploration-exploitation balance
        self.epsilon = max(0.1, self.epsilon * self.epsilon_decay_rate)
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
        if self.model_file:
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.q_table, f)
            print(f"Model '{self.model_file}' saved.")

    def load_model(self):
        if self.model_file and os.path.exists(self.model_file):
            with open(self.model_file, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Model '{self.model_file}' loaded.")
        else:
            print(f"No saved model found at '{self.model_file}', starting new training.")
