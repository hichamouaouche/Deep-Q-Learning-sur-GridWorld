import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Environment
GRID_SIZE = 4
STATE_SIZE = GRID_SIZE * GRID_SIZE
ACTION_SIZE = 4  # up, down, left, right

# Training
GAMMA = 0.9
LEARNING_RATE = 0.01
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 2000
EPISODES = 1000
MAX_STEPS = 50
TARGET_UPDATE_EVERY = 10  # episodes

MOVES = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}


class GridWorld:
    """Simple 4x4 grid environment."""

    def __init__(self):
        self.grid_size = GRID_SIZE
        self.reset()

    def reset(self):
        self.agent_pos = (0, 0)
        self.goal_pos = (3, 3)
        self.obstacle_pos = (1, 1)
        return self.get_state()

    def get_state(self):
        state = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        state[self.agent_pos] = 1.0
        return state.flatten()

    def step(self, action):
        x, y = self.agent_pos
        dx, dy = MOVES[action]
        new_x, new_y = x + dx, y + dy

        if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
            self.agent_pos = (new_x, new_y)

        if self.agent_pos == self.goal_pos:
            return self.get_state(), 10.0, True
        if self.agent_pos == self.obstacle_pos:
            return self.get_state(), -5.0, False
        return self.get_state(), -1.0, False


class DoubleDQNAgent:
    """Double DQN agent: online network selects action, target network evaluates it."""

    def __init__(self):
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON

        self.online_model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential([
            Dense(24, activation="relu", input_shape=(self.state_size,)),
            Dense(24, activation="relu"),
            Dense(self.action_size, activation="linear"),
        ])
        model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.online_model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        q_values = self.online_model.predict(np.array([state]), verbose=0)
        return int(np.argmax(q_values[0]))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)

        states = np.array([x[0] for x in batch], dtype=np.float32)
        actions = np.array([x[1] for x in batch], dtype=np.int32)
        rewards = np.array([x[2] for x in batch], dtype=np.float32)
        next_states = np.array([x[3] for x in batch], dtype=np.float32)
        dones = np.array([x[4] for x in batch], dtype=bool)

        current_q = self.online_model.predict(states, verbose=0)
        next_q_online = self.online_model.predict(next_states, verbose=0)
        next_q_target = self.target_model.predict(next_states, verbose=0)

        targets = np.array(current_q, copy=True)

        for i in range(BATCH_SIZE):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                best_next_action = int(np.argmax(next_q_online[i]))
                targets[i, actions[i]] = rewards[i] + GAMMA * next_q_target[i, best_next_action]

        self.online_model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(self.epsilon, EPSILON_MIN)


def train_double_dqn():
    env = GridWorld()
    agent = DoubleDQNAgent()

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

        agent.replay()

        if episode % TARGET_UPDATE_EVERY == 0:
            agent.update_target_model()

        print(
            f"Episode {episode}/{EPISODES} | "
            f"Score: {total_reward:.1f} | "
            f"Epsilon: {agent.epsilon:.4f}"
        )

    agent.online_model.save("my_model.keras")
    print("Training complete. Model saved to my_model.keras")


if __name__ == "__main__":
    train_double_dqn()
