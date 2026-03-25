"""
Script complet : Entraînement Double DQN + Génération de Graphiques
Génère des figures professionnelles et les sauvegarde dans dossier "figures/"
"""

import random
from collections import deque
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION & PARAMÈTRES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

# Dossier de sortie pour les figures
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENVIRONNEMENT GRIDWORLD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT DOUBLE DQN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRAÎNEMENT AVEC TRAÇAGE DES MÉTRIQUES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_double_dqn():
    """Entraîne l'agent et sauvegarde les métriques."""
    env = GridWorld()
    agent = DoubleDQNAgent()

    # Listes pour stocker les métriques
    episode_rewards = []
    episode_epsilons = []
    episode_steps = []
    avg_q_values = []

    print("=" * 70)
    print("ENTRAÎNEMENT DOUBLE DQN - GridWorld 4x4")
    print("=" * 70)

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0.0
        steps = 0

        for step in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        agent.replay()

        if episode % TARGET_UPDATE_EVERY == 0:
            agent.update_target_model()

        # Sauvegarde des métriques
        episode_rewards.append(total_reward)
        episode_epsilons.append(agent.epsilon)
        episode_steps.append(steps)

        # Affichage du progression tous les 100 épisodes
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode:4d}/{EPISODES} | Avg Score (100): {avg_reward:7.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | Steps: {steps}")

    # Sauvegarde du modèle
    agent.online_model.save("my_model.keras")
    print("\n✓ Entraînement complété. Modèle sauvegardé à 'my_model.keras'")

    return episode_rewards, episode_epsilons, episode_steps, agent, env


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GENERATION DES GRAPHIQUES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_training_curve(rewards):
    """Graphique 1 : Courbe d'entraînement"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(rewards, label='Score par Episode', alpha=0.6, color='blue', linewidth=0.8)

    # Moyenne mobile (100 episodes)
    avg_100 = np.convolve(rewards, np.ones(100) / 100, mode='valid')
    ax.plot(range(100, len(rewards) + 1), avg_100, label='Moyenne Mobile (100)', 
            color='red', linewidth=2)

    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Récompense Cumulative', fontsize=12, fontweight='bold')
    ax.set_title('Double DQN - Courbe d\'Entraînement (GridWorld 4x4)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    filepath = os.path.join(FIGURES_DIR, '01_training_curve.png')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Sauvegardé : {filepath}")
    plt.close()


def plot_epsilon_decay(epsilons):
    """Graphique 2 : Décroissance d'Epsilon"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(epsilons, color='green', linewidth=2, label='Epsilon (taux d\'exploration)')
    ax.fill_between(range(len(epsilons)), epsilons, alpha=0.3, color='green')

    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Epsilon', fontsize=12, fontweight='bold')
    ax.set_title('Décroissance d\'Epsilon - Stratégie ε-Greedy', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    filepath = os.path.join(FIGURES_DIR, '02_epsilon_decay.png')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Sauvegardé : {filepath}")
    plt.close()


def plot_reward_distribution(rewards):
    """Graphique 3 : Distribution des récompenses"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Histogramme
    ax.hist(rewards, bins=50, color='purple', alpha=0.7, edgecolor='black')

    # Statistiques
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    std_reward = np.std(rewards)

    ax.axvline(mean_reward, color='red', linestyle='--', linewidth=2, label=f'Moyenne: {mean_reward:.2f}')
    ax.axvline(median_reward, color='orange', linestyle='--', linewidth=2, label=f'Médiane: {median_reward:.2f}')

    ax.set_xlabel('Récompense', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fréquence (Episodes)', fontsize=12, fontweight='bold')
    ax.set_title(f'Distribution des Récompenses (μ={mean_reward:.2f}, σ={std_reward:.2f})', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=11)

    filepath = os.path.join(FIGURES_DIR, '03_reward_distribution.png')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Sauvegardé : {filepath}")
    plt.close()


def plot_rewards_by_phase(rewards):
    """Graphique 4 : Récompenses par phase d'apprentissage"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Diviser en 4 phases
    phase_size = len(rewards) // 4
    phases = [
        rewards[:phase_size],
        rewards[phase_size:2*phase_size],
        rewards[2*phase_size:3*phase_size],
        rewards[3*phase_size:]
    ]
    phase_names = ['Phase 1\n(Exploration)', 'Phase 2\n(Apprentissage)', 
                   'Phase 3\n(Convergence)', 'Phase 4\n(Stabilisation)']
    phase_means = [np.mean(p) for p in phases]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax.bar(phase_names, phase_means, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Ajouter les valeurs sur les barres
    for bar, mean in zip(bars, phase_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Récompense Moyenne', fontsize=12, fontweight='bold')
    ax.set_title('Récompenses Moyennes par Phase d\'Apprentissage', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    filepath = os.path.join(FIGURES_DIR, '04_rewards_by_phase.png')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Sauvegardé : {filepath}")
    plt.close()


def plot_steps_per_episode(steps):
    """Graphique 5 : Nombre de pas par épisode"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(steps, alpha=0.6, color='darkorange', linewidth=0.8, label='Pas par Episode')

    # Moyenne mobile
    avg_steps = np.convolve(steps, np.ones(50) / 50, mode='valid')
    ax.plot(range(50, len(steps) + 1), avg_steps, color='red', linewidth=2, label='Moyenne Mobile (50)')

    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Nombre de Pas', fontsize=12, fontweight='bold')
    ax.set_title('Efficacité de l\'Agent - Pas par Episode', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, MAX_STEPS + 5])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    filepath = os.path.join(FIGURES_DIR, '05_steps_per_episode.png')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Sauvegardé : {filepath}")
    plt.close()


def plot_gridworld_visualization(env, agent):
    """Graphique 6 : Visualisation du GridWorld et du parcours optimal"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Subplot 1 : Grille avec positions ---
    ax = axes[0]
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    # Grille
    for i in range(GRID_SIZE + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1)
        ax.axvline(i - 0.5, color='black', linewidth=1)

    # Position initiale (START)
    circle_start = patches.Circle((0, 0), 0.25, color='green', alpha=0.8)
    ax.add_patch(circle_start)
    ax.text(0, 0, 'S', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Obstacle
    square_obs = patches.Rectangle((0.75, 0.75), 0.5, 0.5, color='red', alpha=0.8)
    ax.add_patch(square_obs)
    ax.text(1, 1, '✗', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    # Objectif (GOAL)
    circle_goal = patches.Circle((3, 3), 0.25, color='blue', alpha=0.8)
    ax.add_patch(circle_goal)
    ax.text(3, 3, 'G', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_title('Environnement GridWorld 4×4', fontsize=12, fontweight='bold')
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)

    # Légende
    ax.text(0.02, 0.98, 'S = Départ\n✗ = Obstacle\nG = Objectif', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- Subplot 2 : Q-values Heatmap ---
    ax = axes[1]

    # Calcul des Q-values moyennes pour chaque position
    q_map = np.zeros((GRID_SIZE, GRID_SIZE))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            state = np.zeros(STATE_SIZE, dtype=np.float32)
            state[i * GRID_SIZE + j] = 1.0
            q_values = agent.online_model.predict(np.array([state]), verbose=0)[0]
            q_map[i, j] = np.max(q_values)

    im = ax.imshow(q_map, cmap='YlOrRd', alpha=0.8)
    
    # Ajouter les valeurs dans les cellules
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            text = ax.text(j, i, f'{q_map[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')

    # Marquer START, OBSTACLE, GOAL
    ax.plot(0, 0, 'go', markersize=15, label='Start')
    ax.plot(1, 1, 'rx', markersize=15, markeredgewidth=3, label='Obstacle')
    ax.plot(3, 3, 'b*', markersize=20, label='Goal')

    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_title('Heatmap Q-values (Max Q par Position)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Max Q-value', fontsize=10)

    plt.tight_layout()
    filepath = os.path.join(FIGURES_DIR, '06_gridworld_visualization.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Sauvegardé : {filepath}")
    plt.close()


def plot_comparison_metrics(rewards, steps):
    """Graphique 7 : Dashboard de comparaison"""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Courbe d'entraînement
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(rewards, alpha=0.5, color='blue', linewidth=0.8)
    avg_100 = np.convolve(rewards, np.ones(100) / 100, mode='valid')
    ax1.plot(range(100, len(rewards) + 1), avg_100, color='red', linewidth=2)
    ax1.set_ylabel('Récompense', fontweight='bold')
    ax1.set_title('Courbe d\'Entraînement - Double DQN', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. Statistiques principales
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    stats_text = f"""
RÉSUMÉ D'ENTRAÎNEMENT
───────────────────────
Episodes:           {len(rewards)}
Récompense Min:     {min(rewards):.2f}
Récompense Max:     {max(rewards):.2f}
Récompense Moyenne: {np.mean(rewards):.2f}
Récompense Finale:  {rewards[-1]:.2f}
Std Dev:            {np.std(rewards):.2f}

EFFICACITÉ
───────────────────────
Pas Min:            {min(steps)}
Pas Max:            {max(steps)}
Pas Moyen:          {np.mean(steps):.1f}
Temps Final:        {steps[-1]} pas
    """
    ax2.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # 3. Histogramme récompenses
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(rewards, bins=40, color='purple', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Récompense', fontweight='bold')
    ax3.set_ylabel('Fréquence', fontweight='bold')
    ax3.set_title('Distribution des Récompenses', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Pas par épisode
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(steps, alpha=0.6, color='darkorange', linewidth=0.8)
    avg_steps = np.convolve(steps, np.ones(50) / 50, mode='valid')
    ax4.plot(range(50, len(steps) + 1), avg_steps, color='red', linewidth=2)
    ax4.set_xlabel('Episode', fontweight='bold')
    ax4.set_ylabel('Pas', fontweight='bold')
    ax4.set_title('Efficacité - Pas par Episode', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Progression par quartile
    ax5 = fig.add_subplot(gs[2, 1])
    quartile_size = len(rewards) // 4
    q_rewards = [
        np.mean(rewards[:quartile_size]),
        np.mean(rewards[quartile_size:2*quartile_size]),
        np.mean(rewards[2*quartile_size:3*quartile_size]),
        np.mean(rewards[3*quartile_size:])
    ]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax5.bar(['Q1', 'Q2', 'Q3', 'Q4'], q_rewards, color=colors, alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Récompense Moyenne', fontweight='bold')
    ax5.set_title('Progression par Quartile', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, q_rewards):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

    filepath = os.path.join(FIGURES_DIR, '07_comparison_dashboard.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Sauvegardé : {filepath}")
    plt.close()


def create_summary_report(rewards, epsilons, steps):
    """Crée un fichier texte résumé des résultats"""
    filepath = os.path.join(FIGURES_DIR, 'training_summary.txt')
    
    report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 RAPPORT D'ENTRAÎNEMENT - DOUBLE DQN                          ║
║                        GridWorld 4×4 - 1000 Episodes                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 CONFIGURATION D'ENTRAÎNEMENT
───────────────────────────────────────────────────────────────────────────────
  • Environnement       : GridWorld 4×4
  • Episodes            : {EPISODES}
  • Max Steps/Episode   : {MAX_STEPS}
  • Batch Size          : {BATCH_SIZE}
  • Memory Size         : {MEMORY_SIZE}
  • Gamma               : {GAMMA}
  • Learning Rate       : {LEARNING_RATE}
  • Epsilon Initial     : {EPSILON}
  • Epsilon Min         : {EPSILON_MIN}
  • Epsilon Decay       : {EPSILON_DECAY}
  • Target Update       : Tous les {TARGET_UPDATE_EVERY} episodes

🎯 RÉSULTATS DE PERFORMANCE
───────────────────────────────────────────────────────────────────────────────
  • Récompense Moyenne Totale   : {np.mean(rewards):>8.2f}
  • Récompense Maximale         : {max(rewards):>8.2f}
  • Récompense Minimale         : {min(rewards):>8.2f}
  • Écart Type                  : {np.std(rewards):>8.2f}
  • Récompense Finale           : {rewards[-1]:>8.2f}

  • Moyenne Derniers 100        : {np.mean(rewards[-100:]):>8.2f}
  • Moyenne Derniers 50         : {np.mean(rewards[-50:]):>8.2f}

🏃 EFFICACITÉ DE L'AGENT
───────────────────────────────────────────────────────────────────────────────
  • Pas Moyen par Episode       : {np.mean(steps):>8.2f}
  • Pas Maximum                 : {max(steps):>8.2f}
  • Pas Minimum                 : {min(steps):>8.2f}
  • Pas Derniers Episodes       : {np.mean(steps[-100:]):>8.2f}

📈 PROGRESSION PAR PHASE
───────────────────────────────────────────────────────────────────────────────
  Phase 1 (Exploration      0-250)  : {np.mean(rewards[0:250]):>8.2f}
  Phase 2 (Apprentissage   250-500) : {np.mean(rewards[250:500]):>8.2f}
  Phase 3 (Convergence    500-750)  : {np.mean(rewards[500:750]):>8.2f}
  Phase 4 (Stabilisation 750-1000)  : {np.mean(rewards[750:1000]):>8.2f}

  Amélioration P1➜P4                : {np.mean(rewards[750:1000]) - np.mean(rewards[0:250]):>8.2f} (+{100*(np.mean(rewards[750:1000]) - np.mean(rewards[0:250]))/abs(np.mean(rewards[0:250])):.1f}%)

🎬 EXPLORATION
───────────────────────────────────────────────────────────────────────────────
  • Epsilon Initial             : {epsilons[0]:.4f}
  • Epsilon Final               : {epsilons[-1]:.4f}
  • Réduction                   : {(1 - epsilons[-1]/epsilons[0])*100:.2f}%

✅ FICHIERS GÉNÉRÉS
───────────────────────────────────────────────────────────────────────────────
  ✓ 01_training_curve.png                  - Courbe d'entraînement
  ✓ 02_epsilon_decay.png                   - Décroissance epsilon
  ✓ 03_reward_distribution.png             - Distribution récompenses
  ✓ 04_rewards_by_phase.png                - Récompenses par phase
  ✓ 05_steps_per_episode.png               - Efficacité (pas/episode)
  ✓ 06_gridworld_visualization.png         - Visualisation GridWorld + Heatmap
  ✓ 07_comparison_dashboard.png            - Dashboard complet
  ✓ training_summary.txt                   - Résumé (ce fichier)
  ✓ my_model.keras                         - Modèle entraîné

═══════════════════════════════════════════════════════════════════════════════
                        Entraînement Complété avec Succès ✓
═══════════════════════════════════════════════════════════════════════════════
"""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ Sauvegardé : {filepath}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN : EXECUTION COMPLÈTE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PIPELINE COMPLET : Entraînement + Génération Graphiques".center(70))
    print("=" * 70 + "\n")

    # Étape 1 : Entraînement
    print("\n[1/2] ENTRAÎNEMENT EN COURS...")
    print("-" * 70)
    rewards, epsilons, steps, agent, env = train_double_dqn()

    # Étape 2 : Génération des graphiques
    print("\n[2/2] GÉNÉRATION DES GRAPHIQUES...")
    print("-" * 70)

    print("\n📊 Génération des figures...")
    plot_training_curve(rewards)
    plot_epsilon_decay(epsilons)
    plot_reward_distribution(rewards)
    plot_rewards_by_phase(rewards)
    plot_steps_per_episode(steps)
    plot_gridworld_visualization(env, agent)
    plot_comparison_metrics(rewards, steps)

    print("\n📄 Génération du résumé texte...")
    create_summary_report(rewards, epsilons, steps)

    # Affichage final
    print("\n" + "=" * 70)
    print("✅ PIPELINE TERMINÉ AVEC SUCCÈS !".center(70))
    print("=" * 70)
    print(f"\n📁 Tous les fichiers sont dans le dossier : {os.path.abspath(FIGURES_DIR)}")
    print(f"\n📊 Graphiques générés :")
    for i, f in enumerate(sorted(os.listdir(FIGURES_DIR)), 1):
        print(f"   {i}. {f}")
    print("\n" + "=" * 70)
