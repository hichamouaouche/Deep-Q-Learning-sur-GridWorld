# Rapport de devoir - Double DQN sur GridWorld

## 1. Informations generales
- Module: Apprentissage par renforcement
- Sujet: Implementation et analyse de Double DQN
- Environnement: GridWorld 4x4
- Fichier principal: devoir_complet.py

## 2. Objectif du devoir
L objectif est de comparer un apprentissage DQN standard avec Double DQN, puis de montrer visuellement le comportement de l apprentissage avec des figures exportees automatiquement.

## 3. Structure du projet

```text
devoir/
|-- devoir.py
|-- train_and_visualize.py
|-- devoir_complet.py
|-- my_model.keras
|-- partie2_double_dqn.md
|-- README.md
`-- figures/
    |-- 01_training_curve.png
    |-- 02_epsilon_decay.png
    |-- 03_reward_distribution.png
    |-- 04_rewards_by_phase.png
    |-- 05_steps_per_episode.png
    |-- 06_gridworld_visualization.png
    |-- 07_comparison_dashboard.png
    `-- training_summary.txt
```

## 4. Rappel theorique
Dans DQN classique, la cible est:

$$y = r + \gamma \max_{a} Q(s', a)$$

Le meme reseau sert a choisir et evaluer l action, ce qui peut surestimer les valeurs Q.

Dans Double DQN:

$$a^* = \arg\max_a Q_{online}(s', a)$$

$$y = r + \gamma Q_{target}(s', a^*)$$

On separe la selection (reseau online) et l evaluation (reseau target).

## 5. Parametres d entrainement
- Episodes: 1000
- Max steps par episode: 50
- Gamma: 0.9
- Learning rate: 0.01
- Epsilon initial: 1.0
- Epsilon min: 0.01
- Epsilon decay: 0.995
- Batch size: 32
- Memory size: 2000
- Target update: tous les 10 episodes

## 6. Code Python (extraits importants)

### 6.1 Creation de l agent Double DQN
```python
class DoubleDQNAgent:
    def __init__(self):
        self.online_model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
```

### 6.2 Calcul de la cible Double DQN
```python
best_next_action = int(np.argmax(next_q_online[i]))
targets[i, actions[i]] = rewards[i] + GAMMA * next_q_target[i, best_next_action]
```

### 6.3 Lancement complet
```python
if __name__ == "__main__":
    rewards, epsilons, steps, agent, env = train_double_dqn()
    plot_training_curve(rewards)
    plot_epsilon_decay(epsilons)
    plot_reward_distribution(rewards)
    plot_rewards_by_phase(rewards)
    plot_steps_per_episode(steps)
    plot_gridworld_visualization(env, agent)
    plot_comparison_metrics(rewards, steps)
```

Le code complet est dans `devoir_complet.py`.

## 7. Figures du devoir

### Figure 1 - Courbe d entrainement
![Training Curve](figures/01_training_curve.png)

### Figure 2 - Decroissance epsilon
![Epsilon Decay](figures/02_epsilon_decay.png)

### Figure 3 - Distribution des recompenses
![Reward Distribution](figures/03_reward_distribution.png)

### Figure 4 - Recompenses par phase
![Rewards by Phase](figures/04_rewards_by_phase.png)

### Figure 5 - Nombre de pas par episode
![Steps per Episode](figures/05_steps_per_episode.png)

### Figure 6 - GridWorld et heatmap Q-values
![GridWorld Visualization](figures/06_gridworld_visualization.png)

### Figure 7 - Dashboard global
![Comparison Dashboard](figures/07_comparison_dashboard.png)

## 8. Resultats observes
- La moyenne des recompenses augmente au fil des episodes.
- Epsilon diminue progressivement vers epsilon_min.
- Le nombre de pas moyen baisse avec la convergence.
- La politique devient plus stable en fin d entrainement.

## 9. Procedure d execution

Depuis le dossier devoir:

```bash
python devoir_complet.py
```

Sorties generees:
- my_model.keras
- 7 figures dans le dossier figures
- training_summary.txt

## 10. Conclusion
Double DQN corrige le biais de surestimation du DQN classique en separant selection et evaluation. Dans ce devoir, l approche donne une convergence plus stable et de meilleurs indicateurs visuels sur GridWorld.

---

## Annexes
- Code principal complet: devoir_complet.py
- Variante entrainement + figures: train_and_visualize.py
- Version Double DQN simple: devoir.py
