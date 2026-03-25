# Double DQN sur GridWorld

Agent Double DQN entrainé sur GridWorld 4x4 avec génération automatique de figures et rapport Word.

## Structure du projet

```
Deep Q-Learning sur GridWorld/
|-- devoir_complet.py              # Code principal: entrainement + figures
|-- generate_word_report.py        # Génere rapport Word avec figures
|-- requirements.txt               # Dépendances Python
|-- README.md                       # Ce fichier
|-- my_model.keras                 # Modèle entraîné
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

## Démarrage rapide

### 1) Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2) Lancer l'entraînement
```bash
python devoir_complet.py
```

Génère :
- `my_model.keras` (modèle)
- `figures/*.png` (7 figures)
- `figures/training_summary.txt` (résumé)

### 3) Générer rapport Word avec figures
```bash
python generate_word_report.py
```

Génère : `rapport_devoir_word.docx` (contient résumé + 7 figures)

## Théorie

**Double DQN** sépare la sélection et l'évaluation des actions :

$$a^* = \arg\max_a Q_{online}(s', a)$$

$$y = r + \gamma Q_{target}(s', a^*)$$

**Avantages** :
- Réduit le biais de surestimation du DQN classique
- Entraînement plus stable et régulier
- Meilleure convergence

## Paramètres d'entraînement

- Episodes: 1000
- Max steps: 50
- Gamma: 0.9
- Learning rate: 0.01
- Epsilon: 1.0 → 0.01 (décroissance: 0.995)
- Batch: 32 | Memory: 2000
- Target network update: tous les 10 episodes

## Résultats

Les 7 figures générées montrent :
- Courbe d'entraînement (récompense croissante)
- Décroissance epsilon (exploration → exploitation)
- Distribution et efficacité de l'agent
- Heatmap des Q-values sur la grille
- Dashboard global de performance
