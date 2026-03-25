# Rapport de Devoir - Double DQN sur GridWorld

## Resume executif

Ce devoir implemente un agent **Double DQN** pour resoudre un probleme de navigation dans un **GridWorld 4x4**. L'agent doit atteindre une case objectif en minimisant le nombre d'actions et en evitant un obstacle.

L'experience est menee sur `1000` episodes avec une strategie `epsilon-greedy` et un replay buffer. Les resultats montrent une progression nette des performances : la recompense moyenne passe d'une phase initiale negative a une phase finale stabilisee et positive.

L'analyse quantitative (recompenses, pas/episode, progression par phase) et qualitative (figures et heatmap) confirme que l'agent converge vers une politique efficace.

## 1. Contexte et objectif

Ce projet présente la mise en oeuvre d'un agent d'apprentissage par renforcement de type **Double DQN (Deep Q-Network)** appliqué a un environnement **GridWorld 4x4**.

L'objectif principal est de montrer, de maniere reproductible, qu'un agent peut apprendre une politique efficace pour atteindre un but dans une grille, tout en evitant un obstacle, avec analyse complete des performances via figures.

## 2. Problematique

L'environnement est defini comme suit :

- Grille de taille `4x4`
- Position initiale de l'agent : `(0,0)`
- Position de l'objectif : `(3,3)`
- Position de l'obstacle : `(1,1)`
- Actions possibles : `haut`, `bas`, `gauche`, `droite`

Schema de recompense :

- `+10` si l'objectif est atteint
- `-5` si l'agent passe sur l'obstacle
- `-1` pour chaque deplacement standard

Cette formulation encourage des trajectoires courtes et penalise les comportements inefficaces.

## 3. Approche methodologique

### 3.1 Pourquoi Double DQN

Le DQN classique a tendance a surestimer les valeurs d'action. Le **Double DQN** corrige ce biais en separant :

- la selection de la meilleure action via le reseau `online`
- l'evaluation de cette action via le reseau `target`

Formulation utilisee :

$$a^* = \arg\max_a Q_{online}(s', a)$$

$$y = r + \gamma Q_{target}(s', a^*)$$

Cette separation stabilise l'apprentissage et ameliore la convergence.

### 3.2 Composants de l'agent

- `Replay Buffer` (memoire d'experience) pour decorreler les echantillons
- Strategie `epsilon-greedy` pour equilibrer exploration/exploitation
- Reseau `target` mis a jour periodiquement
- Reseau de neurones dense :
    - Couche 1 : 24 neurones (ReLU)
    - Couche 2 : 24 neurones (ReLU)
    - Sortie : 4 neurones (Q-values des 4 actions)

## 4. Configuration experimentale

Les hyperparametres utilises dans l'experience sont :

- Episodes : `1000`
- Pas max par episode : `50`
- `gamma` : `0.9`
- Learning rate : `0.01`
- Epsilon initial : `1.0`
- Epsilon minimum : `0.01`
- Decroissance epsilon : `0.995`
- Batch size : `32`
- Taille memoire : `2000`
- Mise a jour du target network : toutes les `10` episodes

### 4.1 Protocole experimental

- Initialisation aleatoire fixee (`seed = 42`) pour garantir la reproductibilite.
- Entrainement sur une seule configuration d'environnement (objectif et obstacle fixes).
- Evaluation de la convergence via :
    - la moyenne glissante des recompenses,
    - la moyenne des `50` et `100` derniers episodes,
    - la reduction du nombre de pas par episode.
- Les metriques sont enregistrees automatiquement dans `figures/training_summary.txt`.

## 5. Resultats quantitatifs

Resultats extraits de `figures/training_summary.txt` :

- Recompense moyenne globale : `1.31`
- Recompense maximale : `5.00`
- Recompense minimale : `-74.00`
- Ecart-type : `10.08`
- Recompense finale : `5.00`

- Moyenne des 100 derniers episodes : `4.86`
- Moyenne des 50 derniers episodes : `4.90`

- Pas moyen par episode : `8.73`
- Pas minimum : `6`
- Pas maximum : `50`
- Moyenne des pas en fin d'apprentissage : `6.10`

Progression par phases :

- Phase 1 (0-250, exploration) : `-7.88`
- Phase 2 (250-500, apprentissage) : `3.58`
- Phase 3 (500-750, convergence) : `4.69`
- Phase 4 (750-1000, stabilisation) : `4.87`

Amelioration de la phase 1 a la phase 4 : `+12.75` (soit `+161.9%`).

## 6. Analyse des figures

Les figures sont generees automatiquement dans le dossier `figures/`.

### Figure 1 - Courbe d'entrainement

![Courbe d'entrainement](figures/01_training_curve.png)

Montre l'evolution des recompenses par episode et leur moyenne mobile. On observe une tendance ascendante nette, signe d'un apprentissage effectif.

### Figure 2 - Decroissance epsilon

![Decroissance epsilon](figures/02_epsilon_decay.png)

Visualise la transition progressive de l'exploration vers l'exploitation. Epsilon passe de `1.0` a `0.01`, soit une reduction de `99%`.

### Figure 3 - Distribution des recompenses

![Distribution des recompenses](figures/03_reward_distribution.png)

Permet d'analyser la variabilite des episodes. La distribution illustre une forte amelioration globale avec diminution progressive des episodes tres negatifs.

### Figure 4 - Recompenses par phase

![Recompenses par phase](figures/04_rewards_by_phase.png)

Mise en evidence des differentes etapes d'apprentissage (exploration, apprentissage, convergence, stabilisation) et de l'amelioration continue entre phases.

### Figure 5 - Nombre de pas par episode

![Pas par episode](figures/05_steps_per_episode.png)

Mesure l'efficacite de la politique apprise. La baisse du nombre moyen de pas indique que l'agent trouve des trajets plus courts vers l'objectif.

### Figure 6 - Visualisation GridWorld et heatmap

![GridWorld et heatmap](figures/06_gridworld_visualization.png)

Représentation de l'environnement et des valeurs apprises, utile pour interpreter qualitativement la strategie de l'agent sur la grille.

### Figure 7 - Dashboard comparatif global

![Dashboard global](figures/07_comparison_dashboard.png)

Vue synthetique des metriques principales pour une lecture rapide des performances finales du systeme.

## 7. Interprétation et discussion

Les resultats montrent que l'agent Double DQN apprend une politique stable et performante :

- Les recompenses deviennent majoritairement positives en fin d'entrainement
- Le nombre de pas diminue vers une trajectoire quasi optimale
- La stabilisation sur les derniers episodes confirme la convergence

Malgre quelques episodes difficiles en debut d'apprentissage (scores tres negatifs), la dynamique globale est conforme au comportement attendu d'un agent en apprentissage par renforcement profond.

### 7.1 Lecture pedagogique des metriques

- Le score final proche de `5.0` est coherent avec la fonction de recompense :
  - atteindre l'objectif donne `+10`,
  - un trajet efficace implique quelques penalites `-1`,
  - ce qui amene une recompense nette autour de `+4` a `+5`.
- La baisse des pas/episode vers environ `6` indique que la politique apprise se rapproche d'un chemin court vers l'objectif.
- Les episodes tres negatifs en debut d'entrainement sont normaux : l'agent explore fortement et ne maitrise pas encore la dynamique de l'environnement.

## 8. Hypotheses et limites

- Etat simplifie : representation basee principalement sur la position de l'agent.
- Environnement statique : obstacle et objectif fixes, sans dynamique temporelle.
- Une seule taille de grille (`4x4`) testee.
- Resultats bases sur une seule seed principale (`42`).
- Pas de comparaison experimentale explicite avec un DQN simple dans ce rendu.

Ces limites n'invalident pas le travail, mais elles encadrent le perimetre scientifique de l'etude.

## 9. Pseudo-code de l'algorithme

```text
Initialiser GridWorld, online_model, target_model, replay_buffer
Copier les poids online -> target

Pour episode = 1 a EPISODES:
    state = reset(env)
    total_reward = 0

    Pour t = 1 a MAX_STEPS:
        Avec proba epsilon: action aleatoire
        Sinon: action = argmax_a Q_online(state, a)

        next_state, reward, done = env.step(action)
        stocker (state, action, reward, next_state, done) dans replay_buffer
        state = next_state
        total_reward += reward

        Si done: sortir de la boucle

    Echantillonner un mini-batch du replay_buffer
    Pour chaque transition:
        a* = argmax_a Q_online(next_state, a)
        target = reward + gamma * Q_target(next_state, a*) si non terminal
        target = reward sinon
    Mettre a jour online_model par descente de gradient

    Reduire epsilon
    Tous les TARGET_UPDATE_EVERY episodes: copier online -> target
```

## 10. Structure du projet

```
Deep Q-Learning sur GridWorld/
|-- devoir_complet.py
|-- my_model.keras
|-- README.md
|-- requirements.txt
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

## 11. Reproduction des experiences

### 9.1 Installation

```bash
pip install -r requirements.txt
```

### 9.2 Entrainement + generation des figures

```bash
python devoir_complet.py
```

Fichiers generes :

- `my_model.keras`
- `figures/*.png`
- `figures/training_summary.txt`

### 11.3 Checklist de reproductibilite

- Verifier que les dependances sont installees via `requirements.txt`.
- Lancer `python devoir_complet.py` sans erreur.
- Verifier la presence des fichiers :
    - `my_model.keras`
    - `figures/01_training_curve.png`
    - `figures/02_epsilon_decay.png`
    - `figures/03_reward_distribution.png`
    - `figures/04_rewards_by_phase.png`
    - `figures/05_steps_per_episode.png`
    - `figures/06_gridworld_visualization.png`
    - `figures/07_comparison_dashboard.png`
    - `figures/training_summary.txt`
- Verifier dans le resume que :
    - la moyenne des derniers episodes est positive,
    - epsilon final est proche de `0.01`,
    - les pas moyens finaux sont inferieurs aux pas initiaux.

## 12. Perspectives

- Comparer Double DQN avec DQN standard sur le meme protocole.
- Tester plusieurs seeds et rapporter moyenne et ecart-type inter-runs.
- Augmenter la complexite de l'environnement (grille plus grande, obstacles multiples).
- Introduire des objectifs variables pour evaluer la robustesse de la politique.

## 13. Conclusion

Ce travail valide l'utilisation de **Double DQN** sur un probleme de navigation discret de type GridWorld. L'agent converge vers une politique efficace, avec des gains nets entre le debut et la fin de l'entrainement.

Les figures et metriques produites constituent une base solide pour un rendu de devoir, avec une lecture a la fois quantitative (scores, pas, variabilite) et qualitative (visualisation de la strategie).
