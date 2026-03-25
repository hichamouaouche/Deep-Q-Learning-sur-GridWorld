# Partie 2 : Implémentation de l'agent Double DQN

## 1. Limites du DQN classique

Dans un DQN classique, la valeur cible utilise la sortie du même réseau qui est en cours d'entraînement. Autrement dit, le réseau sert à la fois à choisir la meilleure action et à évaluer cette action. Ce mécanisme peut introduire un biais de surestimation des valeurs Q, car les erreurs de prédiction se renforcent entre elles.

Ainsi, l'apprentissage peut devenir instable et la convergence plus lente, en particulier lorsque l'environnement est bruité ou lorsque les récompenses sont rares.

## 2. Principe du Double DQN

Le Double DQN corrige ce problème en séparant la sélection et l'évaluation de l'action :

- Le réseau principal (online network) choisit l'action optimale dans l'état suivant.
- Le réseau cible (target network) évalue la valeur de cette action.

Cette séparation permet de réduire la surestimation et d'améliorer la stabilité de l'entraînement.

La sélection de l'action se fait avec le réseau online :

a* = argmax_a Q_online(s', a)

La cible est ensuite calculée avec le réseau target :

y = r + gamma * Q_target(s', a*)

Si l'épisode est terminé, la cible devient simplement :

y = r

## 3. Application au même exemple GridWorld

Dans ce devoir, nous conservons exactement le même environnement GridWorld (grille 4x4, position de départ, objectif et obstacle). La différence se situe uniquement dans l'algorithme d'apprentissage de l'agent.

Les modifications apportées sont les suivantes :

1. Ajout d'un deuxième réseau de neurones : target_model.
2. Initialisation du target_model avec les poids du réseau principal.
3. Pendant replay :
   - Le réseau principal choisit la meilleure action dans next_state.
   - Le réseau cible évalue la valeur Q de cette action.
   - La cible est calculée puis utilisée pour entraîner le réseau principal.
4. Mise à jour périodique du réseau cible en copiant les poids du réseau principal (par exemple tous les N épisodes).

Cette méthode permet d'obtenir des cibles d'entraînement plus fiables que dans le DQN standard.

## 4. Avantages observés

Par rapport au DQN classique, le Double DQN présente plusieurs bénéfices :

- Réduction du biais de surestimation des Q-values.
- Entraînement plus stable.
- Convergence plus régulière.
- Politique finale souvent plus robuste sur le même environnement.

## 5. Conclusion

Le Double DQN constitue une amélioration directe et efficace du DQN. En dissociant la sélection de l'action (réseau online) et son évaluation (réseau target), il limite la propagation des erreurs de prédiction.

Dans notre exemple GridWorld, cette approche améliore la qualité de l'apprentissage et donne un agent plus fiable. Elle représente donc une étape naturelle après l'implémentation du DQN classique.
