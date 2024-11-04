# Projet de Deep Learning : Rétropropagation et Classification Logique

### Elyes KHALFALLAH - 5230635
### 04/11/2024
---

Ce projet, réalisé dans le cadre du cours de Deep Learning, explore en profondeur la rétropropagation dans les réseaux de neurones. L’objectif est de classifier les fonctions logiques (`AND`, `OR`, `XOR`) à l'aide d'un réseau de neurones multicouche. 

## Contenu du Projet

Le rapport du projet se trouve dans les cellules Markdown de `main.ipynb`. Ces cellules contiennent toutes les explications nécessaires sur le processus de rétropropagation, les choix de fonctions d’activation (sigmoid, relu, tanh), et les détails de chaque étape.

Pour réduire la taille de `main.ipynb`, toutes les fonctions utilisées sont écrites dans `fonctions.py`, puis importées dans le notebook principal. Ceci permet de garder `main.ipynb` plus lisible et de se concentrer sur l’exécution et la visualisation des résultats.

Mon rendu complete toutes les spécifications principales présentes dans *rendu1.pdf*, et deux parmi trois des éléments additionnels proposés, étant "permettre a l’utilisateur de preciser la fonction d’activation de son choix" et "suivre l’evolution de differentes mesures, par exemple la fonction de cout". 

## Exécution du Projet

Lancer `main.ipynb` permettra de générer un répertoire `results` contenant les résultats sous forme d'images de graphes pour chaque combinaison de fonction d’activation, d'architecture de réseau, et de fonction logique.

Mon dossier `results/` a été inclus dans le gitignore, rendant donc cette étape obligatoire pour la génération des résultats. Pour rappel, de l'aléatoire (sans seed prédéfini) est inclu dans ce projet, donc un lancement "malchanceux" peut arriver, aboutissant a des résultats questionnables (mais généralement pour la fonction sigmoide les résultats sont prometteurs). 

**Temps d'exécution estimé** : Environ 4 minutes. je recommande de lire le Markdown pendant l'exécution.

## Installation et Lancement de l’Environnement Virtuel

1. **Créer l'environnement virtuel :**
   ```bash
   python -m venv env
   ```

2. **Activer l'environnement virtuel :**
    - **Sous Windows**
        ```bash
        .\env\Scripts\activate
        ```
    - **Sous macOS et Linux**
        ```bash
        source env/bin/activate
        ```

3. **Installer les dépendances :** elles ne sont pas nombreuses (numpy et matplotlib)
    ```bash
   pip install -r requirements.txt
   ```
