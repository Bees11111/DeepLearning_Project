"""
Elyes KHALFALLAH - 5230635

04 / 11 / 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# -----------------------------------------------------------
# Création des fonctions d'activation et leurs dérivées


def sigmoid(z):
    """Fonction d'activation sigmoïde."""
    return 1 / (1 + np.exp(-z))


def relu(z):
    """Fonction d'activation ReLU."""
    return np.maximum(0, z)


def tanh(z):
    """Fonction d'activation tanh."""
    return np.tanh(z)


def sigmoid_derivative(z):
    """Dérivée de la fonction sigmoïde."""
    s = sigmoid(z)
    return s * (1 - s)


def relu_derivative(z):
    """Dérivée de la fonction ReLU."""
    return (z > 0).astype(float)


def tanh_derivative(z):
    """Dérivée de la fonction tanh."""
    return 1 - np.tanh(z) ** 2


# -----------------------------------------------------------


def initialize_parameters(layer_dims):
    """
    Initialise les paramètres (poids et biais) du réseau de neurones en utilisant l'initialisation Xavier.

    Arguments:
    layer_dims -- Liste contenant le nombre de neurones pour chaque couche (y compris l'entrée et la sortie).

    Retourne:
    parameters -- Dictionnaire contenant les poids et les biais pour chaque couche du réseau.
    """
    parameters = {}
    # np.random.seed(1)  # Graine pour la reproductibilité (décommenter si nécessaire)

    # Boucle à travers chaque couche pour initialiser les poids et les biais
    for l in range(1, len(layer_dims)):
        # Initialisation des poids W pour la couche l avec l'initialisation Xavier
        parameters["W" + str(l)] = np.random.randn(
            layer_dims[l], layer_dims[l - 1]
        ) * np.sqrt(1 / layer_dims[l - 1])

        # Initialisation des biais b pour la couche l avec des zéros
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters  # Renvoie le dictionnaire de paramètres


def forward_propagation(X, parameters, activation_fn):
    """
    Effectue la propagation avant à travers le réseau de neurones.

    Arguments:
    X -- Données d'entrée, de dimension (nombre de features, nombre d'exemples).
    parameters -- Dictionnaire contenant les paramètres (poids et biais) du réseau.
    activation_fn -- Nom de la fonction d'activation à utiliser ('sigmoid', 'relu', 'tanh').

    Retourne:
    activations -- Dictionnaire contenant les activations de chaque couche.
    caches -- Dictionnaire contenant les valeurs intermédiaires "z" de chaque couche.
    """
    # Initialisation du dictionnaire d'activations avec l'entrée X
    activations = {"A0": X}
    # Initialisation du dictionnaire des valeurs intermédiaires "z" pour chaque couche
    caches = {}

    # Sélection de la fonction d'activation en fonction de l'argument activation_fn
    activation_func = {"sigmoid": sigmoid, "relu": relu, "tanh": tanh}[activation_fn]
    L = len(parameters) // 2  # Calcul du nombre de couches dans le réseau

    # Propagation à travers chaque couche du réseau
    for l in range(1, L + 1):
        # Extraction des poids W et des biais b pour la couche l
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        # Récupération de l'activation de la couche précédente
        A_prev = activations["A" + str(l - 1)]

        # Calcul du pré-activation z pour la couche actuelle
        z = np.dot(W, A_prev) + b
        # Application de la fonction d'activation sur z pour obtenir A
        A = activation_func(z)

        # Stockage de z pour la rétropropagation
        caches["z" + str(l)] = z
        # Stockage de l'activation pour la couche actuelle
        activations["A" + str(l)] = A

    # Renvoie les activations et les valeurs intermédiaires z pour toutes les couches
    return activations, caches


def compute_cost(Y_hat, Y):
    """
    Calcule la fonction de coût (erreur des moindres carrés).

    Arguments:
    Y_hat -- Sortie prédite du réseau, de dimension (1, nombre d'exemples).
    Y -- Valeurs réelles, de dimension (1, nombre d'exemples).

    Retourne:
    cost -- Valeur scalaire de la fonction de coût.
    """
    m = Y.shape[1]  # Nombre d'exemples
    cost = (1 / (2 * m)) * np.sum((Y_hat - Y) ** 2)
    return cost


def backward_propagation(activations, caches, Y, parameters, activation_fn):
    """
    Effectue la rétropropagation pour calculer les gradients des poids et biais.

    Arguments:
    activations -- Dictionnaire contenant les activations de chaque couche.
    caches -- Dictionnaire contenant les valeurs intermédiaires "z" de chaque couche.
    Y -- Valeurs réelles, de dimension (1, nombre d'exemples).
    parameters -- Dictionnaire contenant les paramètres (poids et biais) du réseau.
    activation_fn -- Nom de la fonction d'activation utilisée ('sigmoid', 'relu', 'tanh').

    Retourne:
    grads -- Dictionnaire contenant les gradients des poids et biais pour chaque couche.
    """
    grads = {}
    L = len(parameters) // 2  # Nombre total de couches dans le réseau
    m = Y.shape[1]  # Nombre d'exemples

    # Sélection de la fonction de dérivée de l'activation en fonction de activation_fn
    activation_deriv = {
        "sigmoid": sigmoid_derivative,
        "relu": relu_derivative,
        "tanh": tanh_derivative,
    }[activation_fn]

    # Initialisation du gradient d'erreur pour la couche de sortie
    dA = (
        activations["A" + str(L)] - Y
    )  # Dérivée du coût par rapport à l'activation de la dernière couche

    # Boucle à travers les couches en ordre inverse pour la rétropropagation
    for l in reversed(range(1, L + 1)):
        A_prev = activations["A" + str(l - 1)]  # Activation de la couche précédente
        W = parameters["W" + str(l)]  # Poids de la couche actuelle
        z = caches["z" + str(l)]  # Valeur intermédiaire z de la couche actuelle

        # Calcul du gradient de z en utilisant la dérivée de la fonction d'activation
        dz = dA * activation_deriv(z)

        # Calcul des gradients des poids et biais pour la couche actuelle
        grads["dW" + str(l)] = (1 / m) * np.dot(dz, A_prev.T)
        grads["db" + str(l)] = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        # Mise à jour du gradient d'erreur pour la couche précédente
        if l > 1:  # Ne pas calculer pour la couche d'entrée
            dA = np.dot(W.T, dz)

    return grads  # Retourne les gradients calculés pour chaque couche


def update_parameters(parameters, grads, learning_rate):
    """
    Met à jour les paramètres (poids et biais) en utilisant les gradients calculés par rétropropagation.

    Arguments:
    parameters -- Dictionnaire contenant les poids et les biais du réseau.
    grads -- Dictionnaire contenant les gradients des poids et biais pour chaque couche.
    learning_rate -- Taux d'apprentissage pour la mise à jour des paramètres.

    Retourne:
    parameters -- Dictionnaire contenant les paramètres mis à jour.
    """
    L = len(parameters) // 2
    for l in range(1, L + 1):
        # Mise à jour des poids et biais
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    return parameters


def train_model(
    X,
    Y,
    layer_dims,
    learning_rate=0.01,
    threshold=0.01,
    max_iterations=10000,
    activation_fn="sigmoid",
):
    """
    Entraîne le réseau de neurones jusqu'à ce que le coût soit inférieur au seuil défini.

    Arguments:
    X -- Données d'entrée, de dimension (nombre de features, nombre d'exemples).
    Y -- Valeurs réelles, de dimension (1, nombre d'exemples).
    layer_dims -- Liste contenant le nombre de neurones pour chaque couche.
    learning_rate -- Taux d'apprentissage pour la mise à jour des paramètres.
    threshold -- Seuil de coût en dessous duquel l'entraînement s'arrête.
    max_iterations -- Nombre maximal d'itérations pour éviter les boucles infinies.
    activation_fn -- Nom de la fonction d'activation à utiliser ('sigmoid', 'relu', 'tanh').

    Retourne:
    parameters -- Dictionnaire contenant les paramètres (poids et biais) du réseau après entraînement.
    costs -- Liste des coûts enregistrés à chaque itération.
    """
    # Initialisation des paramètres du réseau en utilisant la fonction initialize_parameters
    parameters = initialize_parameters(layer_dims)
    costs = []  # Liste pour stocker le coût à chaque itération

    # Boucle d'entraînement pour un maximum de max_iterations
    for i in range(max_iterations):
        # Étape de propagation avant
        activations, caches = forward_propagation(X, parameters, activation_fn)

        # Calcul du coût pour l'itération actuelle
        cost = compute_cost(activations["A" + str(len(layer_dims) - 1)], Y)
        costs.append(cost)  # Enregistrement du coût pour analyse de convergence

        # Critère d'arrêt : si le coût est en-dessous du seuil spécifié, arrêt de l'entraînement
        if cost < threshold:
            print(
                f"Entraînement terminé après {i + 1} itérations, coût final : {cost}\n"
            )
            break

        # Étape de rétropropagation pour calculer les gradients
        grads = backward_propagation(activations, caches, Y, parameters, activation_fn)

        # Mise à jour des paramètres en utilisant les gradients calculés
        parameters = update_parameters(parameters, grads, learning_rate)
    else:
        # Message affiché si l'entraînement atteint le nombre maximal d'itérations sans atteindre le seuil
        print(
            f"Nombre max d'itérations atteint ({max_iterations}). Coût final : {cost}\n"
        )

    # Retourne les paramètres finaux du réseau ainsi que la liste des coûts
    return parameters, costs


# Fonction de prédiction pour tracer les frontières de décision
def predict(X, parameters, activation_fn="sigmoid"):
    """
    Effectue une prédiction binaire pour chaque exemple dans X en utilisant les paramètres appris
    et une fonction d'activation spécifique.

    Arguments:
    X -- Données d'entrée de dimension (nombre de features, nombre d'exemples).
    parameters -- Dictionnaire contenant les poids et les biais du réseau appris.
    activation_fn -- Nom de la fonction d'activation à utiliser ('sigmoid', 'relu', 'tanh').

    Retourne:
    predictions -- Prédictions binaires (0 ou 1) pour chaque exemple dans X.
    """
    # Effectue la propagation avant pour obtenir les activations finales
    activations, _ = forward_propagation(X, parameters, activation_fn)

    # Récupération de l'activation de la dernière couche (couche de sortie)
    Y_hat = activations["A" + str(len(parameters) // 2)]

    # Conversion des activations en prédictions binaires : 1 si Y_hat > 0.5, sinon 0
    predictions = (Y_hat > 0.5).astype(int)

    return predictions  # Retourne les prédictions sous forme de tableau binaire


def plot_cost_and_decision_boundary(X, Y, parameters, costs, title, filename):
    """
    Trace l'évolution du coût et les frontières de décision.

    Arguments:
    X -- Données d'entrée de dimension (nombre de features, nombre d'exemples).
    Y -- Sorties réelles de dimension (1, nombre d'exemples), utilisées pour la couleur des points.
    parameters -- Dictionnaire contenant les poids et les biais du réseau appris.
    costs -- Liste des coûts enregistrés à chaque itération.
    title -- Titre du graphique pour indiquer la fonction logique illustrée.
    filename -- Nom de fichier pour sauvegarder l'image.
    """
    # Crée une figure avec deux sous-parties côte à côte
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Tracé de l'évolution du coût sur le premier graphique (ax1)
    ax1.plot(costs)
    ax1.set_title("Évolution du coût")  # Titre pour l'évolution du coût
    ax1.set_xlabel("Itération")  # Légende de l'axe des x
    ax1.set_ylabel("Coût")  # Légende de l'axe des y
    ax1.grid(True)  # Affiche une grille pour faciliter la lecture

    # Définition des limites pour les frontières de décision
    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5

    # Création d'une grille de points pour évaluer les prédictions sur l'ensemble du plan
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid_points = np.c_[xx.ravel(), yy.ravel()].T  # Conversion en tableau de points
    predictions = predict(grid_points, parameters, activation_fn="sigmoid").reshape(
        xx.shape
    )

    # Tracé des frontières de décision sur le second graphique (ax2)
    ax2.contourf(
        xx, yy, predictions, cmap=ListedColormap(["#FFAAAA", "#AAAAFF"]), alpha=0.8
    )

    # Superpose les points d'origine pour montrer les vraies classes
    ax2.scatter(
        X[0, :],
        X[1, :],
        c=Y.ravel(),
        edgecolors="k",
        marker="o",
        cmap=ListedColormap(["red", "blue"]),
    )
    ax2.set_title("Frontière de décision")  # Titre pour les frontières de décision
    ax2.set_xlabel("Feature 1")  # Légende de l'axe des x
    ax2.set_ylabel("Feature 2")  # Légende de l'axe des y

    # Titre global pour la figure et sauvegarde de l'image
    fig.suptitle(title)
    plt.savefig(filename)
    plt.close()  # Ferme la figure pour libérer de la mémoire
