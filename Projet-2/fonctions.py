import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Définition des fonctions utiles


def one_hot(y, num_classes):
    """
    Transforme un vecteur d'indices en une matrice one-hot.
    Chaque élément y[i] est transformé en un vecteur de taille num_classes avec un 1 à la position y[i].

    Arguments :
    - y : tableau d'indices (ex. [0, 2, 1])
    - num_classes : nombre total de classes

    Retourne :
    - oh : matrice one-hot (shape : [len(y), num_classes])
    """
    oh = np.zeros((len(y), num_classes))  # Initialisation d'une matrice nulle
    oh[np.arange(len(y)), y] = 1.0  # Mise à 1 des positions correspondant aux indices
    return oh


def softmax(z):
    """
    Calcule la fonction softmax, utilisée pour normaliser les sorties d'un modèle en probabilités.

    Arguments :
    - z : matrice des scores (shape : [batch_size, num_classes])

    Retourne :
    - Matrice normalisée où chaque ligne est une distribution de probabilité.
    """
    z = z - np.max(
        z, axis=1, keepdims=True
    )  # Stabilisation numérique pour éviter des débordements
    return np.exp(z) / np.sum(
        np.exp(z), axis=1, keepdims=True
    )  # Normalisation par la somme


def cross_entropy_loss(y_pred, y_true):
    """
    Calcule l'erreur de cross-entropy entre les prédictions et les vraies étiquettes.

    Arguments :
    - y_pred : prédictions du modèle (shape : [batch_size, num_classes], probabilités softmax)
    - y_true : vecteur d'indices des vraies classes (ex. [0, 2, 1])

    Retourne :
    - Moyenne de la perte de cross-entropy pour toutes les données.
    """
    y_true_oh = one_hot(y_true, y_pred.shape[1])  # Conversion des étiquettes en one-hot
    eps = 1e-9  # Petite constante pour éviter log(0)
    return -np.mean(np.sum(y_true_oh * np.log(y_pred + eps), axis=1))  # Perte moyenne


def accuracy(y_pred, y_true):
    """
    Calcule la précision en comparant les prédictions aux étiquettes vraies.

    Arguments :
    - y_pred : prédictions du modèle (shape : [batch_size, num_classes], probabilités softmax)
    - y_true : vecteur d'indices des vraies classes (ex. [0, 2, 1])

    Retourne :
    - Précision moyenne entre 0 et 1.
    """
    pred_labels = np.argmax(
        y_pred, axis=1
    )  # Extraction des indices des classes prédites
    return np.mean(pred_labels == y_true)  # Proportion de prédictions correctes


def relu(x):
    """
    Applique la fonction d'activation ReLU (Rectified Linear Unit).

    Arguments :
    - x : matrice des scores (shape : quelconque)

    Retourne :
    - Matrice où les valeurs négatives sont remplacées par 0.
    """
    return np.maximum(0, x)  # Remplace les valeurs négatives par 0


def relu_deriv(x):
    """
    Calcule la dérivée de la fonction ReLU pour le backpropagation.

    Arguments :
    - x : matrice des scores (shape : quelconque)

    Retourne :
    - Matrice binaire indiquant où ReLU est actif (1 pour x > 0, sinon 0).
    """
    return (x > 0).astype(float)  # Retourne 1 si x > 0, sinon 0


# -----------------------------------------------------------
# Classe du réseau de neurones


class MLP:
    def __init__(self, input_dim, layer_sizes, num_classes):
        """
        Initialise les poids et les biais pour chaque couche du réseau.

        Arguments :
        - input_dim : dimension d'entrée des données
        - layer_sizes : liste contenant le nombre de neurones pour chaque couche cachée
        - num_classes : nombre de classes (dimension de sortie)
        """
        self.layers = []  # Liste pour stocker les poids et biais de chaque couche
        prev_dim = input_dim  # Dimension de la couche précédente

        # Initialisation des couches cachées
        for h in layer_sizes:
            W = (
                np.random.randn(prev_dim, h) * 0.01
            )  # Poids initialisés avec une distribution normale
            b = np.zeros((1, h))  # Biais initialisés à 0
            self.layers.append((W, b))  # Ajout de la couche à la liste
            prev_dim = h  # Mise à jour pour la prochaine couche

        # Initialisation de la couche de sortie
        W_out = np.random.randn(prev_dim, num_classes) * 0.01  # Poids pour la sortie
        b_out = np.zeros((1, num_classes))  # Biais pour la sortie
        self.layers.append((W_out, b_out))  # Ajout de la couche de sortie

    def forward(self, X):
        """
        Effectue une passe avant (forward pass) dans le réseau.

        Arguments :
        - X : données d'entrée (shape : [batch_size, input_dim])

        Retourne :
        - activations : liste contenant les activations de chaque couche, y compris la sortie finale.
        """
        activations = [X]  # Stocke les activations, en commençant par l'entrée

        # Passe à travers les couches cachées
        for i in range(len(self.layers) - 1):
            W, b = self.layers[i]
            Z = activations[-1].dot(W) + b  # Calcul des scores linéaires
            A = relu(Z)  # Application de la fonction d'activation ReLU
            activations.append(A)  # Stocke les activations

        # Calcul pour la couche de sortie
        W_out, b_out = self.layers[-1]
        Z_out = activations[-1].dot(W_out) + b_out  # Scores linéaires pour la sortie
        A_out = softmax(Z_out)  # Application de softmax pour obtenir les probabilités
        activations.append(A_out)  # Stocke la sortie finale
        return activations

    def backward(self, activations, y_true):
        """
        Effectue une passe arrière (backpropagation) pour calculer les gradients.

        Arguments :
        - activations : liste des activations de la passe avant
        - y_true : vecteur d'étiquettes vraies

        Retourne :
        - grads : liste des gradients pour les poids et les biais de chaque couche.
        """
        grads = []  # Liste pour stocker les gradients
        y_true_oh = one_hot(
            y_true, activations[-1].shape[1]
        )  # Convertit y_true en one-hot
        A_out = activations[-1]  # Sortie du modèle
        delta = (A_out - y_true_oh) / len(y_true)  # Erreur pour la couche de sortie

        # Backpropagation à travers les couches
        for i in reversed(range(len(self.layers))):
            W, b = self.layers[i]
            A_prev = activations[i]  # Activation de la couche précédente
            dW = A_prev.T.dot(delta)  # Gradient des poids
            db = np.sum(delta, axis=0, keepdims=True)  # Gradient des biais
            grads.append((dW, db))  # Ajout des gradients à la liste
            if i > 0:  # Propagation de l'erreur si ce n'est pas la première couche
                delta = delta.dot(W.T) * relu_deriv(
                    A_prev
                )  # Calcul de delta pour la couche précédente

        grads.reverse()  # Inverser l'ordre des gradients pour correspondre aux couches
        return grads

    def update_params(self, grads, lr):
        """
        Met à jour les poids et les biais en utilisant les gradients calculés.

        Arguments :
        - grads : liste des gradients pour chaque couche
        - lr : taux d'apprentissage (learning rate)
        """
        new_layers = []  # Liste pour stocker les nouveaux paramètres
        for i, (W, b) in enumerate(self.layers):
            dW, db = grads[i]  # Gradients des poids et des biais
            W -= lr * dW  # Mise à jour des poids
            b -= lr * db  # Mise à jour des biais
            new_layers.append((W, b))  # Ajout des paramètres mis à jour
        self.layers = new_layers  # Mise à jour des couches du réseau

    def predict(self, X):
        """
        Prédit les classes pour les données d'entrée.

        Arguments :
        - X : données d'entrée (shape : [batch_size, input_dim])

        Retourne :
        - Prédictions des classes sous forme d'indices (shape : [batch_size,])
        """
        A = self.forward(X)[-1]  # Sortie finale après la passe avant
        return np.argmax(
            A, axis=1
        )  # Retourne les indices des classes avec la plus haute probabilité


# -----------------------------------------------------------
# Fonction d'entraînement et d'évaluation


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=20,
    batch_size=32,
    lr=0.001,
    verbose=True,
):
    """
    Entraîne un modèle MLP sur un ensemble d'entraînement et évalue sa performance sur un ensemble de validation.

    Arguments :
    - model : instance du modèle MLP à entraîner
    - X_train : données d'entraînement (features)
    - y_train : étiquettes d'entraînement
    - X_val : données de validation (features)
    - y_val : étiquettes de validation
    - epochs : nombre d'époques d'entraînement
    - batch_size : taille des mini-lots (mini-batches)
    - lr : taux d'apprentissage
    - verbose : affiche les métriques à chaque époque si True

    Retourne :
    - history : dictionnaire contenant les pertes et précisions pour l'entraînement et la validation à chaque époque.
    """
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }  # Historique des métriques

    for epoch in range(epochs):  # Boucle principale sur les époques
        # Mélange des données pour garantir des mini-lots aléatoires à chaque époque
        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        # Découpe les données en mini-lots
        batches = range(0, len(X_train), batch_size)
        for start in batches:
            end = start + batch_size
            X_batch = X_train[start:end]  # Extraction des données du mini-lot
            y_batch = y_train[start:end]  # Extraction des étiquettes du mini-lot

            # Passe avant, rétropropagation, et mise à jour des paramètres
            activations = model.forward(X_batch)  # Calcul des activations
            grads = model.backward(activations, y_batch)  # Calcul des gradients
            model.update_params(grads, lr)  # Mise à jour des poids et biais

        # Évaluation sur l'ensemble d'entraînement
        train_activations = model.forward(X_train)[
            -1
        ]  # Activations finales sur l'ensemble d'entraînement
        train_loss = cross_entropy_loss(
            train_activations, y_train
        )  # Calcul de la perte
        train_acc = accuracy(train_activations, y_train)  # Calcul de la précision

        # Évaluation sur l'ensemble de validation
        val_activations = model.forward(X_val)[
            -1
        ]  # Activations finales sur l'ensemble de validation
        val_loss = cross_entropy_loss(val_activations, y_val)  # Calcul de la perte
        val_acc = accuracy(val_activations, y_val)  # Calcul de la précision

        # Stockage des métriques dans l'historique
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Affichage des métriques si verbose est activé
        if verbose:
            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

    return history  # Retourne l'historique des métriques
