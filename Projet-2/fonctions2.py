# Imports utiles
import numpy as np
from sentence_transformers import SentenceTransformer

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
    # Stabilisation numérique pour éviter des débordements
    z = z - np.max(z, axis=1, keepdims=True)

    # Normalisation par la somme
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


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
    # Extraction des indices des classes prédites
    pred_labels = np.argmax(y_pred, axis=1)
    # Proportion de prédictions correctes
    return np.mean(pred_labels == y_true)


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
# Fonctions pour l'encodage textuel


def load_text_encoder(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Charge un modèle Sentence-BERT depuis Hugging Face.
    """
    # Chargement du modèle Sentence-BERT spécifié par le nom
    model = SentenceTransformer(model_name)
    return model  # Retourne le modèle chargé


def encode_titles(model, titles, batch_size=32):
    """
    Encode une liste de titres en vecteurs numériques.
    """
    # Vérifie si les titres sont une liste, sinon les convertit en liste
    if not isinstance(titles, list):
        titles = titles.tolist()
    # Utilise le modèle pour encoder les titres en lots, retourne des vecteurs numériques
    embeddings = model.encode(titles, batch_size=batch_size, convert_to_numpy=True)
    return embeddings  # Retourne les vecteurs d'embedding


# -----------------------------------------------------------
# Classe du réseau (MLP) simple pour classification sur embeddings textuels


class SimpleMLP:
    def __init__(self, input_dim, layer_sizes, num_classes):
        """
        Initialise un MLP simple sans batchnorm, sans dropout, sans gradient clipping.
        - input_dim : dimension d'entrée (dimension des embeddings textuels)
        - layer_sizes : liste contenant le nombre de neurones pour chaque couche cachée
        - num_classes : nombre de classes en sortie
        """
        self.layers = []  # Liste pour stocker les couches (poids et biais)
        prev_dim = input_dim  # Dimension d'entrée initiale

        # Initialisation des couches cachées
        for h in layer_sizes:
            # Poids initialisés avec une petite valeur aléatoire
            W = np.random.randn(prev_dim, h) * 0.01
            # Biais initialisés à zéro
            b = np.zeros((1, h))
            self.layers.append(
                (W, b)
            )  # Ajout des poids et biais à la liste des couches
            prev_dim = h  # Met à jour la dimension pour la prochaine couche

        # Initialisation de la couche de sortie
        W_out = (
            np.random.randn(prev_dim, num_classes) * 0.01
        )  # Poids de la couche de sortie
        b_out = np.zeros((1, num_classes))  # Biais de la couche de sortie
        self.layers.append((W_out, b_out))  # Ajout de la couche de sortie à la liste

    def forward(self, X):
        """
        Passe avant : calcule les activations à travers les couches du réseau.
        Retourne toutes les activations intermédiaires et la sortie finale.
        """
        activations = [X]  # Stocke les activations, commence avec l'entrée
        for i in range(len(self.layers) - 1):  # Parcourt les couches cachées
            W, b = self.layers[i]  # Récupère les poids et biais de la couche i
            Z = activations[-1].dot(W) + b  # Calcul des activations linéaires
            A = relu(Z)  # Application de l'activation ReLU
            activations.append(A)  # Ajoute les activations de la couche
        # Couche de sortie
        W_out, b_out = self.layers[-1]  # Poids et biais de la dernière couche
        Z_out = (
            activations[-1].dot(W_out) + b_out
        )  # Calcul des scores de la couche de sortie
        A_out = softmax(Z_out)  # Application de la fonction softmax
        activations.append(A_out)  # Ajoute les activations finales
        return activations  # Retourne les activations intermédiaires et finales

    def backward(self, activations, y_true):
        """
        Passe arrière : calcule les gradients des poids et biais.
        """
        grads = []  # Liste pour stocker les gradients
        # Convertit les labels en one-hot encoding
        y_true_oh = one_hot(y_true, activations[-1].shape[1])
        A_out = activations[-1]  # Activations finales
        # Calcul de l'erreur pour la couche de sortie
        delta = (A_out - y_true_oh) / len(y_true)

        # Calcul des gradients pour la couche de sortie
        W_out, b_out = self.layers[-1]
        A_prev = activations[-2]  # Activation de la dernière couche cachée
        dW_out = A_prev.T.dot(delta)  # Gradient des poids
        db_out = np.sum(delta, axis=0, keepdims=True)  # Gradient des biais
        grads.append((dW_out, db_out))  # Ajoute les gradients de la couche de sortie

        delta = delta.dot(W_out.T)  # Propagation de l'erreur aux couches cachées

        # Calcul des gradients pour les couches cachées (en sens inverse)
        for i in reversed(range(len(self.layers) - 1)):
            W, b = self.layers[i]  # Poids et biais de la couche i
            A_prev = activations[i]  # Activation avant la couche i
            Z = A_prev.dot(W) + b  # Calcul des activations linéaires
            dZ = delta * relu_deriv(Z)  # Application de la dérivée de ReLU

            dW = A_prev.T.dot(dZ)  # Gradient des poids
            db = np.sum(dZ, axis=0, keepdims=True)  # Gradient des biais
            grads.append((dW, db))  # Ajoute les gradients de la couche courante

            delta = dZ.dot(W.T)  # Propagation de l'erreur à la couche précédente

        grads.reverse()  # Inverse l'ordre des gradients pour correspondre aux couches
        return grads  # Retourne les gradients calculés

    def update_params(self, grads, lr):
        """
        Met à jour les paramètres du réseau en utilisant les gradients calculés.
        """
        for i in range(len(self.layers)):
            W, b = self.layers[i]  # Récupère les poids et biais de la couche i
            dW, db = grads[i]  # Récupère les gradients correspondants
            W -= lr * dW  # Mise à jour des poids
            b -= lr * db  # Mise à jour des biais
            self.layers[i] = (W, b)  # Sauvegarde des nouveaux poids et biais

    def predict(self, X):
        """
        Prédit les classes des données X.
        """
        A_out = self.forward(X)[
            -1
        ]  # Effectue une passe avant et récupère la sortie finale
        return np.argmax(A_out, axis=1)  # Retourne les indices des classes prédites


# -----------------------------------------------------------
# Fonction d'entraînement


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
    Entraîne le MLP sur les données textuelles (embeddings) et renvoie l'historique.
    """
    # Dictionnaire pour stocker les métriques d'entraînement et de validation
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):  # Boucle sur le nombre d'époques
        # Mélange les données d'entraînement à chaque époque
        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        # Divise les données en batches
        batches = range(0, len(X_train), batch_size)
        for start in batches:
            end = start + batch_size
            X_batch = X_train[start:end]  # Sélection des données du batch
            y_batch = y_train[start:end]  # Sélection des labels du batch

            # Passe avant et arrière pour calculer les gradients
            activations = model.forward(X_batch)
            grads = model.backward(activations, y_batch)
            model.update_params(
                grads, lr
            )  # Mise à jour des paramètres avec le taux d'apprentissage

        # Évaluation sur les données d'entraînement
        train_A = model.forward(X_train)[-1]  # Activations finales sur le train
        train_loss = cross_entropy_loss(train_A, y_train)  # Calcul de la perte
        train_acc = accuracy(train_A, y_train)  # Calcul de l'exactitude

        # Évaluation sur les données de validation
        val_A = model.forward(X_val)[-1]  # Activations finales sur la validation
        val_loss = cross_entropy_loss(val_A, y_val)  # Calcul de la perte
        val_acc = accuracy(val_A, y_val)  # Calcul de l'exactitude

        # Stocke les métriques dans l'historique
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            # Affiche les métriques pour chaque époque si verbose est activé
            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

    return history  # Retourne l'historique des pertes et des exactitudes
