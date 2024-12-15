import numpy as np

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
# Classe du réseau de neurones avec BatchNorm, Dropout, Gradient Clipping


class MLP:
    def __init__(
        self,
        input_dim,
        layer_sizes,
        num_classes,
        dropout_rate=0.0,
        use_batchnorm=False,
        clip_norm=None,
    ):
        """
        Initialise un réseau MLP avec options pour dropout, BatchNorm et gradient clipping.
        - input_dim : dimension des entrées.
        - layer_sizes : liste avec le nombre de neurones pour chaque couche cachée.
        - num_classes : nombre de classes en sortie.
        - dropout_rate : taux de dropout (par défaut 0.0, donc désactivé).
        - use_batchnorm : active la Batch Normalization si True.
        - clip_norm : applique gradient clipping si défini.
        """
        self.use_batchnorm = use_batchnorm  # Batch Normalization activée/désactivée
        self.dropout_rate = dropout_rate  # Taux de dropout à appliquer
        self.clip_norm = (
            clip_norm  # Valeur limite pour gradient clipping (None = pas utilisé)
        )

        self.layers = []  # Liste pour stocker les poids et biais des couches
        self.bn_params = []  # Paramètres gamma et beta pour BatchNorm
        self.bn_running_mean = []  # Moyenne en cours pour BatchNorm
        self.bn_running_var = []  # Variance en cours pour BatchNorm

        prev_dim = input_dim  # Dimension de la couche précédente (commence par la dimension d'entrée)
        for (
            h
        ) in layer_sizes:  # Boucle sur chaque couche cachée définie dans layer_sizes
            W = (
                np.random.randn(prev_dim, h) * 0.01
            )  # Initialisation aléatoire des poids
            b = np.zeros((1, h))  # Initialisation des biais à zéro
            self.layers.append(
                (W, b)
            )  # Ajout des paramètres de la couche aux couches du réseau

            if self.use_batchnorm:
                # Initialisation des paramètres pour BatchNorm
                gamma = np.ones((1, h))  # Facteur d'échelle
                beta = np.zeros((1, h))  # Décalage
                self.bn_params.append((gamma, beta))
                # Moyenne et variance initialisées
                self.bn_running_mean.append(np.zeros((1, h)))
                self.bn_running_var.append(np.ones((1, h)))
            else:
                # Si BatchNorm désactivée, on ajoute des valeurs vides
                self.bn_params.append(None)
                self.bn_running_mean.append(None)
                self.bn_running_var.append(None)

            prev_dim = h  # Mise à jour de la dimension pour la prochaine couche

        # Couche de sortie
        W_out = (
            np.random.randn(prev_dim, num_classes) * 0.01
        )  # Poids de la couche finale
        b_out = np.zeros((1, num_classes))  # Biais de la couche finale
        self.layers.append((W_out, b_out))  # Ajout de la couche de sortie

        # Pas de BatchNorm pour la couche de sortie (ajout de valeurs None)
        self.bn_params.append(None)
        self.bn_running_mean.append(None)
        self.bn_running_var.append(None)

    def _batchnorm_forward(
        self, Z, gamma, beta, running_mean, running_var, momentum=0.9, training=True
    ):
        """
        Passe avant pour Batch Normalization.
        En mode entraînement, calcule la moyenne et la variance du batch et met à jour les moyennes/variances cumulées.
        En mode test, utilise les moyennes/variances cumulées.
        """
        if training:
            batch_mean = np.mean(Z, axis=0, keepdims=True)  # Moyenne du batch
            batch_var = np.var(Z, axis=0, keepdims=True)  # Variance du batch

            # Mise à jour des moyennes et variances cumulées
            running_mean = momentum * running_mean + (1 - momentum) * batch_mean
            running_var = momentum * running_var + (1 - momentum) * batch_var

            Z_norm = (Z - batch_mean) / np.sqrt(batch_var + 1e-8)  # Normalisation
        else:
            Z_norm = (Z - running_mean) / np.sqrt(running_var + 1e-8)

        # Transformation linéaire avec gamma et beta
        out = gamma * Z_norm + beta
        cache = (
            Z,
            Z_norm,
            gamma,
            beta,
            running_mean,
            running_var,
        )  # Cache pour backward
        return out, cache, running_mean, running_var

    def _batchnorm_backward(self, dout, cache):
        """
        Passe arrière pour Batch Normalization.
        Calcule les gradients pour gamma, beta, et l'entrée Z.
        """
        Z, Z_norm, gamma, beta, running_mean, running_var = cache
        N = Z.shape[0]  # Nombre d'échantillons dans le batch

        dgamma = np.sum(dout * Z_norm, axis=0, keepdims=True)  # Gradient de gamma
        dbeta = np.sum(dout, axis=0, keepdims=True)  # Gradient de beta

        batch_var = np.var(Z, axis=0, keepdims=True)
        batch_mean = np.mean(Z, axis=0, keepdims=True)

        # Gradients par rapport à Z
        dZ_norm = dout * gamma
        dvar = np.sum(
            dZ_norm * (Z - batch_mean) * (-0.5) * (batch_var + 1e-8) ** (-3 / 2),
            axis=0,
            keepdims=True,
        )
        dmean = np.sum(
            dZ_norm * (-1 / np.sqrt(batch_var + 1e-8)), axis=0, keepdims=True
        ) + dvar * np.mean(-2 * (Z - batch_mean), axis=0, keepdims=True)
        dZ = (
            (dZ_norm / np.sqrt(batch_var + 1e-8))
            + (dvar * 2 * (Z - batch_mean) / N)
            + (dmean / N)
        )

        return dZ, dgamma, dbeta

    def _dropout_forward(self, A, training=True):
        """
        Passe avant pour Dropout.
        En mode entraînement, applique le masque de dropout, sinon retourne les activations sans modification.
        """
        if self.dropout_rate > 0 and training:
            mask = (np.random.rand(*A.shape) > self.dropout_rate).astype(
                float
            )  # Génère un masque
            A_drop = A * mask / (1.0 - self.dropout_rate)  # Application du masque
            return A_drop, mask
        else:
            return A, None

    def forward(self, X, training=True):
        """
        Passe avant dans le réseau, incluant BatchNorm, ReLU, et Dropout.
        Retourne les activations, caches, et masques de dropout.
        """
        activations = [X]
        caches = []
        dropout_masks = []

        num_hidden_layers = len(self.layers) - 1

        for i in range(num_hidden_layers):
            W, b = self.layers[i]
            A_prev = activations[-1]
            Z = A_prev.dot(W) + b

            bn_cache = None
            if self.use_batchnorm:
                gamma, beta = self.bn_params[i]
                rm = self.bn_running_mean[i]
                rv = self.bn_running_var[i]
                Z_norm, bn_cache, rm_up, rv_up = self._batchnorm_forward(
                    Z, gamma, beta, rm, rv, training=training
                )
                self.bn_running_mean[i] = rm_up
                self.bn_running_var[i] = rv_up
            else:
                Z_norm = Z

            A = relu(Z_norm)  # ReLU activation
            A_drop, mask = self._dropout_forward(A, training=training)  # Dropout

            caches.append((A_prev, W, b, Z, bn_cache))  # Stocke le cache pour backward
            dropout_masks.append(mask)  # Stocke le masque de dropout
            activations.append(A_drop)

        # Couche de sortie
        W_out, b_out = self.layers[-1]
        A_final = activations[-1].dot(W_out) + b_out
        A_out = softmax(A_final)  # Sortie softmax
        caches.append((activations[-1], W_out, b_out, A_final, None))
        dropout_masks.append(None)
        activations.append(A_out)

        return activations, caches, dropout_masks

    def backward(self, activations, caches, dropout_masks, y_true, training=True):
        """
        Passe arrière pour calculer les gradients des paramètres du réseau.
        Inclut le traitement de Dropout et BatchNorm si activés.
        """
        grads = []
        y_true_oh = one_hot(
            y_true, activations[-1].shape[1]
        )  # Encode labels en one-hot
        A_out = activations[-1]
        delta = (A_out - y_true_oh) / len(y_true)  # Erreur pour la couche de sortie

        # Couche de sortie
        (A_prev, W_out, b_out, Z_out, _) = caches[-1]
        dW_out = A_prev.T.dot(delta)
        db_out = np.sum(delta, axis=0, keepdims=True)
        grads.append((dW_out, db_out))

        delta = delta.dot(W_out.T)

        num_hidden_layers = len(self.layers) - 1
        # Backpropagation sur les couches cachées
        for i in reversed(range(num_hidden_layers)):
            (A_prev, W, b, Z, bn_cache) = caches[i]
            mask = dropout_masks[i]

            if self.dropout_rate > 0 and training:
                delta = delta * mask / (1.0 - self.dropout_rate)  # Reverse Dropout

            if self.use_batchnorm:
                # Backpropagation ReLU et BatchNorm
                dZ_norm_relu = relu_deriv(Z) * delta
                dZ, dgamma, dbeta = self._batchnorm_backward(
                    dZ_norm_relu, bn_cache, training=training
                )
                gamma, beta = self.bn_params[i]
                gamma -= dgamma * 0.001  # Mise à jour gamma
                beta -= dbeta * 0.001  # Mise à jour beta
                self.bn_params[i] = (gamma, beta)
            else:
                dZ = relu_deriv(Z) * delta

            dW = A_prev.T.dot(dZ)
            db = np.sum(dZ, axis=0, keepdims=True)
            grads.append((dW, db))
            delta = dZ.dot(W.T)

        grads.reverse()
        return grads

    def update_params(self, grads, lr):
        """
        Met à jour les paramètres du réseau avec option de gradient clipping.
        """
        if self.clip_norm is not None:
            total_norm = 0
            for dW, db in grads:
                total_norm += np.sum(dW**2) + np.sum(db**2)
            total_norm = np.sqrt(total_norm)
            if total_norm > self.clip_norm:
                ratio = self.clip_norm / (total_norm + 1e-8)
                grads = [(dW * ratio, db * ratio) for (dW, db) in grads]

        for i in range(len(self.layers)):
            W, b = self.layers[i]
            dW, db = grads[i]
            W -= lr * dW
            b -= lr * db
            self.layers[i] = (W, b)

    def predict(self, X):
        """
        Effectue une passe avant et retourne les prédictions.
        """
        A_out = self.forward(X, training=False)[0][-1]
        return np.argmax(A_out, axis=1)


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
    Entraîne un modèle MLP sur des données d'entraînement et évalue les performances sur validation.
    - model : instance du modèle MLP
    - X_train, y_train : données et étiquettes d'entraînement
    - X_val, y_val : données et étiquettes de validation
    - epochs : nombre d'époques
    - batch_size : taille des mini-lots
    - lr : taux d'apprentissage
    - verbose : affiche les résultats par époque si True
    """
    # Initialisation de l'historique des métriques
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # Mélange des données d'entraînement
        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        # Découpage en mini-lots
        batches = range(0, len(X_train), batch_size)
        for start in batches:
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            # Passe avant, rétropropagation et mise à jour des paramètres
            activations, caches, dropout_masks = model.forward(X_batch, training=True)
            grads = model.backward(
                activations, caches, dropout_masks, y_batch, training=True
            )
            model.update_params(grads, lr)

        # Évaluation sur l'ensemble d'entraînement
        train_activations, _, _ = model.forward(X_train, training=False)
        train_activations = train_activations[-1]
        train_loss = cross_entropy_loss(train_activations, y_train)
        train_acc = accuracy(train_activations, y_train)

        # Évaluation sur l'ensemble de validation
        val_activations, _, _ = model.forward(X_val, training=False)
        val_activations = val_activations[-1]
        val_loss = cross_entropy_loss(val_activations, y_val)
        val_acc = accuracy(val_activations, y_val)

        # Mise à jour de l'historique des métriques
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Affichage des métriques si demandé
        if verbose:
            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

    return history
