# Imports utiles
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

            Z_norm = (Z - batch_mean) / np.sqrt(
                batch_var + 1e-8
            )  # Normalisation des données du batch
        else:
            Z_norm = (Z - running_mean) / np.sqrt(
                running_var + 1e-8
            )  # Normalisation en utilisant les moyennes cumulées

        # Transformation linéaire avec les paramètres gamma (échelle) et beta (décalage)
        out = gamma * Z_norm + beta
        cache = (
            Z,  # Entrée originale
            Z_norm,  # Entrée normalisée
            gamma,  # Paramètre d'échelle
            beta,  # Paramètre de décalage
            running_mean,  # Moyenne cumulative
            running_var,  # Variance cumulative
        )  # Cache pour les calculs de la passe arrière

        return (
            out,
            cache,
            running_mean,
            running_var,
        )  # Résultat de la passe avant, cache, moyennes et variances mises à jour

    def _batchnorm_backward(self, dout, cache, training=False):
        """
        Passe arrière pour Batch Normalization.
        Calcule les gradients pour gamma, beta, et l'entrée Z.
        """
        # Extraction des variables depuis le cache
        Z, Z_norm, gamma, beta, running_mean, running_var = cache
        N = Z.shape[0]  # Nombre d'échantillons dans le batch

        # Calcul du gradient de gamma (échelle)
        dgamma = np.sum(
            dout * Z_norm, axis=0, keepdims=True
        )  # Somme des gradients pondérés par Z normalisé
        # Calcul du gradient de beta (décalage)
        dbeta = np.sum(dout, axis=0, keepdims=True)  # Somme des gradients directs

        # Recalcul de la moyenne et de la variance du batch pour la passe arrière
        batch_var = np.var(Z, axis=0, keepdims=True)  # Variance du batch
        batch_mean = np.mean(Z, axis=0, keepdims=True)  # Moyenne du batch

        # Calcul du gradient par rapport à Z normalisé
        dZ_norm = dout * gamma  # Contribution de gamma à la dérivée
        # Gradient par rapport à la variance
        dvar = np.sum(
            dZ_norm * (Z - batch_mean) * (-0.5) * (batch_var + 1e-8) ** (-3 / 2),
            axis=0,
            keepdims=True,
        )
        # Gradient par rapport à la moyenne
        dmean = np.sum(
            dZ_norm * (-1 / np.sqrt(batch_var + 1e-8)), axis=0, keepdims=True
        ) + dvar * np.mean(-2 * (Z - batch_mean), axis=0, keepdims=True)
        # Gradient final par rapport à Z
        dZ = (
            (dZ_norm / np.sqrt(batch_var + 1e-8))  # Contribution de la normalisation
            + (dvar * 2 * (Z - batch_mean) / N)  # Contribution de la variance
            + (dmean / N)  # Contribution de la moyenne
        )

        # Retour des gradients calculés
        return dZ, dgamma, dbeta

    def _dropout_forward(self, A, training=True):
        """
        Passe avant pour Dropout.
        En mode entraînement, applique le masque de dropout, sinon retourne les activations sans modification.
        """
        # Vérifie si Dropout est activé et si l'on est en mode entraînement
        if self.dropout_rate > 0 and training:
            # Génération d'un masque aléatoire avec des valeurs 0 ou 1 en fonction du taux de dropout
            mask = (np.random.rand(*A.shape) > self.dropout_rate).astype(float)
            # Applique le masque aux activations et les renormalise pour conserver l'échelle
            A_drop = A * mask / (1.0 - self.dropout_rate)
            return (
                A_drop,
                mask,
            )  # Retourne les activations avec dropout et le masque utilisé
        else:
            # En mode test ou si dropout est désactivé, retourne les activations sans modification
            return A, None

    def forward(self, X, training=True):
        """
        Passe avant dans le réseau, incluant BatchNorm, ReLU, et Dropout.
        Retourne les activations, caches, et masques de dropout.
        """
        activations = [X]  # Liste pour stocker les activations des couches
        caches = []  # Liste pour stocker les caches pour la passe arrière
        dropout_masks = []  # Liste pour stocker les masques de dropout

        num_hidden_layers = len(self.layers) - 1  # Nombre de couches cachées

        for i in range(num_hidden_layers):
            # Récupération des poids et biais de la couche courante
            W, b = self.layers[i]
            # Activation de la couche précédente
            A_prev = activations[-1]
            # Calcul des activations linéaires (Z = A_prev * W + b)
            Z = A_prev.dot(W) + b

            bn_cache = None
            if self.use_batchnorm:
                # Si BatchNorm est activé, applique la normalisation
                gamma, beta = self.bn_params[
                    i
                ]  # Paramètres gamma et beta pour BatchNorm
                rm = self.bn_running_mean[i]  # Moyenne cumulative
                rv = self.bn_running_var[i]  # Variance cumulative
                Z_norm, bn_cache, rm_up, rv_up = self._batchnorm_forward(
                    Z, gamma, beta, rm, rv, training=training
                )
                # Mise à jour des moyennes et variances cumulées
                self.bn_running_mean[i] = rm_up
                self.bn_running_var[i] = rv_up
            else:
                Z_norm = Z  # Si pas de BatchNorm, Z reste inchangé

            # Application de l'activation ReLU
            A = relu(Z_norm)
            # Application de Dropout (si activé)
            A_drop, mask = self._dropout_forward(A, training=training)

            # Stocke les données nécessaires pour la passe arrière
            caches.append((A_prev, W, b, Z, bn_cache))
            # Stocke le masque de dropout pour la passe arrière
            dropout_masks.append(mask)
            # Ajoute les activations de la couche courante
            activations.append(A_drop)

        # Couche de sortie
        W_out, b_out = self.layers[-1]  # Poids et biais de la dernière couche
        A_final = activations[-1].dot(W_out) + b_out  # Calcul des scores finaux
        A_out = softmax(A_final)  # Application de la fonction softmax pour la sortie
        # Stockage des données pour la passe arrière de la couche de sortie
        caches.append((activations[-1], W_out, b_out, A_final, None))
        dropout_masks.append(None)  # Pas de masque de dropout pour la couche de sortie
        activations.append(A_out)  # Ajoute les activations finales (sorties)

        # Retourne les activations, caches, et masques de dropout
        return activations, caches, dropout_masks

    def backward(self, activations, caches, dropout_masks, y_true, training=True):
        """
        Passe arrière pour calculer les gradients des paramètres du réseau.
        Inclut le traitement de Dropout et BatchNorm si activés.
        """
        grads = []  # Liste pour stocker les gradients des paramètres
        # Conversion des labels y_true en one-hot encoding
        y_true_oh = one_hot(y_true, activations[-1].shape[1])
        A_out = activations[-1]  # Activations de la couche de sortie
        # Calcul de l'erreur pour la couche de sortie
        delta = (A_out - y_true_oh) / len(y_true)

        # Traitement pour la couche de sortie
        (A_prev, W_out, b_out, Z_out, _) = caches[-1]  # Extraction des données du cache
        dW_out = A_prev.T.dot(delta)  # Gradient des poids
        db_out = np.sum(delta, axis=0, keepdims=True)  # Gradient des biais
        grads.append((dW_out, db_out))  # Ajoute les gradients de la couche de sortie
        delta = delta.dot(W_out.T)  # Propagation de l'erreur à la couche précédente

        num_hidden_layers = len(self.layers) - 1  # Nombre de couches cachées
        # Backpropagation pour les couches cachées
        for i in reversed(range(num_hidden_layers)):
            (A_prev, W, b, Z, bn_cache) = caches[i]  # Extraction des données du cache
            mask = dropout_masks[i]  # Récupération du masque de dropout

            if self.dropout_rate > 0 and training:
                # Ajustement du delta avec le masque de dropout
                delta = delta * mask / (1.0 - self.dropout_rate)

            if self.use_batchnorm:
                # Backpropagation combinée ReLU et BatchNorm
                dZ_norm_relu = relu_deriv(Z) * delta  # Dérivée de ReLU combinée à delta
                dZ, dgamma, dbeta = self._batchnorm_backward(
                    dZ_norm_relu, bn_cache, training=training
                )  # Calcul des gradients avec BatchNorm
                # Mise à jour des paramètres gamma et beta
                gamma, beta = self.bn_params[i]
                gamma -= (
                    dgamma * 0.001
                )  # Mise à jour gamma avec un pas d'apprentissage fixe
                beta -= (
                    dbeta * 0.001
                )  # Mise à jour beta avec un pas d'apprentissage fixe
                self.bn_params[i] = (gamma, beta)  # Sauvegarde des nouveaux paramètres
            else:
                dZ = relu_deriv(Z) * delta  # Dérivée de ReLU appliquée directement

            dW = A_prev.T.dot(dZ)  # Gradient des poids
            db = np.sum(dZ, axis=0, keepdims=True)  # Gradient des biais
            grads.append((dW, db))  # Ajoute les gradients de la couche courante
            delta = dZ.dot(W.T)  # Propagation de l'erreur à la couche précédente

        grads.reverse()  # Inversion de l'ordre des gradients pour correspondre aux couches
        return grads  # Retourne les gradients pour toutes les couches

    def update_params(self, grads, lr):
        """
        Met à jour les paramètres du réseau avec option de gradient clipping.
        """
        if self.clip_norm is not None:
            total_norm = 0  # Variable pour stocker la norme totale des gradients
            for dW, db in grads:
                # Accumulation de la somme des carrés des gradients
                total_norm += np.sum(dW**2) + np.sum(db**2)
            total_norm = np.sqrt(total_norm)  # Calcul de la norme L2 totale
            if total_norm > self.clip_norm:
                # Si la norme dépasse la limite, on ajuste les gradients
                ratio = self.clip_norm / (total_norm + 1e-8)  # Ratio de réduction
                grads = [
                    (dW * ratio, db * ratio) for (dW, db) in grads
                ]  # Application du ratio aux gradients

        for i in range(len(self.layers)):
            # Récupération des paramètres de la couche courante
            W, b = self.layers[i]
            dW, db = grads[i]  # Gradients pour les poids et les biais
            # Mise à jour des poids et des biais avec le taux d'apprentissage
            W -= lr * dW
            b -= lr * db
            # Sauvegarde des nouveaux paramètres
            self.layers[i] = (W, b)

    def predict(self, X):
        """
        Effectue une passe avant et retourne les prédictions.
        """
        # Passe avant en mode test (training=False) et récupération de la sortie
        A_out = self.forward(X, training=False)[0][-1]
        # Retourne les indices des classes avec la plus grande probabilité (prédictions)
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
