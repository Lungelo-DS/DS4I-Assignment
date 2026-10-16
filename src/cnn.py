"""CNN for writer identification (Task 2).

Prediction task: from one 28 x 28 digit image, predict which of the 20 writers
wrote it (20-class classification). The digit label is not given to the model.

Architecture (Lecture 4 pattern): optional augmentation -> Conv(32) -> Conv(32)
-> MaxPool -> Dropout -> Conv(64) -> MaxPool -> Dropout -> Dense(128, "features")
-> Dropout(0.5) -> Dense(20, softmax). Trained with categorical cross-entropy and Adam.
"""

import numpy as np
import pandas as pd
import keras
from keras import layers


def session_split(meta):
    """Row indices for train (sessions 1-3), validation (4) and test (5)."""
    s = meta["session"].to_numpy()
    return np.where(s <= 3)[0], np.where(s == 4)[0], np.where(s == 5)[0]


def random_split(n, seed=0, fractions=(0.6, 0.2)):
    """Random 60/20/20 split of image rows, ignoring sessions (leaky comparison)."""
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(fractions[0] * n), int((fractions[0] + fractions[1]) * n)
    return idx[:a], idx[a:b], idx[b:]


def build_cnn(augment=True, n_classes=20, dropout=0.25, dense_units=128):
    """Build and compile the CNN.

    augment: add random rotation (about +/-15 degrees), zoom (+/-10%) and shift
        (+/-10%), applied to training images only, to mimic between-session changes.
    The 128-unit dense layer is named "features" so it can be reused as an embedding.
    """
    inp = keras.Input((28, 28, 1))
    h = inp
    if augment:
        h = layers.RandomRotation(0.04)(h)
        h = layers.RandomZoom(0.1)(h)
        h = layers.RandomTranslation(0.1, 0.1)(h)
    h = layers.Conv2D(32, 3, activation="relu")(h)
    h = layers.Conv2D(32, 3, activation="relu")(h)
    h = layers.MaxPooling2D(2)(h)
    h = layers.Dropout(dropout)(h)
    h = layers.Conv2D(64, 3, activation="relu")(h)
    h = layers.MaxPooling2D(2)(h)
    h = layers.Dropout(dropout)(h)
    h = layers.Flatten()(h)
    feats = layers.Dense(dense_units, activation="relu", name="features")(h)
    h = layers.Dropout(0.5)(feats)
    out = layers.Dense(n_classes, activation="softmax")(h)
    model = keras.Model(inp, out, name="cnn")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_cnn(model, x_train, y_train, x_val, y_val, n_classes=20, epochs=40,
              batch_size=128, patience=5, verbose=0):
    """Fit with early stopping on validation loss (best weights kept).

    y_train / y_val are integer labels 0..n_classes-1. Returns the history as a DataFrame.
    """
    hist = model.fit(
        x_train, keras.utils.to_categorical(y_train, n_classes),
        validation_data=(x_val, keras.utils.to_categorical(y_val, n_classes)),
        epochs=epochs, batch_size=batch_size, verbose=verbose,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                                 restore_best_weights=True)])
    df = pd.DataFrame(hist.history)
    df.insert(0, "epoch", np.arange(1, len(df) + 1))
    return df


def feature_model(cnn):
    """Model that outputs the CNN's 128-unit 'features' layer (penultimate layer)."""
    return keras.Model(cnn.input, cnn.get_layer("features").output)


def topk_accuracy(probs, y, k=1):
    """Share of rows whose true label (0-based) is among the k most probable classes."""
    top = np.argsort(-probs, axis=1)[:, :k]
    return float((top == np.asarray(y)[:, None]).any(axis=1).mean())


def multi_image_accuracy(probs, y, ks=(1, 2, 3, 5, 10), n_draws=200, seed=0):
    """Accuracy when k images from the same writer are combined.

    For each writer, k test images are drawn at random; their predicted probabilities
    are averaged and the most likely writer is chosen. Repeated n_draws times per writer.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    rows = []
    for k in ks:
        hits = []
        for w in np.unique(y):
            members = np.where(y == w)[0]
            for _ in range(n_draws):
                chosen = rng.choice(members, size=k, replace=False)
                hits.append(probs[chosen].mean(axis=0).argmax() == w)
        rows.append({"images per decision": k, "accuracy": float(np.mean(hits))})
    return pd.DataFrame(rows)
