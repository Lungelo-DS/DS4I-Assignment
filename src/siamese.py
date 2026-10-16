"""Siamese network for 'same writer?' matching (Tasks 3-6).

Design follows the Keras triplet-loss example from the course notes:
a shared embedding CNN maps each image to a vector, and the triplet loss
    L(A, P, N) = max(||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + margin, 0)
pulls an anchor towards a positive (same writer) and pushes it away from a
negative (different writer).

Data layout: the dataset is perfectly balanced, so every image can be looked
up by (person, digit, session, replicate) in a 4-D index table.
"""

import numpy as np
import keras
from keras import layers, ops
from sklearn.metrics import roc_auc_score, roc_curve


# --------------------------------------------------------------------------- index table

def make_index_table(meta):
    """table[person, digit, session, replicate] -> row number in `images`.

    person and session keep their original 1-based values (index 0 unused);
    digit 0-4 and replicate 1-50 (index 0 unused).
    """
    table = np.full((21, 5, 6, 51), -1, dtype=int)
    table[meta.person_id.values, meta.digit.values,
          meta.session.values, meta.replicate.values] = np.arange(len(meta))
    return table


def _other(values, current, rng):
    """For each element of `current`, pick a random value from `values` that differs from it."""
    values = np.asarray(values)
    pos = np.searchsorted(values, current)
    offset = rng.integers(1, len(values), size=len(current))
    return values[(pos + offset) % len(values)]


# --------------------------------------------------------------------------- training triplets

def sample_triplets(table, writers, sessions, n, rng):
    """Random digit-controlled triplets.

    anchor   : random writer, digit, session, replicate
    positive : SAME writer, SAME digit, DIFFERENT session (if more than one session)
    negative : DIFFERENT writer, SAME digit, any session

    Keeping the digit fixed means the network cannot tell writers apart by digit
    shape alone. Taking the positive from another session stops it from learning
    the near-duplicate replicates within one session.
    """
    writers, sessions = np.sort(writers), np.sort(sessions)
    p = rng.choice(writers, n)
    d = rng.integers(0, 5, n)
    s = rng.choice(sessions, n)
    r = rng.integers(1, 51, n)

    s_pos = _other(sessions, s, rng) if len(sessions) > 1 else s
    r_pos = rng.integers(1, 51, n)
    p_neg = _other(writers, p, rng)
    s_neg = rng.choice(sessions, n)
    r_neg = rng.integers(1, 51, n)

    return table[p, d, s, r], table[p, d, s_pos, r_pos], table[p_neg, d, s_neg, r_neg]


# --------------------------------------------------------------------------- evaluation pairs

def make_pairs(table, writers, n, rng, mode="diff_session", sessions=(1, 2, 3, 4, 5),
               ref_sessions=None):
    """Balanced positive (same writer) / negative (different writer) pairs.

    mode
      "diff_session" : same digit, the two images come from different sessions (main test)
      "same_session" : same digit, same session (near-duplicates -> optimistic)
      "diff_digit"   : the two images show DIFFERENT digits (pair not controlled for digit)
    ref_sessions: if given, image 1 comes from `sessions` and image 2 from `ref_sessions`
      (used for the session split: new session vs enrolled sessions).
    Returns idx1, idx2, label (1 = same writer), digit of image 1.
    """
    writers, sessions = np.sort(writers), np.sort(sessions)
    half = n // 2
    p1 = rng.choice(writers, n)
    d1 = rng.integers(0, 5, n)
    s1 = rng.choice(sessions, n)
    r1 = rng.integers(1, 51, n)

    p2 = p1.copy()
    p2[half:] = _other(writers, p1[half:], rng)          # second half = negatives
    label = np.r_[np.ones(half, int), np.zeros(n - half, int)]

    d2 = d1.copy() if mode != "diff_digit" else _other(np.arange(5), d1, rng)
    if ref_sessions is not None:
        s2 = rng.choice(np.sort(ref_sessions), n)
    elif mode == "same_session":
        s2 = s1.copy()
    else:
        s2 = _other(sessions, s1, rng)
    r2 = rng.integers(1, 51, n)
    if mode == "same_session":                            # avoid pairing an image with itself
        clash = (p1 == p2) & (r1 == r2)
        r2[clash] = r2[clash] % 50 + 1

    return table[p1, d1, s1, r1], table[p2, d2, s2, r2], label, d1


# --------------------------------------------------------------------------- model

def build_embedding(dim=64, augment=True, dropout=0.25):
    """Shared CNN that maps one 28x28 image to a unit-length embedding vector."""
    inp = keras.Input((28, 28, 1))
    x = inp
    if augment:                                   # active only during training
        x = layers.RandomRotation(0.04)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomTranslation(0.1, 0.1)(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(dim)(x)
    out = layers.UnitNormalization()(x)           # embeddings lie on the unit sphere
    return keras.Model(inp, out, name="embedding")


def triplet_loss(margin):
    """Triplet loss on the concatenated [anchor, positive, negative] embeddings."""
    def loss(y_true, y_pred):
        a, p, n = ops.split(y_pred, 3, axis=1)
        d_ap = ops.sum(ops.square(a - p), axis=1)   # squared distance anchor-positive
        d_an = ops.sum(ops.square(a - n), axis=1)   # squared distance anchor-negative
        return ops.maximum(d_ap - d_an + margin, 0.0)
    return loss


def build_triplet_model(embedding, margin=0.5, learning_rate=1e-3):
    """Three inputs share ONE embedding network (same weights)."""
    a = keras.Input((28, 28, 1), name="anchor")
    p = keras.Input((28, 28, 1), name="positive")
    n = keras.Input((28, 28, 1), name="negative")
    out = layers.Concatenate()([embedding(a), embedding(p), embedding(n)])
    model = keras.Model([a, p, n], out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=triplet_loss(margin))
    return model


def embed(embedding, x, batch_size=1024):
    return embedding.predict(x, batch_size=batch_size, verbose=0)


def pair_distance(emb, idx1, idx2):
    """Squared Euclidean distance between embeddings (0 = identical, 4 = opposite)."""
    return ((emb[idx1] - emb[idx2]) ** 2).sum(axis=1)


# --------------------------------------------------------------------------- metrics

def verification_metrics(dist, label, threshold=None):
    """AUC, equal error rate, and error rates at a distance threshold.

    Pairs with distance <= threshold are accepted as 'same writer'.
    If no threshold is given, the equal-error-rate threshold of these pairs is used.
    """
    score = -dist
    auc = roc_auc_score(label, score)
    fpr, tpr, thr = roc_curve(label, score)
    i = np.argmin(np.abs(fpr - (1 - tpr)))
    eer = (fpr[i] + 1 - tpr[i]) / 2
    if threshold is None:
        threshold = -thr[i]
    accept = dist <= threshold
    far = accept[label == 0].mean()               # false accept rate
    frr = 1 - accept[label == 1].mean()           # false reject rate
    acc = (accept == label.astype(bool)).mean()
    return {"AUC": auc, "EER": eer, "threshold": threshold,
            "false accept": far, "false reject": frr, "accuracy": acc}


def prototypes(emb, meta, rows):
    """Mean embedding per (writer, digit) over `rows`, re-normalised (Utkin et al.)."""
    keys = meta.iloc[rows][["person_id", "digit"]].to_numpy()
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    proto = np.zeros((len(uniq), emb.shape[1]))
    np.add.at(proto, inv.ravel(), emb[rows])
    proto /= np.linalg.norm(proto, axis=1, keepdims=True)
    return proto, uniq[:, 0]                      # prototypes and their writer IDs


def identify(emb_query, proto, proto_writer, top=3):
    """Nearest-prototype writer identification. Returns top-1 writer and top-k writers."""
    sim = emb_query @ proto.T                     # cosine similarity (unit vectors)
    order = np.argsort(-sim, axis=1)
    ranked = proto_writer[order]
    top_k = np.array([list(dict.fromkeys(row))[:top] for row in ranked])
    return ranked[:, 0], top_k


# --------------------------------------------------------------------------- training loop

def train_siamese(x, table, train_writers, train_sessions, val_fn, dim=64, margin=0.5,
                  learning_rate=1e-3, augment=True, n_triplets=20000, max_epochs=25,
                  patience=4, seed=0, verbose=True):
    """Train with fresh random triplets every epoch; keep the epoch with the best validation score.

    val_fn(embedding) -> number where higher is better (e.g. validation AUC).
    """
    keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    embedding = build_embedding(dim=dim, augment=augment)
    model = build_triplet_model(embedding, margin=margin, learning_rate=learning_rate)

    history, best, best_weights, wait = [], -np.inf, None, 0
    for epoch in range(1, max_epochs + 1):
        a, p, n = sample_triplets(table, train_writers, train_sessions, n_triplets, rng)
        h = model.fit([x[a], x[p], x[n]], np.zeros((len(a), 1)),
                      batch_size=128, epochs=1, verbose=0)
        score = val_fn(embedding)
        history.append({"epoch": epoch, "train loss": h.history["loss"][0], "validation": score})
        if verbose:
            print(f"epoch {epoch:2d}  train loss {h.history['loss'][0]:.4f}  validation {score:.4f}")
        if score > best:
            best, best_weights, wait = score, embedding.get_weights(), 0
        else:
            wait += 1
            if wait >= patience:
                break
    embedding.set_weights(best_weights)
    return embedding, history
