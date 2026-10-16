"""Alternative Siamese design with contrastive loss (Task 3 comparison).

Python port of Lehlonono Nkejane's R implementation (Step_3_Siamese.Rmd):
- writer-disjoint split, one fixed set of 6,000 training pairs (half same writer);
  every pair shows the same digit and comes from two different sessions;
- encoder Conv(16) -> MaxPool -> Conv(32) -> MaxPool -> Flatten -> Dropout(0.1) -> Dense(16);
- Euclidean distance d between the two embeddings;
- contrastive loss  y * d^2 + (1 - y) * max(1 - d, 0)^2   (Hadsell et al., 2006),
  Adam(3e-4), batch size 32, 10 epochs.
"""

import numpy as np
import keras
from keras import layers, ops


def sample_pairs(meta, writers, n, rng):
    """Random pairs from `writers`: same digit, different session, 50% same writer."""
    P = meta.person_id.to_numpy()
    D = meta.digit.to_numpy()
    S = meta.session.to_numpy()
    ids = np.where(np.isin(P, writers))[0]
    a = rng.choice(ids, n)
    b = np.empty(n, dtype=int)
    for k, i in enumerate(a):
        same = rng.random() < 0.5
        pool = ids[(D[ids] == D[i]) & (S[ids] != S[i]) & ((P[ids] == P[i]) == same)]
        b[k] = rng.choice(pool)
    return a, b, (P[a] == P[b]).astype("float32")


def contrastive_loss(y, d, margin=1.0):
    """Same-writer pairs are pulled together; others are pushed beyond `margin`."""
    y = ops.cast(y, "float32")
    return ops.mean(y * ops.square(d) + (1 - y) * ops.square(ops.maximum(margin - d, 0.0)))


def build_contrastive_siamese(embedding_dim=16, learning_rate=3e-4):
    """Return (encoder, siamese model that outputs the pair distance)."""
    inp = keras.Input((28, 28, 1))
    z = layers.Conv2D(16, 3, activation="relu")(inp)
    z = layers.MaxPooling2D(2)(z)
    z = layers.Conv2D(32, 3, activation="relu")(z)
    z = layers.MaxPooling2D(2)(z)
    z = layers.Flatten()(z)
    z = layers.Dropout(0.1)(z)
    z = layers.Dense(embedding_dim)(z)
    encoder = keras.Model(inp, z, name="encoder")

    a = keras.Input((28, 28, 1))
    b = keras.Input((28, 28, 1))
    dist = layers.Lambda(lambda t: ops.sqrt(
        ops.sum(ops.square(t[0] - t[1]), axis=1, keepdims=True) + 1e-7))([encoder(a), encoder(b)])
    model = keras.Model([a, b], dist)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=contrastive_loss)
    return encoder, model
