"""All model training for the report, with results cached in `results/`.

Each `run_*` function trains its models, writes its outputs to `results/`, and
skips the work if those files already exist (pass force=True to recompute).
`run_experiments.py` in the repository root calls them in order; the Quarto report
only reads the saved files, so it renders in seconds.

Steps and approximate CPU time:
  run_cnn                 Task 2/4   CNN baseline, augmented, random split   ~8 min
  run_siamese_folds       Task 3/4   triplet Siamese, 5 writer folds         ~25 min
  run_contrastive_folds   Task 3     contrastive Siamese, same folds         ~2 min
  run_hpo                 Task 5     margin x embedding size grid + seeds    ~30 min
  run_comparison          Task 6     CNN vs Siamese on session 5             ~2 min
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import keras
from scipy.stats import spearmanr

from data_utils import load_handwriting, REPO_ROOT
import cnn as cnn_mod
import contrastive as con
from siamese import (make_index_table, make_pairs, train_siamese, embed, pair_distance,
                     verification_metrics, prototypes, identify, build_embedding)

RESULTS = REPO_ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def _done(*names):
    return all((RESULTS / n).exists() for n in names)


def _data():
    images, meta = load_handwriting()
    return images, images[..., None], meta


def writer_folds(seed=0):
    """Five writer-disjoint folds: (train writers, validation writers, test writers).

    Writers are shuffled and cut into 5 groups of 4; fold k tests group k, validates
    on group k+1 and trains on the other 12 writers.
    """
    writers = np.random.default_rng(seed).permutation(np.arange(1, 21))
    groups = writers.reshape(5, 4)
    folds = []
    for k in range(5):
        test_w, val_w = groups[k], groups[(k + 1) % 5]
        folds.append((np.setdiff1d(writers, np.r_[test_w, val_w]), val_w, test_w))
    return folds


# --------------------------------------------------------------------------- Task 2 / 4

def run_cnn(force=False):
    """CNN on the session split (baseline and augmented) and on a random split."""
    files = ("cnn_history.csv", "cnn_test_probs.npz", "cnn_summary.json", "cnn_aug.weights.h5")
    if _done(*files) and not force:
        return
    _, x, meta = _data()
    y = meta.person_id.to_numpy() - 1
    tr, va, te = cnn_mod.session_split(meta)

    hist, probs = [], {}
    for name, augment in [("baseline", False), ("augmented", True)]:
        keras.utils.set_random_seed(42)
        model = cnn_mod.build_cnn(augment=augment)
        h = cnn_mod.train_cnn(model, x[tr], y[tr], x[va], y[va])
        hist.append(h.assign(model=name))
        probs[name] = model.predict(x[te], verbose=0)
        if augment:
            model.save_weights(RESULTS / "cnn_aug.weights.h5")

    keras.utils.set_random_seed(42)
    r_tr, r_va, r_te = cnn_mod.random_split(len(x))
    rmodel = cnn_mod.build_cnn(augment=False)
    cnn_mod.train_cnn(rmodel, x[r_tr], y[r_tr], x[r_va], y[r_va], epochs=15)
    random_acc = float((rmodel.predict(x[r_te], verbose=0).argmax(1) == y[r_te]).mean())

    pd.concat(hist).to_csv(RESULTS / "cnn_history.csv", index=False)
    np.savez(RESULTS / "cnn_test_probs.npz", baseline=probs["baseline"],
             augmented=probs["augmented"], test_rows=te)
    json.dump({"random_split_accuracy": random_acc,
               "n_params": int(rmodel.count_params())},
              open(RESULTS / "cnn_summary.json", "w"), indent=2)


# --------------------------------------------------------------------------- Task 3 / 4

def run_siamese_folds(force=False):
    """Triplet Siamese on 5 writer-disjoint folds; test pairs from unseen writers only."""
    files = ("siamese_writer_folds.csv", "siamese_writer_folds_by_digit.csv",
             "siamese_folds_history.csv", "siamese_fold0_dist.npz")
    if _done(*files) and not force:
        return
    images, x, meta = _data()
    X_flat = images.reshape(len(images), -1)
    table = make_index_table(meta)

    rows, digit_rows, histories = [], [], []
    for k, (train_w, val_w, test_w) in enumerate(writer_folds()):
        v1, v2, v_label, _ = make_pairs(table, val_w, 4000, np.random.default_rng(100 + k))

        def val_auc(embedding):
            e = embed(embedding, x[np.r_[v1, v2]])
            return verification_metrics(((e[:len(v1)] - e[len(v1):]) ** 2).sum(1), v_label)["AUC"]

        embedding, hist = train_siamese(x, table, train_w, np.arange(1, 6), val_auc,
                                        n_triplets=15000, max_epochs=15, patience=3,
                                        seed=k, verbose=False)
        histories.append(pd.DataFrame(hist).assign(fold=k))
        emb = embed(embedding, x)
        threshold = verification_metrics(pair_distance(emb, v1, v2), v_label)["threshold"]

        rng = np.random.default_rng(200 + k)
        for mode in ["diff_session", "same_session", "diff_digit"]:
            i1, i2, label, digit = make_pairs(table, test_w, 6000, rng, mode)
            dist = pair_distance(emb, i1, i2)
            res = verification_metrics(dist, label, threshold)
            res.update(fold=k, pairs=mode, model="Siamese")
            rows.append(res)
            if mode == "diff_session":
                raw = verification_metrics(((X_flat[i1] - X_flat[i2]) ** 2).sum(1), label)
                rows.append({"AUC": raw["AUC"], "EER": raw["EER"], "fold": k,
                             "pairs": mode, "model": "raw pixels"})
                for d in range(5):
                    m = digit == d
                    digit_rows.append({"fold": k, "digit": d,
                                       "AUC": verification_metrics(dist[m], label[m])["AUC"]})
                if k == 0:
                    np.savez(RESULTS / "siamese_fold0_dist.npz", dist=dist, label=label,
                             threshold=threshold)

    pd.DataFrame(rows).to_csv(RESULTS / "siamese_writer_folds.csv", index=False)
    pd.DataFrame(digit_rows).to_csv(RESULTS / "siamese_writer_folds_by_digit.csv", index=False)
    pd.concat(histories).to_csv(RESULTS / "siamese_folds_history.csv", index=False)


def run_contrastive_folds(force=False):
    """Contrastive Siamese (L. Nkejane's design) on the same folds and test pairs."""
    if _done("contrastive_folds.csv") and not force:
        return
    _, x, meta = _data()
    table = make_index_table(meta)
    rows = []
    for k, (train_w, val_w, test_w) in enumerate(writer_folds()):
        keras.utils.set_random_seed(k)
        rng = np.random.default_rng(2026 + k)
        ta, tb, ty = con.sample_pairs(meta, train_w, 6000, rng)
        va, vb, vy = con.sample_pairs(meta, val_w, 2000, rng)
        encoder, model = con.build_contrastive_siamese()
        start = time.time()
        model.fit([x[ta], x[tb]], ty, validation_data=([x[va], x[vb]], vy),
                  epochs=10, batch_size=32, verbose=0)
        emb = encoder.predict(x, batch_size=1024, verbose=0)
        # identical test pairs to run_siamese_folds (same generator seed, first draw)
        i1, i2, label, _ = make_pairs(table, test_w, 6000, np.random.default_rng(200 + k))
        dist = ((emb[i1] - emb[i2]) ** 2).sum(1)
        val_dist = ((emb[va] - emb[vb]) ** 2).sum(1)
        thr = verification_metrics(val_dist, vy.astype(int))["threshold"]
        res = verification_metrics(dist, label, thr)
        res.update(fold=k, model="Siamese (contrastive)",
                   accuracy_fixed_0_5=float(((np.sqrt(dist) < 0.5) == label).mean()),
                   seconds=time.time() - start)
        rows.append(res)
    pd.DataFrame(rows).to_csv(RESULTS / "contrastive_folds.csv", index=False)


# --------------------------------------------------------------------------- Task 5

def _session_val_pairs(table):
    return make_pairs(table, np.arange(1, 21), 6000, np.random.default_rng(42),
                      sessions=[4], ref_sessions=[1, 2, 3])


def run_hpo(force=False):
    """Margin x embedding-size grid (seed 0) plus two extra seeds of the default.

    Trained on sessions 1-3 of all writers, scored on session 4 only. The weights of
    the run with the highest validation AUC are saved for Task 6.
    """
    if _done("siamese_hpo.csv", "siamese_best.weights.h5") and not force:
        return
    _, x, meta = _data()
    y = meta.person_id.to_numpy()
    table = make_index_table(meta)
    session = meta.session.to_numpy()
    train_rows, val_rows = np.where(session <= 3)[0], np.where(session == 4)[0]
    v1, v2, v_label, _ = _session_val_pairs(table)

    def val_auc(embedding):
        e = embed(embedding, x[np.r_[v1, v2]])
        return verification_metrics(((e[:len(v1)] - e[len(v1):]) ** 2).sum(1), v_label)["AUC"]

    grid = [(m, d, 0) for m in [0.2, 0.5, 1.0] for d in [16, 64]] + [(0.5, 64, 1), (0.5, 64, 2)]
    rows, best = [], -1
    for margin, dim, seed in grid:
        start = time.time()
        emb_model, hist = train_siamese(x, table, np.arange(1, 21), [1, 2, 3], val_auc,
                                        dim=dim, margin=margin, n_triplets=15000,
                                        max_epochs=15, patience=3, seed=seed, verbose=False)
        emb = embed(emb_model, x)
        ver = verification_metrics(pair_distance(emb, v1, v2), v_label)
        proto, proto_w = prototypes(emb, meta, train_rows)
        top1, top3 = identify(emb[val_rows], proto, proto_w)
        rows.append({"margin": margin, "embedding size": dim, "seed": seed, "epochs": len(hist),
                     "val AUC": ver["AUC"], "val EER": ver["EER"],
                     "val identification": (top1 == y[val_rows]).mean(),
                     "val top-3": (top3 == y[val_rows][:, None]).any(axis=1).mean(),
                     "minutes": (time.time() - start) / 60})
        if ver["AUC"] > best:
            best = ver["AUC"]
            emb_model.save_weights(RESULTS / "siamese_best.weights.h5")
    pd.DataFrame(rows).to_csv(RESULTS / "siamese_hpo.csv", index=False)


# --------------------------------------------------------------------------- Task 6

def run_comparison(force=False):
    """CNN vs Siamese on session 5: identification, verification, digits, sessions."""
    files = ("comparison_identification.csv", "comparison_verification.csv",
             "comparison_by_digit.csv", "comparison_per_writer.csv", "comparison_sessions.csv")
    if _done(*files) and not force:
        return
    images, x, meta = _data()
    X_flat = images.reshape(len(images), -1)
    y = meta.person_id.to_numpy()
    digit = meta.digit.to_numpy()
    session = meta.session.to_numpy()
    table = make_index_table(meta)
    tr, va, te = cnn_mod.session_split(meta)

    cnn = cnn_mod.build_cnn(augment=True)
    cnn.load_weights(RESULTS / "cnn_aug.weights.h5")
    f = cnn_mod.feature_model(cnn).predict(x, batch_size=1024, verbose=0)
    emb_cnn = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)

    hpo = pd.read_csv(RESULTS / "siamese_hpo.csv")
    best = hpo.sort_values("val AUC", ascending=False).iloc[0]
    sia = build_embedding(dim=int(best["embedding size"]))
    sia.load_weights(RESULTS / "siamese_best.weights.h5")
    emb_sia = embed(sia, x)

    def cnn_predict(rows):
        p = cnn.predict(x[rows], verbose=0)
        return p.argmax(1) + 1, np.argsort(-p, axis=1)[:, :3] + 1

    proto_s, pw_s = prototypes(emb_sia, meta, tr)
    proto_c, pw_c = prototypes(emb_cnn, meta, tr)
    truth = y[te]
    c1, c3 = cnn_predict(te)
    s1, s3 = identify(emb_sia[te], proto_s, pw_s)
    p1, p3 = identify(emb_cnn[te], proto_c, pw_c)

    def acc(t1, t3):
        return (t1 == truth).mean(), (t3 == truth[:, None]).any(1).mean()

    ident = pd.DataFrame([("CNN (softmax)", *acc(c1, c3)),
                          ("Siamese (nearest prototype)", *acc(s1, s3)),
                          ("CNN features (nearest prototype)", *acc(p1, p3))],
                         columns=["model", "top-1 accuracy", "top-3 accuracy"])
    ident.to_csv(RESULTS / "comparison_identification.csv", index=False)

    # verification: session-5 image vs session 1-3 image, same digit
    i1, i2, label, pair_digit = make_pairs(table, np.arange(1, 21), 20000,
                                           np.random.default_rng(5), sessions=[5],
                                           ref_sessions=[1, 2, 3])
    w1, w2, w_label, _ = make_pairs(table, np.arange(1, 21), 10000, np.random.default_rng(4),
                                    sessions=[4], ref_sessions=[1, 2, 3])
    dist_fns = {"Siamese": lambda a, b: pair_distance(emb_sia, a, b),
                "CNN features": lambda a, b: pair_distance(emb_cnn, a, b),
                "raw pixels": lambda a, b: ((X_flat[a] - X_flat[b]) ** 2).sum(1)}
    ver_rows, dists = [], {}
    for name, fn in dist_fns.items():
        thr = verification_metrics(fn(w1, w2), w_label)["threshold"]
        dists[name] = fn(i1, i2)
        r = verification_metrics(dists[name], label, thr)
        r["model"] = name
        ver_rows.append(r)
    pd.DataFrame(ver_rows).to_csv(RESULTS / "comparison_verification.csv", index=False)

    test_digit = digit[te]
    by_digit = []
    for d in range(5):
        m, pm = test_digit == d, pair_digit == d
        by_digit.append({
            "digit": d,
            "CNN identification": (c1[m] == truth[m]).mean(),
            "Siamese identification": (s1[m] == truth[m]).mean(),
            "Siamese verification AUC": verification_metrics(dists["Siamese"][pm], label[pm])["AUC"],
            "CNN-feature verification AUC":
                verification_metrics(dists["CNN features"][pm], label[pm])["AUC"]})
    pd.DataFrame(by_digit).to_csv(RESULTS / "comparison_by_digit.csv", index=False)

    shift = []
    for w in range(1, 21):
        per_digit = [np.linalg.norm(
            X_flat[(y == w) & (digit == d) & (session == 5)].mean(0)
            - X_flat[(y == w) & (digit == d) & (session <= 3)].mean(0)) for d in range(5)]
        shift.append(np.mean(per_digit))
    per_writer = pd.DataFrame({
        "writer": np.arange(1, 21), "session shift": shift,
        "CNN accuracy": pd.Series(c1 == truth).groupby(truth).mean().values,
        "Siamese accuracy": pd.Series(s1 == truth).groupby(truth).mean().values})
    per_writer.to_csv(RESULTS / "comparison_per_writer.csv", index=False)

    sess = []
    for name, rows in [("session 4 (validation)", va), ("session 5 (test)", te)]:
        cc, _ = cnn_predict(rows)
        ss, _ = identify(emb_sia[rows], proto_s, pw_s)
        sess.append({"session": name, "CNN": (cc == y[rows]).mean(),
                     "Siamese": (ss == y[rows]).mean()})
    pd.DataFrame(sess).to_csv(RESULTS / "comparison_sessions.csv", index=False)


def run_all(force=False):
    for step in (run_cnn, run_siamese_folds, run_contrastive_folds, run_hpo, run_comparison):
        start = time.time()
        print(f"{step.__name__} ...", flush=True)
        step(force=force)
        print(f"{step.__name__} done in {(time.time() - start) / 60:.1f} min", flush=True)
