"""Classifier two-sample test (C2ST) for distributional realism on HeartSteps.

A trajectory is summarised by transferable features (variability, lag-one
autocorrelation, volatility, large-jump rate, and per-action mean lifts). Real and
simulated trajectories are mean-shifted to a common level -- so the test probes
the *shape* of behaviour rather than an overall offset -- and a logistic-ridge
classifier is trained with patient-disjoint cross-validation to tell them apart.
An accuracy near 0.5 means the simulator's behaviour is indistinguishable from
the real data; an accuracy well above 0.5 means it is not.

The classifier, features, fold splitting and significance test match the
evaluation used for the reported experiments.
"""
import math
import random
import statistics
from math import comb


def extract_patient_series(patient):
    """Return (real, sim, prob, actions) aligned step series for one patient."""
    real, sim, prob, actions = [], [], [], []
    for step in patient.get("steps") or []:
        if step.get("gt_override"):
            continue
        gt = step.get("gt_adherence")
        if gt is None:
            continue
        real.append(float(gt))
        sm = step.get("adherence")
        sim.append(float(sm) if sm is not None else None)
        pr = step.get("adherence_prob")
        prob.append(float(pr) if pr is not None else None)
        actions.append(step.get("action_flat"))
    return real, sim, prob, actions


def autocorr1(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n < 2:
        return None
    mean = sum(xs) / n
    num = sum((xs[i] - mean) * (xs[i - 1] - mean) for i in range(1, n))
    den = sum((v - mean) ** 2 for v in xs)
    return num / den if den else None


def action_vocab(records, limit=6):
    """The most frequent action labels (appearing at least 5 times)."""
    counts = {}
    for _, _, _, actions in records:
        for action in actions:
            if action is None:
                continue
            key = str(action)
            counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [action for action, count in ordered[:limit] if count >= 5]


def _safe_mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def trajectory_features(series, acts=None, vocab=None, include_mean=True):
    """Transferable summary features of one trajectory (None if too short)."""
    s = [v for v in series if v is not None]
    if len(s) < 5:
        return None
    n = len(s)
    mean = sum(s) / n
    std = (sum((v - mean) ** 2 for v in s) / n) ** 0.5
    ac = autocorr1(s) or 0.0
    diffs = [abs(s[i] - s[i - 1]) for i in range(1, n)]
    volatility = sum(diffs) / len(diffs) if diffs else 0.0
    # fraction of day-to-day changes larger than 0.05 (in either direction)
    jump_rate = sum(1 for d in diffs if d > 0.05) / len(diffs) if diffs else 0.0
    feats = [std, ac, volatility, jump_rate]

    if acts and vocab:
        paired = [(v, str(a)) for v, a in zip(series, acts) if v is not None and a is not None]
        for action in vocab:
            vals = [v for v, a in paired if a == action]
            action_mean = _safe_mean(vals)
            feats.append((action_mean - mean) if action_mean is not None and len(vals) >= 3 else 0.0)

    return ([mean] + feats) if include_mean else feats


def residual_features(records):
    """Build mean-shifted (residual) feature vectors for the real and simulated series."""
    vocab = action_vocab(records)
    real_feats, sim_feats = [], []
    for real, sim, _prob, actions in records:
        real_valid = [v for v in real if v is not None]
        sim_valid = [v for v in sim if v is not None]
        if len(real_valid) < 5 or len(sim_valid) < 5:
            continue
        shift = (sum(real_valid) / len(real_valid)) - (sum(sim_valid) / len(sim_valid))
        sim_shifted = [min(1.0, max(0.0, v + shift)) if v is not None else None for v in sim]
        fr = trajectory_features(real, actions, vocab, include_mean=False)
        fs = trajectory_features(sim_shifted, actions, vocab, include_mean=False)
        if fr is not None and fs is not None:
            real_feats.append(fr)
            sim_feats.append(fs)
    return real_feats, sim_feats


def _standardize_train_test(train, test):
    if not train or not test:
        return None, None
    dim = len(train[0])
    means = [sum(x[j] for x in train) / len(train) for j in range(dim)]
    stds = []
    for j in range(dim):
        var = sum((x[j] - means[j]) ** 2 for x in train) / len(train)
        stds.append(var ** 0.5 if var > 1e-12 else 1.0)
    z_train = [[(x[j] - means[j]) / stds[j] for j in range(dim)] for x in train]
    z_test = [[(x[j] - means[j]) / stds[j] for j in range(dim)] for x in test]
    return z_train, z_test


def _fit_logistic_ridge(X, y, steps=180, lr=0.08, ridge=0.05):
    if not X:
        return None
    dim = len(X[0])
    w = [0.0] * (dim + 1)
    n = len(X)
    for _ in range(steps):
        grad = [0.0] * (dim + 1)
        for xi, yi in zip(X, y):
            z = w[0] + sum(w[j + 1] * xi[j] for j in range(dim))
            if z >= 0:
                p = 1.0 / (1.0 + math.exp(-z))
            else:
                ez = math.exp(z)
                p = ez / (1.0 + ez)
            err = p - yi
            grad[0] += err
            for j in range(dim):
                grad[j + 1] += err * xi[j]
        w[0] -= lr * grad[0] / n
        for j in range(dim):
            w[j + 1] -= lr * ((grad[j + 1] / n) + ridge * w[j + 1])
    return w


def _predict_logistic(w, X):
    out = []
    dim = len(w) - 1
    for xi in X:
        z = w[0] + sum(w[j + 1] * xi[j] for j in range(dim))
        if z >= 0:
            out.append(1.0 / (1.0 + math.exp(-z)))
        else:
            ez = math.exp(z)
            out.append(ez / (1.0 + ez))
    return out


def c2st_cv(real_feats, sim_feats, folds=5):
    """Cross-validated C2ST accuracy with per-fold spread and a binomial p-value.

    Returns a dict with the pooled held-out accuracy, the per-fold mean and sd,
    and a one-sided binomial test of accuracy > 0.5 (None if too few patients).
    """
    n = min(len(real_feats), len(sim_feats))
    if n < 10:
        return None
    labels = [1] * n + [0] * n
    n_folds = max(2, min(folds, n))
    per_fold, pooled_true, pooled_pred = [], [], []
    for fold in range(n_folds):
        test_idx = [i for i in range(n) if i % n_folds == fold]
        train_idx = [i for i in range(n) if i % n_folds != fold]
        train_X = [real_feats[i] for i in train_idx] + [sim_feats[i] for i in train_idx]
        train_y = [labels[i] for i in train_idx] + [labels[n + i] for i in train_idx]
        test_X = [real_feats[i] for i in test_idx] + [sim_feats[i] for i in test_idx]
        test_y = [labels[i] for i in test_idx] + [labels[n + i] for i in test_idx]
        z_train, z_test = _standardize_train_test(train_X, test_X)
        if z_train is None or z_test is None:
            continue
        model = _fit_logistic_ridge(z_train, train_y)
        if model is None:
            continue
        pred = _predict_logistic(model, z_test)
        acc = sum((p >= 0.5) == (y == 1) for p, y in zip(pred, test_y)) / len(test_y)
        per_fold.append(acc)
        pooled_true += test_y
        pooled_pred += pred
    if not pooled_true:
        return None

    correct = sum((p >= 0.5) == (y == 1) for p, y in zip(pooled_pred, pooled_true))
    total = len(pooled_true)
    pooled = correct / total
    p_binom = sum(comb(total, i) for i in range(correct, total + 1)) / (2 ** total)
    # "gap": mean distance of the held-out predictions from 0.5; a decision-free
    # measure of separability (0 = indistinguishable), used for the cross-dataset
    # and emission-parameter tables.
    gap = sum(abs(p - 0.5) for p in pooled_pred) / total
    return {
        "pooled": round(pooled, 3),
        "mean": round(statistics.mean(per_fold), 3),
        "sd": round(statistics.pstdev(per_fold), 3) if len(per_fold) > 1 else 0.0,
        "folds": len(per_fold),
        "per_fold": [round(x, 3) for x in per_fold],
        "n_items": total,
        "k": correct,
        "p_binom": p_binom,
        "gap": round(gap, 3),
    }


# --- descriptive between-patient metrics ---------------------------------
# These accompany the C2ST in the realism tables: the rank correlation of
# per-patient mean activity (does the simulator preserve who is more active?),
# the heterogeneity ratio (does it preserve the spread between patients?), the
# lag-one autocorrelation error, and the pooled mean activity.

def _rank(xs):
    """Average ranks (ties share the mean rank), 1-based."""
    pairs = sorted(((v, i) for i, v in enumerate(xs)), key=lambda p: p[0])
    out = [0.0] * len(xs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[pairs[k][1]] = avg
        i = j + 1
    return out


def _pearson(a, b):
    n = len(a)
    if n < 2:
        return None
    mean_a, mean_b = sum(a) / n, sum(b) / n
    num = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    var_a = sum((a[i] - mean_a) ** 2 for i in range(n))
    var_b = sum((b[i] - mean_b) ** 2 for i in range(n))
    if var_a <= 0 or var_b <= 0:
        return None
    return num / (var_a * var_b) ** 0.5


def _spearman(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    return _pearson(_rank(list(xs)), _rank(list(ys)))


def _std(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n < 2:
        return None
    mean = sum(xs) / n
    return (sum((x - mean) ** 2 for x in xs) / n) ** 0.5


def _patient_descriptives(records):
    real_means, sim_means, autocorr_errs = [], [], []
    for real, sim, _prob, _acts in records:
        real_valid = [v for v in real if v is not None]
        sim_valid = [v for v in sim if v is not None]
        if len(real_valid) < 5 or len(sim_valid) < 5:
            continue
        real_means.append(sum(real_valid) / len(real_valid))
        sim_means.append(sum(sim_valid) / len(sim_valid))
        ac_real, ac_sim = autocorr1(real_valid), autocorr1(sim_valid)
        autocorr_errs.append(abs(ac_sim - ac_real) if ac_real is not None and ac_sim is not None else None)
    return real_means, sim_means, autocorr_errs


def descriptive_metrics(records, n_boot=2000, seed=0):
    """Bootstrap the between-patient metrics over patients (None if too few)."""
    real_means, sim_means, autocorr_errs = _patient_descriptives(records)
    n = len(real_means)
    if n < 5:
        return None

    rng = random.Random(seed)
    spearmans, heterogeneities, autocorrs = [], [], []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boot_real = [real_means[i] for i in idx]
        boot_sim = [sim_means[i] for i in idx]
        rho = _spearman(boot_real, boot_sim)
        if rho is not None:
            spearmans.append(rho)
        sd_real, sd_sim = _std(boot_real), _std(boot_sim)
        if sd_real and sd_real > 0 and sd_sim is not None:
            heterogeneities.append(sd_sim / sd_real)
        valid = [autocorr_errs[i] for i in idx if autocorr_errs[i] is not None]
        if valid:
            autocorrs.append(sum(valid) / len(valid))

    def mean_sd(values):
        if len(values) > 1:
            return round(statistics.mean(values), 3), round(statistics.pstdev(values), 3)
        return (round(values[0], 3) if values else None), 0.0

    pooled_real = [v for real, _, _, _ in records for v in real if v is not None]
    pooled_sim = [v for _, sim, _, _ in records for v in sim if v is not None]
    return {
        "spearman": mean_sd(spearmans),
        "heterogeneity": mean_sd(heterogeneities),
        "autocorr_err": mean_sd(autocorrs),
        "mean_real": round(sum(pooled_real) / len(pooled_real), 3) if pooled_real else None,
        "mean_sim": round(sum(pooled_sim) / len(pooled_sim), 3) if pooled_sim else None,
        "n_pat": n,
    }
