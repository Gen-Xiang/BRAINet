from pathlib import Path
import hashlib
import json

from sklearn.model_selection import StratifiedKFold
import numpy as np

from utils.channels import CHANNELS


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_dataset(path):
    with np.load(path, allow_pickle=False) as data:
        x, y, ids = data["x"], data["y"], data["subject_id"].astype(str)
        if data["channels"].tolist() != CHANNELS or float(data["sfreq"]) != 250:
            raise ValueError("Input must have 250 Hz sampling and the exact documented channel order")
        apen = data["apen"] if "apen" in data else None
    if x.ndim != 3 or x.shape[1:] != (63, 500) or not len(x):
        raise ValueError("Expected nonempty (epochs, 63, 500) array")
    if y.shape != (len(x),) or ids.shape != y.shape or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Invalid labels or subject IDs")
    if not np.isfinite(x).all() or np.any(ids == ""):
        raise ValueError("Nonfinite EEG or empty subject IDs")
    if apen is not None and (apen.shape != (len(x), 63) or not np.isfinite(apen).all()):
        raise ValueError("Invalid ApEn")
    for sid in np.unique(ids):
        if len(np.unique(y[ids == sid])) != 1:
            raise ValueError(f"Conflicting labels for patient {sid}")
    return x.astype(np.float32), y.astype(np.int64), ids, apen


def epoch_zscore(x):
    mean = x.mean(-1, keepdims=True, dtype=np.float64)
    std = x.std(-1, keepdims=True, dtype=np.float64)
    return ((x - mean) / np.where(std > 0, std, 1)).astype(np.float32)


def make_splits(y, ids, seed=666, protocol="fivefold"):
    patients = np.unique(ids)
    labels = np.array([y[ids == sid][0] for sid in patients])
    if protocol == "loso":
        pairs = [(np.flatnonzero(patients != sid), np.flatnonzero(patients == sid)) for sid in patients]
    elif protocol == "fivefold":
        if any((labels == c).sum() < 5 for c in range(3)):
            raise ValueError("Five-fold stratification needs at least five patients per class")
        pairs = StratifiedKFold(5, shuffle=True, random_state=seed).split(patients, labels)
    else:
        raise ValueError(protocol)
    folds = [{"fold": n, "train": patients[tr].tolist(), "test": patients[te].tolist()}
             for n, (tr, te) in enumerate(pairs, 1)]
    validate_splits(folds, y, ids, protocol)
    return folds


def validate_splits(folds, y, ids, protocol):
    universe = set(ids)
    if len(folds) != (5 if protocol == "fivefold" else len(universe)):
        raise ValueError("Unexpected number of folds")
    seen = []
    for n, f in enumerate(folds, 1):
        tr, te = set(f["train"]), set(f["test"])
        if f["fold"] != n or len(tr) != len(f["train"]) or len(te) != len(f["test"]):
            raise ValueError("Invalid fold number or duplicate IDs")
        if not tr or not te or tr & te or tr | te != universe:
            raise ValueError("Patient leakage or incomplete fold coverage")
        if set(y[np.isin(ids, list(tr))]) != {0, 1, 2}:
            raise ValueError("Every training fold must contain all three classes")
        if protocol == "fivefold" and set(y[np.isin(ids, list(te))]) != {0, 1, 2}:
            raise ValueError("Every five-fold test set must contain all three classes")
        if protocol == "loso" and len(te) != 1:
            raise ValueError("LOSO must hold out exactly one patient")
        seen.extend(te)
    if len(seen) != len(universe) or set(seen) != universe:
        raise ValueError("Each patient must occur exactly once in OOF predictions")


def save_json(path, value):
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple, np.ndarray)):
            return [clean(x) for x in v]
        if isinstance(v, np.generic):
            return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
