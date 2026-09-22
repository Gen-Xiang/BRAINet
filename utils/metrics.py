import argparse

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
import pandas as pd

from dataset.dataset import save_json


PROBS = ["prob_class_0", "prob_class_1", "prob_class_2"]


def validate_predictions(frame, patient_level=False):
    required = {"subject_id", "true_label", "fold", *PROBS}
    if not required <= set(frame) or frame.empty:
        raise ValueError(f"Expected columns {sorted(required)}")
    if frame[list(required)].isna().any().any() or not frame.true_label.isin([0, 1, 2]).all():
        raise ValueError("Missing values or invalid labels")
    p = frame[PROBS].to_numpy(float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid probability vectors")
    if (frame.groupby("subject_id").true_label.nunique() != 1).any() or (frame.groupby("subject_id").fold.nunique() != 1).any():
        raise ValueError("Conflicting diagnosis or subject appears in multiple test folds")
    if patient_level and not frame.subject_id.is_unique:
        raise ValueError("Expected one OOF row per patient")
    if "epoch_index" in frame and frame.epoch_index.duplicated().any():
        raise ValueError("Duplicate OOF epochs")


def aggregate(frame):
    validate_predictions(frame)
    grouped = frame.groupby(["fold", "subject_id", "true_label"], sort=True)
    result = grouped[PROBS].mean().reset_index()
    result["n_epochs"] = grouped.size().to_numpy()
    result["pred_label"] = result[PROBS].to_numpy().argmax(1)
    validate_predictions(result, True)
    return result


def metrics(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    tp = cm.diagonal(); fn = cm.sum(1) - tp; fp = cm.sum(0) - tp
    tn = cm.sum() - tp - fn - fp
    recall = np.divide(tp, tp + fn, out=np.full(3, np.nan), where=tp + fn > 0)
    specificity = np.divide(tn, tn + fp, out=np.full(3, np.nan), where=tn + fp > 0)
    auc = float(roc_auc_score(y, p, labels=[0, 1, 2], multi_class="ovr", average="macro")) if len(np.unique(y)) == 3 else None
    return dict(balanced_accuracy=float(np.nanmean(recall)), macro_precision=float(precision_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
                weighted_recall=float(recall_score(y, pred, average="weighted", zero_division=0)),
                macro_specificity=float(np.nanmean(specificity)), macro_f1=float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
                weighted_f1=float(f1_score(y, pred, average="weighted", zero_division=0)), auc=auc,
                brier_score=float(np.square(p - np.eye(3)[y]).sum(1).mean()), confusion_matrix=cm.tolist(),
                per_class_recall=recall.tolist(), per_class_specificity=specificity.tolist())


def summarize(frame, protocol="fivefold"):
    patient = aggregate(frame)
    if protocol == "fivefold":
        if frame.fold.nunique() != 5 or any(set(g.true_label) != {0, 1, 2} for _, g in frame.groupby("fold")):
            raise ValueError("Expected five completed test folds, each containing all three classes")
    elif protocol == "loso":
        if any(g.subject_id.nunique() != 1 for _, g in frame.groupby("fold")):
            raise ValueError("Each LOSO test fold must contain one patient")
    else:
        raise ValueError("Unknown evaluation protocol")
    result = {"protocol": protocol, "std_ddof": 0, "n_subjects": len(patient), "n_epochs": len(frame)}
    for level, data in (("epoch", frame), ("subject", patient)):
        pooled = metrics(data.true_label, data[PROBS])
        result[level] = {"pooled": pooled}
        if protocol == "fivefold":
            fold_metrics = {str(f): metrics(g.true_label, g[PROBS]) for f, g in data.groupby("fold")}
            result[level]["folds"] = fold_metrics
            result[level]["fold_mean_sd"] = {
                key: {"mean": float(np.mean([m[key] for m in fold_metrics.values()])),
                      "sd": float(np.std([m[key] for m in fold_metrics.values()], ddof=0))}
                for key, value in pooled.items() if isinstance(value, (float, int)) and all(m[key] is not None for m in fold_metrics.values())}
    return result


def main():
    p = argparse.ArgumentParser(description='Evaluate epoch and subject predictions')
    p.add_argument("predictions"); p.add_argument("output"); p.add_argument("--protocol", choices=["fivefold", "loso"], default="fivefold")
    a = p.parse_args()
    frame = pd.read_csv(a.predictions, dtype={"subject_id": str})
    save_json(a.output, summarize(frame, a.protocol))


if __name__ == "__main__":
    main()
